"""The model is served from the pinned safetensors conversion; the pickle source is read only to convert it.

The conversion tests swap the YOLOX architecture for a two-layer stand-in, so they run offline in CI. The last
test runs only where the real checkpoint and its conversion are staged under weights/.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from yolox_detection_pipeline import pipeline as pipeline_module
from yolox_detection_pipeline.pipeline import (
    CHECKPOINT_FILE,
    CONVERTED_FILE,
    CONVERTED_SHA256,
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    YoloxDetectionPipeline,
)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_converted_file_is_pinned_safetensors() -> None:
    assert CONVERTED_FILE.endswith(".safetensors")
    assert len(CONVERTED_SHA256) == 64 and all(c in "0123456789abcdef" for c in CONVERTED_SHA256)
    assert CONVERTED_FILE.removesuffix(".safetensors") == CHECKPOINT_FILE.removesuffix(".pth")


def test_validator_declares_the_converted_digest() -> None:
    path = Path(__file__).resolve().parents[1] / "tools" / "validate_release_assets.py"
    spec = importlib.util.spec_from_file_location("validate_release_assets", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    assert set(validator.EXTERNAL_WEIGHT_DIGESTS) == {CONVERTED_SHA256}


def test_serving_path_never_unpickles() -> None:
    source = inspect.getsource(YoloxDetectionPipeline.from_pretrained)
    assert "torch.load(" not in source and "load_file(" in source
    assert "torch.load(" not in inspect.getsource(YoloxDetectionPipeline.load_artifact)


def test_altered_conversion_is_refused_before_model_imports(tmp_path, forbid_model_imports) -> None:
    (tmp_path / CONVERTED_FILE).write_bytes(b"not the pinned conversion")
    with pytest.raises(ValueError, match="sha256"):
        YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """A manifest-pinned two-layer 'release checkpoint' plus the digest its conversion must have."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    def build(num_classes=None):
        return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))

    torch.manual_seed(0)
    state = build().state_dict()
    root = tmp_path / "weights"
    root.mkdir()
    # The release file's layout: the state dict under "model" beside training-only entries.
    torch.save({"model": state, "start_epoch": 300, "optimizer": {}, "amp": None}, root / CHECKPOINT_FILE)
    size = (root / CHECKPOINT_FILE).stat().st_size
    entry = {"path": CHECKPOINT_FILE, "bytes": size, "sha256": _sha(root / CHECKPOINT_FILE)}
    manifest = {"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": [entry], "totalBytes": size}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    reference = tmp_path / "reference.safetensors"
    pipeline_module._write_converted(state, reference)
    monkeypatch.setattr(pipeline_module, "CONVERTED_SHA256", _sha(reference))
    monkeypatch.setattr(pipeline_module, "_build_model", build)
    return torch, root, state


def test_conversion_is_byte_reproducible_and_strict(stand_in) -> None:
    _torch, root, state = stand_in
    first = pipeline_module.convert_checkpoint(root)
    (root / CONVERTED_FILE).unlink()
    second = pipeline_module.convert_checkpoint(root)
    assert first["converted"]["sha256"] == second["converted"]["sha256"] == pipeline_module.CONVERTED_SHA256
    assert first["tensors"] == len(state)
    assert not list(root.glob("*.partial"))


def test_conversion_with_another_digest_is_refused_and_leaves_nothing(stand_in, monkeypatch) -> None:
    _torch, root, _state = stand_in
    monkeypatch.setattr(pipeline_module, "CONVERTED_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        pipeline_module.convert_checkpoint(root)
    assert not (root / CONVERTED_FILE).exists() and not list(root.glob("*.partial"))


def test_from_pretrained_converts_once_then_serves_without_the_source(stand_in, monkeypatch) -> None:
    torch, root, state = stand_in
    first = YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=root)  # converts
    assert (root / CONVERTED_FILE).is_file()
    (root / CHECKPOINT_FILE).unlink()
    (root / MANIFEST_NAME).unlink()

    def refuse(*_args, **_kwargs):
        raise AssertionError("torch.load called on the serving path")

    monkeypatch.setattr(torch, "load", refuse)
    served = YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=root)
    weights = served.model.state_dict()
    assert set(weights) == set(state) and all(torch.equal(weights[k], state[k]) for k in state)
    assert served.base_state_digest == first.base_state_digest


@pytest.mark.skipif(
    not all((DEFAULT_WEIGHTS_DIR / name).is_file() for name in (CHECKPOINT_FILE, CONVERTED_FILE)),
    reason="real checkpoint and conversion not staged",
)
def test_real_conversion_equals_the_source_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    pipeline_module.verify_snapshot(DEFAULT_WEIGHTS_DIR)
    pipeline_module.verify_converted(DEFAULT_WEIGHTS_DIR)
    payload = torch.load(DEFAULT_WEIGHTS_DIR / CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    source = payload["model"]
    converted = load_file(str(DEFAULT_WEIGHTS_DIR / CONVERTED_FILE))
    assert set(source) == set(converted)
    assert all(torch.equal(source[k], converted[k]) and source[k].dtype == converted[k].dtype for k in source)
