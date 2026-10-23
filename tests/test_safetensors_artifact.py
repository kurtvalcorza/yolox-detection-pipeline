"""The exported adapter is a safetensors state dict plus a plain-JSON sidecar, never a pickle.

Covers the round trip (identical tensors and identical predictions), the digest check that runs before
the weights are opened, the refusal of `.pt` paths in both directions, and the sidecar being pure JSON.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from PIL import Image  # noqa: E402

from yolox_detection_pipeline import pipeline as pipeline_module  # noqa: E402
from yolox_detection_pipeline.pipeline import (  # noqa: E402
    ARTIFACT_FILE,
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    YoloxDetectionPipeline,
)

CLASSES = ("stop-sign", "yield-sign", "speed-limit-sign")


@pytest.fixture(scope="module")
def adapted_pipe() -> YoloxDetectionPipeline:
    torch.manual_seed(0)
    model = YoloxDetectionPipeline._build(len(CLASSES)).eval()
    return YoloxDetectionPipeline(
        model=model,
        device="cpu",
        class_names=CLASSES,
        source="test",
        base_state_digest="0" * 64,
        adapted=True,
    )


@pytest.fixture(scope="module")
def exported(adapted_pipe, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifact")
    descriptor = adapted_pipe.save_artifact(out / ARTIFACT_FILE, notes="unit test")
    return out, descriptor


def _copy_artifact(src_dir, dst_dir):
    for name in (ARTIFACT_FILE, ARTIFACT_FILE.replace(".safetensors", ".json")):
        (dst_dir / name).write_bytes((src_dir / name).read_bytes())
    return dst_dir / ARTIFACT_FILE


def test_artifact_file_is_safetensors() -> None:
    assert ARTIFACT_FILE.endswith(".safetensors")
    assert ARTIFACT_FORMAT.endswith("/2")


def test_save_writes_safetensors_and_json_sidecar_and_no_pickle(exported) -> None:
    out, descriptor = exported
    names = sorted(p.name for p in out.iterdir())
    sidecar_name = ARTIFACT_FILE.replace(".safetensors", ".json")
    assert names == sorted([ARTIFACT_FILE, sidecar_name])
    assert not any(p.suffix in {".pt", ".pth", ".bin", ".ckpt", ".partial"} for p in out.iterdir())
    # A safetensors file starts with an 8-byte little-endian header length followed by a JSON header;
    # a torch.save zip archive starts with "PK" and a raw pickle with 0x80.
    raw = (out / ARTIFACT_FILE).read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_len])
    assert header["__metadata__"]["format"] == ARTIFACT_FORMAT
    assert raw[:2] != b"PK" and raw[0] != 0x80
    assert descriptor["path"].endswith(ARTIFACT_FILE)
    assert descriptor["sidecar"].endswith(sidecar_name)


def test_sidecar_is_pure_json_with_the_weights_digest(exported) -> None:
    out, descriptor = exported
    sidecar_path = out / ARTIFACT_FILE.replace(".safetensors", ".json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))

    def plain(value) -> bool:
        if isinstance(value, dict):
            return all(isinstance(k, str) and plain(v) for k, v in value.items())
        if isinstance(value, list):
            return all(plain(v) for v in value)
        return value is None or isinstance(value, (str, int, float, bool))

    assert plain(payload)
    # Round-tripping through json gives the same object: nothing in it needed a custom encoder.
    assert json.loads(json.dumps(payload)) == payload
    assert payload["format"] == ARTIFACT_FORMAT
    assert payload["model_id"] == MODEL_ID and payload["model_revision"] == MODEL_REVISION
    assert payload["model_key"] == MODEL_KEY
    assert payload["class_names"] == list(CLASSES)
    assert payload["weights_file"] == ARTIFACT_FILE
    assert payload["weights_sha256"] == descriptor["sha256"]
    assert payload["weights_sha256"] == pipeline_module._sha256(out / ARTIFACT_FILE)
    assert "state_dict" not in payload


def test_round_trip_gives_identical_tensors_and_predictions(adapted_pipe, exported) -> None:
    out, _ = exported
    reloaded = YoloxDetectionPipeline.load_artifact(out / ARTIFACT_FILE, device="cpu")
    assert reloaded.class_names == CLASSES and reloaded.adapted
    before, after = adapted_pipe.model.state_dict(), reloaded.model.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert before[name].dtype == after[name].dtype, name
        assert torch.equal(before[name].cpu(), after[name].cpu()), name

    generator = torch.Generator().manual_seed(1)
    batch = torch.rand(1, 3, 128, 128, generator=generator)
    with torch.no_grad():
        assert torch.equal(adapted_pipe.model(batch), reloaded.model(batch))
    image = Image.new("RGB", (96, 80), (120, 40, 200))
    assert adapted_pipe.detect(image, threshold=0.01) == reloaded.detect(image, threshold=0.01)


def test_digest_mismatch_is_refused_before_the_weights_are_opened(exported, tmp_path, monkeypatch) -> None:
    out, _ = exported
    weights = _copy_artifact(out, tmp_path)
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))

    import safetensors.torch

    def never(*_args, **_kwargs):
        raise AssertionError("safetensors.torch.load_file reached despite a digest mismatch")

    monkeypatch.setattr(safetensors.torch, "load_file", never)
    monkeypatch.setattr(YoloxDetectionPipeline, "_build", staticmethod(never))
    with pytest.raises(ValueError, match="digest mismatch"):
        YoloxDetectionPipeline.load_artifact(weights)


def test_missing_or_digestless_sidecar_is_refused(exported, tmp_path) -> None:
    out, _ = exported
    weights = _copy_artifact(out, tmp_path)
    sidecar = weights.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.pop("weights_sha256")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="weights_sha256"):
        YoloxDetectionPipeline.load_artifact(weights)
    sidecar.unlink()
    with pytest.raises(ValueError, match="sidecar"):
        YoloxDetectionPipeline.load_artifact(weights)


def test_pickle_paths_are_refused_on_save_and_on_load(adapted_pipe, tmp_path) -> None:
    with pytest.raises(ValueError, match="safetensors"):
        adapted_pipe.save_artifact(tmp_path / "adapter.pt")
    assert not any(tmp_path.iterdir())
    legacy = tmp_path / "legacy-adapter.pt"
    torch.save({"format": "dimer-yolox-detection-adapter/1", "state_dict": {}}, legacy)
    with pytest.raises(ValueError, match="pickle"):
        YoloxDetectionPipeline.load_artifact(legacy)
