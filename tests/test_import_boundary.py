"""Import-boundary contract (fleet RTM-001).

Two things are being protected here. The fleet rule: a rejected request must stop before model
libraries are imported. And, specific to this profile: the upstream checkpoint is a **pickle**, so the
digest check must complete before anything unpickles it — a snapshot that fails verification must
never reach `torch.load`.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from yolox_detection_pipeline.pipeline import (
    CHECKPOINT_FILE,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    YoloxDetectionPipeline,
)

PAYLOAD = b"stand-in for yolox_s.pth; never unpickled by these tests"


def _snapshot(root, *, tamper: bool = False) -> None:
    (root / CHECKPOINT_FILE).write_bytes(PAYLOAD)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": CHECKPOINT_FILE,
                "bytes": len(PAYLOAD),
                "sha256": "0" * 64 if tamper else hashlib.sha256(PAYLOAD).hexdigest(),
            }
        ],
        "totalBytes": len(PAYLOAD),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_from_pretrained_refuses_without_a_manifest_before_model_imports(
    tmp_path, forbid_model_imports
) -> None:
    with pytest.raises(FileNotFoundError, match="no manifest at"):
        YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_a_missing_checkpoint_before_model_imports(
    tmp_path, forbid_model_imports
) -> None:
    _snapshot(tmp_path)
    (tmp_path / CHECKPOINT_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_a_tampered_checkpoint_before_it_is_unpickled(
    tmp_path, forbid_model_imports
) -> None:
    _snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_an_out_of_contract_vocabulary_before_model_imports(
    tmp_path, forbid_model_imports
) -> None:
    _snapshot(tmp_path)
    with pytest.raises(ValueError, match="class_names must hold"):
        YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, class_names=[])


def test_from_pretrained_reaches_the_model_import_only_after_verification(
    tmp_path, forbid_model_imports
) -> None:
    _snapshot(tmp_path)
    with pytest.raises(AssertionError, match="model dependency imported before rejection"):
        YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_load_artifact_rejects_a_foreign_format_before_building_a_model(tmp_path) -> None:
    """Artifacts are pickles too. The format tag and the pinned identity are checked on the way in."""
    import torch

    path = tmp_path / "artifact.pt"
    torch.save({"format": "something-else/1", "class_names": ["a"], "state_dict": {}}, path)
    with pytest.raises(ValueError, match="artifact format"):
        YoloxDetectionPipeline.load_artifact(path)


def test_load_artifact_rejects_the_sibling_variants_artifact(tmp_path) -> None:
    """YOLOX-S and YOLOX-X are assets of the same upstream release, so they share MODEL_ID and
    MODEL_REVISION; only MODEL_KEY tells them apart. Without that check a sibling's adapter passes
    the identity check and fails afterwards inside load_state_dict with a tensor-shape error."""
    import torch

    from yolox_detection_pipeline.pipeline import ARTIFACT_FORMAT, MODEL_KEY

    other = "yolox-x" if MODEL_KEY == "yolox-s" else "yolox-s"
    path = tmp_path / "sibling.pt"
    torch.save(
        {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": other,
            "class_names": ["a"],
            "state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="variant"):
        YoloxDetectionPipeline.load_artifact(path)


def test_load_artifact_rejects_an_artifact_built_on_another_checkpoint(tmp_path) -> None:
    import torch

    from yolox_detection_pipeline.pipeline import ARTIFACT_FORMAT

    path = tmp_path / "artifact.pt"
    torch.save(
        {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": "0.3.0",
            "class_names": ["a"],
            "state_dict": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="package pins"):
        YoloxDetectionPipeline.load_artifact(path)
