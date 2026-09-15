"""Package contract: identity, snapshot handling, geometry, preprocessing and the output shape.

None of these tests load the 72 MB checkpoint; they pin the contract the notebook and the model card
both quote. The behaviour that needs real weights is recorded in docs/release-verification.md.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from yolox_detection_pipeline import pipeline as mod
from yolox_detection_pipeline.pipeline import (
    CHECKPOINT_FILE,
    INPUT_SIZE,
    LABELS,
    MANIFEST_NAME,
    MAX_DETECTIONS,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    PAD_VALUE,
    RELEASE_ASSET_URL,
    UPSTREAM_CODE_REVISION,
    YoloxDetectionPipeline,
    box_iou,
    preprocess,
    stage_missing_files,
    verify_snapshot,
)

PAYLOAD = b"not really a checkpoint, but it hashes"


def _snapshot(root, *, tamper: bool = False, missing: bool = False) -> None:
    if not missing:
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


# --- identity ---------------------------------------------------------------------------------


def test_identity_constants_are_the_pinned_release() -> None:
    assert MODEL_ID == "Megvii-BaseDetection/YOLOX"
    assert MODEL_REVISION == "0.1.1rc0"
    assert MODEL_KEY == "yolox-s"
    assert MODEL_LICENSE == "apache-2.0"
    assert CHECKPOINT_FILE == "yolox_s.pth"
    assert len(UPSTREAM_CODE_REVISION) == 40
    assert RELEASE_ASSET_URL.endswith(f"/releases/download/{MODEL_REVISION}/{CHECKPOINT_FILE}")


def test_committed_manifest_matches_the_package_identity() -> None:
    manifest = json.loads((mod.DEFAULT_WEIGHTS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["upstreamCodeRevision"] == UPSTREAM_CODE_REVISION
    assert manifest["sourceUrl"] == RELEASE_ASSET_URL
    assert [entry["path"] for entry in manifest["files"]] == [CHECKPOINT_FILE]
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_labels_are_the_eighty_coco_classes_in_checkpoint_order() -> None:
    assert len(LABELS) == 80
    assert LABELS[0] == "person"
    assert LABELS[-1] == "toothbrush"
    assert len(set(LABELS)) == 80


def test_anchor_count_is_the_detection_ceiling() -> None:
    # 80x80 + 40x40 + 20x20 at strides 8/16/32 over a 640x640 letterbox.
    assert MAX_DETECTIONS == 8400
    assert INPUT_SIZE == (640, 640)


# --- snapshot ---------------------------------------------------------------------------------


def test_verify_snapshot_accepts_a_matching_snapshot(tmp_path) -> None:
    _snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["model_id"] == MODEL_ID
    assert result["revision"] == MODEL_REVISION
    assert result["files"] == 1


def test_verify_snapshot_reports_a_digest_mismatch(tmp_path) -> None:
    _snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_reports_a_size_mismatch(tmp_path) -> None:
    _snapshot(tmp_path)
    (tmp_path / CHECKPOINT_FILE).write_bytes(PAYLOAD + b"extra")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_reports_a_missing_file(tmp_path) -> None:
    _snapshot(tmp_path, missing=True)
    with pytest.raises(FileNotFoundError, match="snapshot file missing"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_reports_a_missing_manifest(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_refuses_a_foreign_revision(tmp_path) -> None:
    _snapshot(tmp_path)
    path = tmp_path / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["revision"] = "0.3.0"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest revision"):
        verify_snapshot(tmp_path)


def test_stage_missing_files_is_a_no_op_when_complete(tmp_path) -> None:
    _snapshot(tmp_path)
    assert stage_missing_files(tmp_path, allow_download=True) == []


def test_stage_missing_files_refuses_to_download_unless_asked(tmp_path) -> None:
    _snapshot(tmp_path, missing=True)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path, allow_download=False)


def test_stage_missing_files_fetches_exactly_the_absent_entries(tmp_path) -> None:
    _snapshot(tmp_path, missing=True)
    calls: list[str] = []

    def fake(relative_path: str, root) -> None:
        calls.append(relative_path)
        (root / relative_path).write_bytes(PAYLOAD)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake) == [CHECKPOINT_FILE]
    assert calls == [CHECKPOINT_FILE]
    verify_snapshot(tmp_path)  # the staged file passes the digest check


def test_stage_missing_files_refuses_a_manifest_for_a_different_pin(tmp_path) -> None:
    _snapshot(tmp_path, missing=True)
    path = tmp_path / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["modelId"] = "someone-else/YOLOX"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


def test_from_pretrained_without_a_manifest_names_the_remedy(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no manifest at"):
        YoloxDetectionPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


# --- geometry ---------------------------------------------------------------------------------


def test_box_iou_identical_boxes() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_box_iou_disjoint_boxes() -> None:
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_box_iou_half_overlap() -> None:
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_box_iou_rejects_malformed_boxes() -> None:
    with pytest.raises(ValueError, match="x0 <= x1"):
        box_iou([10, 0, 0, 10], [0, 0, 10, 10])
    with pytest.raises(ValueError, match=r"\[x0, y0, x1, y1\]"):
        box_iou([0, 0, 10], [0, 0, 10, 10])


# --- preprocessing ----------------------------------------------------------------------------


def test_preprocess_square_input_is_a_pure_channel_swap() -> None:
    """The default sample path: a 640x640 image is letterboxed without any resize at all, so the
    tensor is exactly the source pixels with the channels reversed."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
    chw, ratio = preprocess(Image.fromarray(array))
    assert ratio == 1.0
    assert chw.shape == (3, 640, 640)
    assert chw.dtype == np.float32
    np.testing.assert_array_equal(chw, array.transpose(2, 0, 1)[::-1].astype(np.float32))


def test_preprocess_keeps_raw_0_255_values() -> None:
    """No /255 and no mean/std: upstream ValTransform(legacy=False) feeds raw pixel values."""
    chw, _ = preprocess(Image.new("RGB", (640, 640), (255, 128, 0)))
    assert chw[0].max() == 0.0  # blue channel first
    assert chw[1].max() == 128.0
    assert chw[2].max() == 255.0


def test_preprocess_letterboxes_a_wide_image_and_pads_the_rest() -> None:
    chw, ratio = preprocess(Image.new("RGB", (1280, 640), (10, 10, 10)))
    assert ratio == pytest.approx(0.5)
    # 1280x640 scales to 640x320, pasted top-left; everything below row 320 is the pad value.
    assert chw[:, :320, :].max() <= 10.0
    np.testing.assert_array_equal(chw[:, 320:, :], np.full((3, 320, 640), PAD_VALUE, dtype=np.float32))


def test_preprocess_ratio_maps_boxes_back_to_input_pixels() -> None:
    _chw, ratio = preprocess(Image.new("RGB", (1000, 500)))
    assert 100 / ratio == pytest.approx(100 / 0.64)


def test_preprocess_converts_non_rgb_modes() -> None:
    chw, _ = preprocess(Image.new("L", (640, 640), 200).convert("RGB"))
    assert chw.shape == (3, 640, 640)


# --- input ceilings ---------------------------------------------------------------------------


def test_image_side_ceilings_are_ordered() -> None:
    assert 0 < MIN_IMAGE_SIDE < MAX_IMAGE_SIDE
