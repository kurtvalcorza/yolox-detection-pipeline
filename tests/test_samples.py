"""The in-code sample data: determinism, exact reference boxes, and the two-vocabulary boundary.

The tutorial's whole evidence chain rests on these boxes being right. A reference box that is looser
than the object it names silently deflates every IoU and AP number downstream, so the geometry is
checked against the rendered pixels rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from yolox_detection_pipeline.pipeline import INPUT_SIZE, LABELS, MAX_LABELS_PER_IMAGE
from yolox_detection_pipeline.samples import (
    SCENE_SIZE,
    SIGN_CLASSES,
    blank_scene,
    noise_scene,
    sign_dataset,
    tutorial_scene,
)


def _ink_bbox(image: Image.Image, background: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Bounding box of every pixel that differs from the flat background, in xyxy."""
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    mask = np.abs(array - np.array(background, dtype=np.int16)).sum(axis=2) > 12
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# --- tutorial scene ---------------------------------------------------------------------------


def test_tutorial_scene_is_deterministic() -> None:
    first, refs_a = tutorial_scene()
    second, refs_b = tutorial_scene()
    assert first.tobytes() == second.tobytes()
    assert refs_a == refs_b


def test_tutorial_scene_is_drawn_at_the_model_input_size() -> None:
    """640x640 is what keeps the letterbox a no-op, so the default path never touches a resize."""
    image, _ = tutorial_scene()
    assert image.size == SCENE_SIZE == INPUT_SIZE


def test_tutorial_scene_references_are_all_coco_labels() -> None:
    _image, references = tutorial_scene()
    assert set(references) == {"stop sign", "traffic light", "clock", "sports ball", "bench"}
    for label in references:
        assert label in LABELS


def test_tutorial_scene_reference_boxes_lie_inside_the_canvas() -> None:
    image, references = tutorial_scene()
    width, height = image.size
    for label, boxes in references.items():
        for x0, y0, x1, y1 in boxes:
            assert 0 <= x0 < x1 <= width, label
            assert 0 <= y0 < y1 <= height, label


def test_tutorial_scene_is_not_blank() -> None:
    image, _ = tutorial_scene()
    assert len(set(image.convert("RGB").getdata())) > 20


# --- degenerate inputs ------------------------------------------------------------------------


def test_blank_scene_is_uniform() -> None:
    assert set(blank_scene().getdata()) == {(255, 255, 255)}


def test_noise_scene_is_deterministic_per_seed_and_differs_across_seeds() -> None:
    assert noise_scene(0).tobytes() == noise_scene(0).tobytes()
    assert noise_scene(0).tobytes() != noise_scene(1).tobytes()


def test_noise_scene_is_the_model_input_size() -> None:
    assert noise_scene(0).size == INPUT_SIZE


# --- adaptation dataset -----------------------------------------------------------------------


def test_sign_dataset_is_deterministic_per_seed() -> None:
    a, b = sign_dataset(4, seed=5), sign_dataset(4, seed=5)
    assert [r["labels"] for r in a] == [r["labels"] for r in b]
    assert [r["boxes"] for r in a] == [r["boxes"] for r in b]
    assert [r["image"].tobytes() for r in a] == [r["image"].tobytes() for r in b]


def test_sign_dataset_differs_across_seeds() -> None:
    a, b = sign_dataset(4, seed=0), sign_dataset(4, seed=99)
    assert [r["image"].tobytes() for r in a] != [r["image"].tobytes() for r in b]


def test_sign_dataset_record_shape() -> None:
    for record in sign_dataset(5, seed=2):
        assert set(record) == {"image", "boxes", "labels"}
        assert record["image"].size == INPUT_SIZE
        assert len(record["boxes"]) == len(record["labels"])
        assert 1 <= len(record["boxes"]) <= MAX_LABELS_PER_IMAGE


def test_sign_dataset_labels_are_the_sign_vocabulary_not_coco() -> None:
    """The two vocabularies must not be conflated: a fine-tuned model answers in these names only,
    and none of them is the COCO class 'stop sign' (note the space)."""
    labels = {label for record in sign_dataset(12, seed=0) for label in record["labels"]}
    assert labels <= set(SIGN_CLASSES)
    assert not (labels & set(LABELS))
    assert "stop sign" in LABELS and "stop sign" not in SIGN_CLASSES


def test_sign_dataset_boxes_are_inside_the_canvas() -> None:
    width, height = INPUT_SIZE
    for record in sign_dataset(20, seed=0):
        for x0, y0, x1, y1 in record["boxes"]:
            assert 0 <= x0 < x1 <= width
            assert 0 <= y0 < y1 <= height


def test_sign_dataset_single_object_box_is_tight_around_the_drawn_sign() -> None:
    """Against the rendered pixels, not against the drawing code: the reference box must contain the
    sign and hug it. The post below the sign is deliberately excluded from the box, so the measured
    ink extends further down than the box — only the top, left and right edges are checked tightly."""
    records = [r for r in sign_dataset(30, seed=0) if len(r["boxes"]) == 1]
    assert records, "expected at least one single-object image in 30 draws"
    for record in records[:5]:
        x0, y0, x1, y1 = record["boxes"][0]
        background = record["image"].getpixel((5, 5))
        ix0, iy0, ix1, _iy1 = _ink_bbox(record["image"].crop((0, 0, 640, 460)), background)
        assert abs(ix0 - x0) <= 3, (record["labels"], ix0, x0)
        assert abs(iy0 - y0) <= 3, (record["labels"], iy0, y0)
        assert abs(ix1 - x1) <= 3, (record["labels"], ix1, x1)


def test_sign_dataset_covers_every_class_over_a_reasonable_draw() -> None:
    labels = [label for record in sign_dataset(40, seed=0) for label in record["labels"]]
    for name in SIGN_CLASSES:
        assert labels.count(name) >= 5, f"{name} appears {labels.count(name)} times in 40 images"


def test_sign_dataset_rejects_out_of_range_requests() -> None:
    with pytest.raises(ValueError, match="n_images must be"):
        sign_dataset(0)
    with pytest.raises(ValueError, match="max_objects must be"):
        sign_dataset(2, max_objects=9)
