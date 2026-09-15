"""Validation, split and evaluation helpers — the stages the notebook runs around the model.

These carry the tutorial's honesty: what the pipeline refuses, what it calls measurable, and what the
average-precision numbers mean. They run without the checkpoint.
"""

from __future__ import annotations

import pytest
from PIL import Image

from yolox_detection_pipeline.pipeline import (
    COCO_IOU_THRESHOLDS,
    DETECTION_THRESHOLD,
    INPUT_SCHEMA,
    MAX_CLASSES,
    MAX_EPOCHS,
    MAX_LABELS_PER_IMAGE,
    MAX_TRAIN_IMAGES,
    NMS_THRESHOLD,
    TRAIN_SCHEMA,
    average_precision,
    evaluation_report,
    split_records,
    validate_dataset,
    validate_image,
    validate_inputs,
)
from yolox_detection_pipeline.samples import SIGN_CLASSES, sign_dataset

IMAGE = Image.new("RGB", (640, 640), (240, 240, 240))


def _record(n_boxes: int = 1, label: str = "stop-sign") -> dict:
    boxes = [[10.0 + 60 * i, 10.0, 60.0 + 60 * i, 60.0] for i in range(n_boxes)]
    return {"image": IMAGE, "boxes": boxes, "labels": [label] * n_boxes}


# --- validate_inputs --------------------------------------------------------------------------


def test_validate_inputs_returns_the_input_manifest() -> None:
    manifest = validate_inputs(IMAGE, threshold=0.4, nms_threshold=0.5, names=["scene"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["threshold"] == 0.4
    assert manifest["nms_threshold"] == 0.5
    assert manifest["inputs"] == [{"id": "scene", "mode": "RGB", "size": [640, 640]}]
    assert manifest["schema"] == INPUT_SCHEMA


def test_validate_inputs_schema_names_both_thresholds_and_the_labels() -> None:
    assert INPUT_SCHEMA["threshold"] == [0.0, 1.0]
    assert INPUT_SCHEMA["nms_threshold"] == [0.0, 1.0]
    assert len(INPUT_SCHEMA["labels"]) == 80
    assert "BGR" in INPUT_SCHEMA["preprocessing"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": 1.4}, "threshold must be"),
        ({"threshold": -0.1}, "threshold must be"),
        ({"threshold": True}, "threshold must be"),
        ({"threshold": "0.3"}, "threshold must be"),
        ({"nms_threshold": 2.0}, "nms_threshold must be"),
        ({"nms_threshold": None}, "nms_threshold must be"),
    ],
)
def test_validate_inputs_rejects_out_of_contract_thresholds(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        validate_inputs(IMAGE, **kwargs)


def test_validate_inputs_rejects_a_non_image() -> None:
    with pytest.raises(TypeError, match="must be a PIL.Image.Image"):
        validate_inputs("a path", threshold=DETECTION_THRESHOLD)


def test_validate_inputs_rejects_more_than_one_name() -> None:
    with pytest.raises(ValueError, match="exactly one entry"):
        validate_inputs(IMAGE, names=["a", "b"])


def test_validate_image_rejects_sides_outside_the_ceilings() -> None:
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_image(Image.new("RGB", (8, 640)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_image(Image.new("RGB", (5000, 640)))


# --- validate_dataset -------------------------------------------------------------------------


def test_validate_dataset_returns_the_dataset_manifest() -> None:
    manifest = validate_dataset([_record(2), _record(1, "yield-sign")], SIGN_CLASSES, epochs=3)
    assert manifest["verdict"] == "accepted"
    assert manifest["n_images"] == 2
    assert manifest["n_boxes"] == 3
    assert manifest["boxes_per_class"] == {"stop-sign": 2, "yield-sign": 1, "speed-limit-sign": 0}
    assert manifest["schema"] == TRAIN_SCHEMA


def test_validate_dataset_flags_a_class_with_no_boxes_without_refusing() -> None:
    manifest = validate_dataset([_record(1)], SIGN_CLASSES)
    assert manifest["verdict"] == "accepted"
    assert len(manifest["findings"]) == 1
    assert "yield-sign" in manifest["findings"][0]
    assert "speed-limit-sign" in manifest["findings"][0]


def test_validate_dataset_accepts_the_shipped_sample() -> None:
    manifest = validate_dataset(sign_dataset(6, seed=3), SIGN_CLASSES)
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["n_boxes"] >= 6


@pytest.mark.parametrize(
    ("records", "class_names", "kwargs", "match"),
    [
        ([], SIGN_CLASSES, {}, "expected 1.."),
        ([_record()], (), {}, "at least one class"),
        ([_record()], ("a", "a"), {}, "must not repeat"),
        ([_record()], ("a", ""), {}, "non-empty string"),
        ([_record()], SIGN_CLASSES, {"epochs": 0}, "epochs must be"),
        ([_record()], SIGN_CLASSES, {"epochs": MAX_EPOCHS + 1}, "epochs must be"),
        ([_record()], SIGN_CLASSES, {"epochs": True}, "epochs must be"),
    ],
)
def test_validate_dataset_rejects_out_of_contract_requests(records, class_names, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        validate_dataset(records, class_names, **kwargs)


def test_validate_dataset_rejects_too_many_images() -> None:
    with pytest.raises(ValueError, match=f"1..{MAX_TRAIN_IMAGES}"):
        validate_dataset([_record()] * (MAX_TRAIN_IMAGES + 1), SIGN_CLASSES)


def test_validate_dataset_rejects_too_many_classes() -> None:
    with pytest.raises(ValueError, match="MAX_CLASSES"):
        validate_dataset([_record()], tuple(f"c{i}" for i in range(MAX_CLASSES + 1)))


def test_validate_dataset_rejects_a_label_outside_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="not in class_names"):
        validate_dataset([_record(1, "no-such-sign")], SIGN_CLASSES)


def test_validate_dataset_rejects_mismatched_boxes_and_labels() -> None:
    record = {"image": IMAGE, "boxes": [[0, 0, 10, 10], [0, 0, 20, 20]], "labels": ["stop-sign"]}
    with pytest.raises(ValueError, match="2 boxes but 1 labels"):
        validate_dataset([record], SIGN_CLASSES)


def test_validate_dataset_rejects_a_box_outside_the_image() -> None:
    record = {"image": IMAGE, "boxes": [[0, 0, 700, 10]], "labels": ["stop-sign"]}
    with pytest.raises(ValueError, match="falls outside"):
        validate_dataset([record], SIGN_CLASSES)


def test_validate_dataset_rejects_an_inverted_box() -> None:
    record = {"image": IMAGE, "boxes": [[50, 50, 10, 10]], "labels": ["stop-sign"]}
    with pytest.raises(ValueError, match="x0 < x1"):
        validate_dataset([record], SIGN_CLASSES)


def test_validate_dataset_rejects_more_boxes_than_the_head_can_carry() -> None:
    record = {
        "image": IMAGE,
        "boxes": [[0, 0, 5, 5]] * (MAX_LABELS_PER_IMAGE + 1),
        "labels": ["stop-sign"] * (MAX_LABELS_PER_IMAGE + 1),
    }
    with pytest.raises(ValueError, match="MAX_LABELS_PER_IMAGE"):
        validate_dataset([record], SIGN_CLASSES)


def test_validate_dataset_rejects_a_record_missing_a_key() -> None:
    with pytest.raises(ValueError, match=r"missing \['labels'\]"):
        validate_dataset([{"image": IMAGE, "boxes": []}], SIGN_CLASSES)


# --- split ------------------------------------------------------------------------------------


def test_split_records_is_deterministic_and_disjoint() -> None:
    records = sign_dataset(8, seed=1)
    train_a, held_a = split_records(records, holdout=0.25, seed=7)
    train_b, held_b = split_records(records, holdout=0.25, seed=7)
    assert [id(r) for r in train_a] == [id(r) for r in train_b]
    assert [id(r) for r in held_a] == [id(r) for r in held_b]
    assert len(train_a) + len(held_a) == len(records)
    assert not ({id(r) for r in train_a} & {id(r) for r in held_a})


def test_split_records_honours_the_holdout_fraction() -> None:
    train, held = split_records(sign_dataset(8, seed=1), holdout=0.25, seed=0)
    assert (len(train), len(held)) == (6, 2)


def test_split_records_rejects_a_holdout_that_leaves_no_training_data() -> None:
    with pytest.raises(ValueError, match="holdout must be in"):
        split_records(sign_dataset(4, seed=1), holdout=1.0)


# --- evaluation_report ------------------------------------------------------------------------


def _result(detections: list[dict]) -> dict:
    return {
        "detections": detections,
        "threshold": DETECTION_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "width": 640,
        "height": 640,
    }


def test_evaluation_report_is_not_measurable_without_references() -> None:
    report = evaluation_report(_result([{"label": "clock", "score": 0.9, "box": [0, 0, 10, 10]}]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert "labelled boxes" in report["needs"]


def test_evaluation_report_scores_same_label_box_iou() -> None:
    detections = [{"label": "clock", "score": 0.9, "box": [0, 0, 10, 10]}]
    report = evaluation_report(_result(detections), {"clock": [[0, 0, 10, 10]]})
    assert report["verdict"] == "sample-sanity"
    assert report["metrics"][0]["value"] == pytest.approx(1.0)
    assert report["metrics"][0]["matched_score"] == 0.9


def test_evaluation_report_records_a_miss_as_zero_rather_than_dropping_it() -> None:
    """A reference nothing detected must still appear, or the report would flatter the model."""
    detections = [{"label": "clock", "score": 0.9, "box": [0, 0, 10, 10]}]
    report = evaluation_report(_result(detections), {"sports ball": [[100, 100, 150, 150]]})
    assert report["metrics"][0]["value"] == 0.0
    assert report["metrics"][0]["matched_score"] is None
    assert report["metrics"][0]["n_detected_same_label"] == 0


def test_evaluation_report_does_not_match_across_labels() -> None:
    detections = [{"label": "kite", "score": 0.9, "box": [0, 0, 10, 10]}]
    report = evaluation_report(_result(detections), {"sports ball": [[0, 0, 10, 10]]})
    assert report["metrics"][0]["value"] == 0.0


def test_evaluation_report_rejects_an_unknown_reference_label() -> None:
    with pytest.raises(ValueError, match="unknown reference label"):
        evaluation_report(_result([]), {"not-a-coco-class": [[0, 0, 1, 1]]})


def test_evaluation_report_never_claims_a_benchmark() -> None:
    report = evaluation_report(_result([]), {"clock": [[0, 0, 10, 10]]})
    assert "not a detection benchmark" in report["reason"]
    assert report["baselines"] == []


# --- average_precision ------------------------------------------------------------------------


def _refs(*labelled: tuple[str, list[float]]) -> dict:
    return {"boxes": [box for _label, box in labelled], "labels": [label for label, _box in labelled]}


def test_average_precision_is_one_for_exact_predictions() -> None:
    references = [_refs(("stop-sign", [0, 0, 50, 50])), _refs(("yield-sign", [10, 10, 80, 80]))]
    predictions = [
        [{"label": "stop-sign", "score": 1.0, "box": [0, 0, 50, 50]}],
        [{"label": "yield-sign", "score": 1.0, "box": [10, 10, 80, 80]}],
    ]
    result = average_precision(predictions, references, SIGN_CLASSES)
    assert result["ap"] == pytest.approx(1.0)
    assert result["ap50"] == pytest.approx(1.0)
    assert result["scored_classes"] == ["stop-sign", "yield-sign"]


def test_average_precision_is_zero_without_predictions() -> None:
    references = [_refs(("stop-sign", [0, 0, 50, 50]))]
    result = average_precision([[]], references, SIGN_CLASSES)
    assert result["ap"] == 0.0
    assert result["ap50"] == 0.0


def test_average_precision_ignores_classes_with_no_references() -> None:
    """A class nobody labelled contributes no AP, as in COCO — otherwise an unused class would
    silently drag the mean to zero."""
    references = [_refs(("stop-sign", [0, 0, 50, 50]))]
    predictions = [[{"label": "stop-sign", "score": 1.0, "box": [0, 0, 50, 50]}]]
    result = average_precision(predictions, references, SIGN_CLASSES)
    assert result["scored_classes"] == ["stop-sign"]
    assert result["ap50"] == pytest.approx(1.0)


def test_average_precision_penalises_a_wrong_label() -> None:
    references = [_refs(("stop-sign", [0, 0, 50, 50]))]
    predictions = [[{"label": "yield-sign", "score": 1.0, "box": [0, 0, 50, 50]}]]
    assert average_precision(predictions, references, SIGN_CLASSES)["ap50"] == 0.0


def test_average_precision_penalises_a_false_positive_ranked_above_the_hit() -> None:
    """COCO semantics, and worth stating explicitly: a false positive scored *below* every true
    positive costs nothing, because interpolated precision is the maximum over higher recalls. Only a
    false positive ranked above a hit moves the number."""
    references = [_refs(("stop-sign", [0, 0, 50, 50]))]
    hit = {"label": "stop-sign", "score": 0.6, "box": [0, 0, 50, 50]}
    trailing = {"label": "stop-sign", "score": 0.4, "box": [300, 300, 350, 350]}
    leading = {"label": "stop-sign", "score": 0.9, "box": [300, 300, 350, 350]}
    exact = average_precision([[hit]], references, SIGN_CLASSES)["ap50"]
    after = average_precision([[hit, trailing]], references, SIGN_CLASSES)["ap50"]
    before = average_precision([[leading, hit]], references, SIGN_CLASSES)["ap50"]
    assert exact == pytest.approx(1.0)
    assert after == pytest.approx(1.0), "a trailing false positive must not change AP"
    assert before == pytest.approx(0.5), "a false positive ranked first halves precision at recall 1"


def test_average_precision_drops_a_loose_box_at_a_strict_iou_threshold() -> None:
    references = [_refs(("stop-sign", [0, 0, 100, 100]))]
    # IoU 0.64 against the reference: a hit at 0.50, a miss at 0.75.
    predictions = [[{"label": "stop-sign", "score": 1.0, "box": [0, 0, 80, 80]}]]
    result = average_precision(predictions, references, SIGN_CLASSES)
    assert result["ap50"] == pytest.approx(1.0)
    assert result["ap75"] == 0.0
    assert 0.0 < result["ap"] < 1.0


def test_average_precision_uses_the_ten_coco_thresholds() -> None:
    assert COCO_IOU_THRESHOLDS[0] == 0.5
    assert COCO_IOU_THRESHOLDS[-1] == 0.95
    assert len(COCO_IOU_THRESHOLDS) == 10


def test_average_precision_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="prediction lists but"):
        average_precision(
            [[]], [_refs(("stop-sign", [0, 0, 1, 1])), _refs(("stop-sign", [0, 0, 1, 1]))], SIGN_CLASSES
        )
