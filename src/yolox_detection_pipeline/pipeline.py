"""Anchor-free COCO object detection and bounded detection fine-tuning on the pinned YOLOX-S checkpoint.

The checkpoint is ``yolox_s.pth`` from the upstream GitHub release ``0.1.1rc0`` — YOLOX publishes no
Hugging Face model repository, so provenance here is a release asset URL plus the SHA-256 recorded in
``weights/yolox-s/dimer-base-manifest.json``, not a Hub revision. The model code is vendored from the
upstream repository at a pinned commit (``docs/UPSTREAM.md``); nothing is imported from the ``yolox``
distribution and no remote code is executed.

Two capabilities:

* ``detect`` — the 80 COCO classes, score-ordered xyxy pixel boxes, caller-owned confidence and NMS
  thresholds.
* ``finetune`` — a bounded SimOTA fine-tune onto a caller-supplied class vocabulary, re-heading the
  classification branch and keeping the pretrained backbone, neck, box and objectness weights. The
  result is exportable as a single artifact and reloadable from a fresh process.

Two things about the input pipeline are load-bearing and easy to get wrong:

* **Channel order is BGR.** Upstream reads images with ``cv2.imread`` and feeds the array straight to
  the network — raw 0-255 floats, no ``/255``, no mean/std normalisation (``ValTransform(legacy=False)``).
  Feeding RGB instead measurably degrades detection: on this repository's own tutorial scene it loses
  the bench entirely, takes the stop sign from 0.962 to 0.839, and promotes a spurious second clock
  from 0.309 to 0.441.
* **The letterbox is reproduced with Pillow, not OpenCV.** Upstream resizes with
  ``cv2.INTER_LINEAR``; this package uses ``PIL.Image.BILINEAR`` so the dependency set stays at torch,
  torchvision, numpy and Pillow. The two kernels are not bit-identical in general, but the resize is
  skipped outright when the input is already ``INPUT_SIZE`` (verified: a Pillow bilinear resize to the
  source size is the identity), so the default 640x640 sample path is unaffected. Only a BYOD image of
  some other size goes through the differing kernel; ``docs/WEIGHTS.md`` records this.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .coco_classes import COCO_CLASSES

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed
    pass

# --- Identity ---------------------------------------------------------------------------------
# YOLOX has no Hugging Face model repository. MODEL_ID names the upstream project and MODEL_REVISION
# the immutable GitHub release tag whose assets carry every published checkpoint (0.2.0 and 0.3.0 ship
# no weights of their own; upstream's own `yolox/models/build.py` at 0.3.0 points its download URLs
# back at 0.1.1rc0, which is why the 0.3.0 code below pairs with a 0.1.1rc0 checkpoint).
MODEL_ID = "Megvii-BaseDetection/YOLOX"
MODEL_REVISION = "0.1.1rc0"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "yolox-s"
# The upstream commit the vendored model code was taken from; see docs/UPSTREAM.md.
UPSTREAM_CODE_REVISION = "419778480ab6ec0590e5d3831b3afb3b46ab2aa3"
CHECKPOINT_FILE = "yolox_s.pth"
RELEASE_ASSET_URL = (
    f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/{MODEL_REVISION}/{CHECKPOINT_FILE}"
)
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# --- Architecture -----------------------------------------------------------------------------
# YOLOX-S scaling factors, from upstream `exps/default/yolox_s.py`. YOLOX-X is 1.33/1.25 and lives in
# its own repository; these two constants are the only architectural difference between the variants.
MODEL_DEPTH = 0.33
MODEL_WIDTH = 0.50
# `in_channels` and `act` are upstream `Exp` defaults (yolox/exp/yolox_base.py).
BACKBONE_CHANNELS = (256, 512, 1024)
ACTIVATION = "silu"
# BatchNorm overrides applied by upstream `Exp.get_model`'s `init_yolo`.
BN_EPS = 1e-3
BN_MOMENTUM = 0.03
# Upstream `Exp.get_model` calls `head.initialize_biases(1e-2)`: the focal-style prior that starts a
# fresh head at a 1% objectness/class probability instead of 50%.
HEAD_PRIOR_PROB = 1e-2
# Upstream `Exp.test_size`. The letterbox target; any multiple of 32 is architecturally valid.
INPUT_SIZE = (640, 640)
# Upstream `preproc` fills the unused part of the canvas with this grey rather than black.
PAD_VALUE = 114
# The three head strides produce 80x80 + 40x40 + 20x20 = 8400 anchor points at 640x640, and the head
# emits exactly one prediction per point, so nothing can survive NMS beyond this many boxes.
MAX_DETECTIONS = (INPUT_SIZE[0] // 8) ** 2 + (INPUT_SIZE[0] // 16) ** 2 + (INPUT_SIZE[0] // 32) ** 2

# --- Labels -----------------------------------------------------------------------------------
# The 80 COCO 2017 classes in upstream `yolox/data/datasets/coco_classes.py` order, which is the order
# the checkpoint's classification head was trained in.
LABELS: tuple[str, ...] = tuple(COCO_CLASSES)

# --- Thresholds -------------------------------------------------------------------------------
# Upstream ships two threshold pairs for two different purposes and this package keeps both named
# rather than averaging them into one house default:
#   * the demo pair (tools/demo.py --conf 0.3 --nms 0.3), meant for looking at pictures, and
#   * the evaluation pair (Exp.test_conf 0.01 / Exp.nmsthre 0.65), meant for COCO mAP, which keeps
#     far more low-scoring boxes because average precision rewards recall.
# Neither is a calibration for any deployment; the caller owns the choice and passes it explicitly.
DETECTION_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
EVAL_DETECTION_THRESHOLD = 0.01
EVAL_NMS_THRESHOLD = 0.65

# --- Input ceilings ---------------------------------------------------------------------------
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
# Upstream `TrainTransform(max_labels=50)`; the head's label tensor is zero-padded to this width.
MAX_LABELS_PER_IMAGE = 50

# --- Fine-tuning ceilings ---------------------------------------------------------------------
# Bounded so a tutorial adaptation cannot turn into an unbounded training run. These are this
# package's own operating limits, not upstream training recipe values: upstream trains YOLOX-S for
# 300 epochs on COCO with mosaic/mixup augmentation, which is out of scope here.
MAX_TRAIN_IMAGES = 200
MAX_EPOCHS = 20
MAX_CLASSES = 80
# Chosen by measurement, not taste: a grid over epochs x learning rate x freeze on this repository's
# own sample dataset put 6 epochs at 1e-3 with a frozen backbone at held-out AP50 1.0 / AP 0.704,
# against 0.355 / 0.109 at 3 epochs and a total collapse to 0.0 for an unfrozen full fine-tune at the
# same learning rate. docs/release-verification.md records the grid.
DEFAULT_EPOCHS = 6
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SEED = 0
# Artifact format tag written into every exported artifact and checked on reload.
ARTIFACT_FORMAT = "dimer-yolox-detection-adapter/1"
ARTIFACT_FILE = "yolox-detection-adapter.pt"
# IoU thresholds of the COCO primary metric: AP@[.50:.95] averaged over ten thresholds.
COCO_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))
# COCO scores at most this many detections per image. Without the cap a model that has not learned to
# suppress background can post thousands of boxes per image at the 0.01 evaluation threshold, which
# both inflates its own recall tail and makes the matching quadratic in junk.
MAX_EVAL_DETECTIONS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- Snapshot ---------------------------------------------------------------------------------


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch.

    The checkpoint is a pickle, so this runs *before* anything unpickles it: ``from_pretrained``
    verifies, then imports torch, then loads.
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _release_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file from the pinned GitHub release asset URL.

    There is no Hugging Face repository to resolve and no ``main`` to drift onto: a release tag's
    assets are immutable, and the manifest digest is re-checked by ``verify_snapshot`` afterwards, so
    a substituted asset is caught before anything is unpickled.
    """
    url = f"https://github.com/{MODEL_ID}/releases/download/{MODEL_REVISION}/{relative_path}"
    root.mkdir(parents=True, exist_ok=True)
    target = root / relative_path
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:  # noqa: S310
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    tmp.replace(target)


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally; return the relative paths fetched.

    A fresh clone commits the manifest but git-ignores the checkpoint, so this is the normal path.
    ``verify_snapshot`` still runs afterwards and is what makes the download trustworthy.
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them from the {MODEL_REVISION} release assets"
        )
    fetch = downloader or _release_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


# --- Geometry and preprocessing ---------------------------------------------------------------


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy pixel boxes; the building block of the AP computation."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def preprocess(image: Image.Image) -> tuple[np.ndarray, float]:
    """Upstream's ``preproc`` letterbox, in Pillow and in BGR: returns (CHW float32 array, ratio).

    The array is raw 0-255 floats in **BGR** channel order on a ``PAD_VALUE`` canvas with the image
    pasted at the top-left corner and the aspect ratio preserved — exactly what upstream feeds the
    network. Divide the returned box coordinates by ``ratio`` to map predictions back to input pixels.
    """
    width, height = image.size
    ratio = min(INPUT_SIZE[0] / height, INPUT_SIZE[1] / width)
    new_w, new_h = int(width * ratio), int(height * ratio)
    canvas = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), PAD_VALUE, dtype=np.uint8)
    # Skipped when the input is already INPUT_SIZE, which is the default sample path: a Pillow
    # bilinear resize to the source size is the identity, so no resize kernel is involved at all.
    resized = image if (new_w, new_h) == (width, height) else image.resize((new_w, new_h), Image.BILINEAR)
    canvas[:new_h, :new_w] = np.asarray(resized, dtype=np.uint8)
    chw = canvas.transpose(2, 0, 1)[::-1]  # HWC RGB -> CHW BGR
    return np.ascontiguousarray(chw, dtype=np.float32), ratio


# --- Validation -------------------------------------------------------------------------------


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(value: Any, name: str = "threshold") -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one image as PIL.Image.Image (any mode, converted to RGB): a photograph or a rendered scene",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "threshold": [0.0, 1.0],
    "nms_threshold": [0.0, 1.0],
    "labels": list(LABELS),
    "max_detections": MAX_DETECTIONS,
    "preprocessing": (
        f"image converted to RGB, letterboxed onto a {INPUT_SIZE[0]}x{INPUT_SIZE[1]} canvas filled with "
        f"{PAD_VALUE} (aspect ratio preserved, pasted top-left), then reordered to BGR and kept as raw "
        "0-255 float32 with no rescaling and no mean/std normalisation, matching upstream "
        "ValTransform(legacy=False); returned boxes are divided by the letterbox ratio to land back in "
        "input pixels"
    ),
}

TRAIN_SCHEMA: dict[str, Any] = {
    "record": (
        "a mapping with 'image' (PIL.Image.Image), 'boxes' (list of xyxy pixel boxes in that image's "
        "own coordinates) and 'labels' (list of class names, one per box, drawn from class_names)"
    ),
    "images": [1, MAX_TRAIN_IMAGES],
    "labels_per_image": [0, MAX_LABELS_PER_IMAGE],
    "classes": [1, MAX_CLASSES],
    "epochs": [1, MAX_EPOCHS],
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "preprocessing": INPUT_SCHEMA["preprocessing"],
    "note": (
        "boxes are converted to the head's own target form — (class index, cx, cy, w, h) in letterboxed "
        f"canvas pixels, zero-padded to {MAX_LABELS_PER_IMAGE} rows per image — inside finetune; a caller "
        "supplies xyxy in input-image pixels and never touches that conversion"
    ),
}


def _check_inputs(image: Any, threshold: Any, nms_threshold: Any) -> tuple[Image.Image, float, float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``detect`` and ``validate_inputs`` both route through this function so their acceptance criteria
    cannot diverge.
    """
    return (
        validate_image(image),
        _check_threshold(threshold, "threshold"),
        _check_threshold(nms_threshold, "nms_threshold"),
    )


def validate_inputs(
    image: Image.Image,
    *,
    threshold: float = DETECTION_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``detect`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, checked, checked_nms = _check_inputs(image, threshold, nms_threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (detect takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "threshold": checked,
        "nms_threshold": checked_nms,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    epochs: int = DEFAULT_EPOCHS,
) -> dict[str, Any]:
    """Validation stage for the adaptation path: return the dataset manifest, or raise.

    Applies exactly the ceilings ``finetune`` applies, so a dataset this accepts cannot be refused
    later. Boxes are checked in the record's own image pixels, before any letterboxing.
    """
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise TypeError(f"records must be a sequence of mappings, got {type(records).__name__}")
    names = list(class_names)
    if not names:
        raise ValueError("class_names must name at least one class")
    if len(names) > MAX_CLASSES:
        raise ValueError(f"class_names has {len(names)} entries > MAX_CLASSES {MAX_CLASSES}")
    if len(set(names)) != len(names):
        raise ValueError("class_names must not repeat a name")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("every class name must be a non-empty string")
    if not 1 <= len(records) <= MAX_TRAIN_IMAGES:
        raise ValueError(f"records has {len(records)} images, expected 1..{MAX_TRAIN_IMAGES}")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be an int in 1..{MAX_EPOCHS}, got {epochs!r}")

    per_class = dict.fromkeys(names, 0)
    total_boxes = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping, got {type(record).__name__}")
        missing = {"image", "boxes", "labels"} - set(record)
        if missing:
            raise ValueError(f"record {index} is missing {sorted(missing)}")
        image = validate_image(record["image"])
        boxes, labels = list(record["boxes"]), list(record["labels"])
        if len(boxes) != len(labels):
            raise ValueError(f"record {index}: {len(boxes)} boxes but {len(labels)} labels")
        if len(boxes) > MAX_LABELS_PER_IMAGE:
            raise ValueError(
                f"record {index}: {len(boxes)} boxes > MAX_LABELS_PER_IMAGE {MAX_LABELS_PER_IMAGE}"
            )
        width, height = image.size
        for box, label in zip(boxes, labels, strict=True):
            if len(box) != 4:
                raise ValueError(f"record {index}: box {box!r} is not [x0, y0, x1, y1]")
            x0, y0, x1, y1 = (float(v) for v in box)
            if not (x1 > x0 and y1 > y0):
                raise ValueError(f"record {index}: box {box!r} must satisfy x0 < x1 and y0 < y1")
            if not (x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height):
                raise ValueError(f"record {index}: box {box!r} falls outside the {width}x{height} image")
            if label not in per_class:
                raise ValueError(f"record {index}: label {label!r} is not in class_names {names}")
            per_class[label] += 1
            total_boxes += 1
    empty = [name for name, count in per_class.items() if count == 0]
    return {
        "schema": dict(TRAIN_SCHEMA),
        "class_names": names,
        "n_images": len(records),
        "n_boxes": total_boxes,
        "boxes_per_class": per_class,
        "epochs": epochs,
        "verdict": "accepted",
        "findings": (
            [f"classes with no boxes in this dataset: {empty}; their head outputs stay untrained"]
            if empty
            else []
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def split_records(
    records: Sequence[Mapping[str, Any]],
    *,
    holdout: float = 0.25,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Deterministic train/held-out split. The split is the caller's, not the model's: it happens
    before any weight is touched, and the held-out part is never shown to ``finetune``."""
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be in (0, 1), got {holdout!r}")
    order = np.random.default_rng(seed).permutation(len(records))
    n_holdout = max(1, int(round(len(records) * holdout)))
    if n_holdout >= len(records):
        raise ValueError(f"holdout {holdout} leaves no training images out of {len(records)}")
    held = [records[i] for i in order[:n_holdout]]
    train = [records[i] for i in order[n_holdout:]]
    return train, held


# --- Evaluation -------------------------------------------------------------------------------


def average_precision(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """COCO-style average precision over a list of images.

    ``predictions[i]`` are the detections for image ``i`` (``label``, ``score``, ``box``) and
    ``references[i]`` carries that image's ``boxes``/``labels``. Matching is the COCO rule: within a
    class, detections in descending score order each claim the highest-IoU unclaimed reference above
    the threshold; unmatched detections are false positives and unmatched references false negatives.
    Precision is interpolated over 101 recall points, AP is averaged over classes that have at least
    one reference, and ``ap`` is the mean over ``iou_thresholds``.

    This is a faithful small implementation, not ``pycocotools``: it has no area ranges and no crowd
    handling, and it scores whatever list it is handed rather than applying the per-image detection
    cap itself (``evaluate`` applies ``MAX_EVAL_DETECTIONS`` before calling in). It must not be
    compared against published COCO numbers; it exists to score a tutorial's own held-out images.
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} prediction lists but {len(references)} references")
    names = list(class_names)
    recall_points = np.linspace(0.0, 1.0, 101)
    per_threshold: dict[float, dict[str, float]] = {}
    for threshold in iou_thresholds:
        per_class: dict[str, float] = {}
        for name in names:
            scored: list[tuple[float, bool]] = []
            n_references = 0
            for dets, reference in zip(predictions, references, strict=True):
                ref_boxes = [
                    box
                    for box, label in zip(reference["boxes"], reference["labels"], strict=True)
                    if label == name
                ]
                n_references += len(ref_boxes)
                claimed = [False] * len(ref_boxes)
                candidates = sorted((d for d in dets if d["label"] == name), key=lambda d: -float(d["score"]))
                for det in candidates:
                    best, best_iou = -1, 0.0
                    for j, ref_box in enumerate(ref_boxes):
                        if claimed[j]:
                            continue
                        value = box_iou(det["box"], ref_box)
                        if value > best_iou:
                            best, best_iou = j, value
                    hit = best >= 0 and best_iou >= threshold
                    if hit:
                        claimed[best] = True
                    scored.append((float(det["score"]), hit))
            if n_references == 0:
                continue  # a class with no references contributes no AP, as in COCO
            scored.sort(key=lambda pair: -pair[0])
            true_positives = np.cumsum([1 if hit else 0 for _score, hit in scored])
            false_positives = np.cumsum([0 if hit else 1 for _score, hit in scored])
            if not scored:
                per_class[name] = 0.0
                continue
            recall = true_positives / n_references
            precision = true_positives / np.maximum(true_positives + false_positives, 1)
            # Monotone envelope, then sample at the 101 recall points (COCO's interpolation).
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            sampled = np.zeros_like(recall_points)
            indices = np.searchsorted(recall, recall_points, side="left")
            valid = indices < len(precision)
            sampled[valid] = precision[indices[valid]]
            per_class[name] = float(sampled.mean())
        per_threshold[threshold] = per_class
    scored_classes = sorted({name for values in per_threshold.values() for name in values})
    means = {
        threshold: (float(np.mean(list(values.values()))) if values else 0.0)
        for threshold, values in per_threshold.items()
    }
    return {
        "ap": float(np.mean(list(means.values()))) if means else 0.0,
        "ap50": means.get(0.5, 0.0),
        "ap75": means.get(0.75, 0.0),
        "per_class_ap50": per_threshold.get(0.5, {}),
        "iou_thresholds": [float(t) for t in iou_thresholds],
        "scored_classes": scored_classes,
        "n_images": len(references),
        "n_references": sum(len(r["boxes"]) for r in references),
    }


def evaluation_report(
    result: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[Sequence[float]]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Single-image evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth_boxes`` (label -> xyxy reference boxes for one image) the report carries one
    ``box_iou`` entry per reference — the best-overlapping detection **of the same label** — as
    sample-sanity geometry evidence. Without them the verdict is ``not-measurable``. For a scored
    held-out set use ``average_precision`` instead; this helper deliberately does not call a
    single-image IoU a mean average precision.
    """
    detections = list(result["detections"])
    labels = tuple(result.get("class_names") or LABELS)
    base = {
        "task": f"object detection over {len(labels)} classes on one image",
        "decision_rule": (
            "each of the 8400 anchor points emits one box, one objectness logit and one logit per "
            "class; a prediction survives when objectness x class probability (both sigmoids, not a "
            "softmax over classes) reaches the confidence threshold and it wins per-class NMS at the "
            "NMS threshold. Neither threshold is calibrated for any deployment"
        ),
        "threshold": result.get("threshold", DETECTION_THRESHOLD),
        "nms_threshold": result.get("nms_threshold", NMS_THRESHOLD),
        "sample_kind": sample_kind,
        "n_detections": len(detections),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not ground_truth_boxes:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth object boxes were supplied for the evaluated image",
            "needs": (
                "labelled boxes per class on your own images, scored with average_precision over a "
                "held-out set to obtain AP@[.50:.95] and AP50; a single image's box_iou values are "
                "geometry sanity evidence and no labelled image set ships with this repository"
            ),
        }
    metrics = []
    for label, boxes in ground_truth_boxes.items():
        if label not in labels:
            raise ValueError(f"unknown reference label {label!r}; expected one of {len(labels)} class names")
        same_label = [det for det in detections if det["label"] == label]
        for index, box in enumerate(boxes):
            ious = [box_iou(det["box"], box) for det in same_label]
            best = max(range(len(ious)), key=ious.__getitem__) if ious else None
            metrics.append(
                {
                    "id": "box_iou",
                    "reference": f"{label}-{index}",
                    "value": ious[best] if best is not None else 0.0,
                    "matched_score": same_label[best]["score"] if best is not None else None,
                    "n_detected_same_label": len(same_label),
                    "estimation": "one reference box per object on a single image, no dispersion estimate",
                }
            )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} reference box(es) on one tutorial image; geometry sanity evidence, "
            "not a detection benchmark"
        ),
        "needs": (
            "a labelled image set from the deployment domain (cameras, scenes, object classes) for any "
            "average-precision claim; use average_precision on a held-out split for that"
        ),
    }


# --- Pipeline ---------------------------------------------------------------------------------


@dataclass
class YoloxDetectionPipeline:
    """YOLOX-S detection and bounded detection fine-tuning over a verified local checkpoint."""

    model: Any
    device: str
    class_names: tuple[str, ...]
    source: str = "local-snapshot"
    base_state_digest: str | None = None
    adapted: bool = False
    reinitialised: tuple[str, ...] = field(default_factory=tuple)

    # -- construction --------------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        class_names: Sequence[str] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> YoloxDetectionPipeline:
        """Build YOLOX-S and load the verified checkpoint.

        With ``class_names`` the classification branch is rebuilt for that vocabulary and randomly
        initialised while the backbone, neck, box and objectness weights are kept — the starting point
        for ``finetune``. Without it the model keeps the checkpoint's own 80 COCO classes and every
        tensor is loaded ``strict=True``.

        ``seed`` fixes that random initialisation. It matters: the re-headed classification layers are
        the only untrained weights in the model, and they are what the pre-adaptation baseline is
        measuring, so leaving them at whatever the ambient global RNG happened to hold makes the
        baseline number irreproducible from run to run. The global RNG state is restored afterwards so
        seeding here cannot perturb a caller's own stream.
        """
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
        else:
            raise FileNotFoundError(
                f"no manifest at {root}; commit weights/{MODEL_KEY}/{MANIFEST_NAME} and stage "
                f"{CHECKPOINT_FILE} from the {MODEL_REVISION} release assets"
            )
        names = tuple(class_names) if class_names is not None else LABELS
        if not 1 <= len(names) <= MAX_CLASSES:
            raise ValueError(f"class_names must hold 1..{MAX_CLASSES} names, got {len(names)}")

        # Refuse an invalid snapshot before importing torch or unpickling anything.
        import torch

        checkpoint_path = root / CHECKPOINT_FILE
        # The upstream checkpoint is a pickle, not SafeTensors. It is a plain tensor/int dict, so
        # weights_only=True is enough to load it without allowing arbitrary globals; the digest was
        # already checked above. Upstream's own loader passes neither.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        digest = hashlib.sha256(
            b"".join(state[key].detach().cpu().numpy().tobytes() for key in sorted(state))
        ).hexdigest()

        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            model = cls._build(len(names))
        finally:
            torch.set_rng_state(rng_state)
        reinitialised: tuple[str, ...] = ()
        if len(names) == len(LABELS) and class_names is None:
            model.load_state_dict(state, strict=True)
        else:
            # Transfer learning: keep everything whose shape still matches (backbone, PAFPN, the box
            # and objectness heads) and leave the class-conditional tensors at their initialisation.
            own = model.state_dict()
            transferable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
            dropped = tuple(sorted(set(own) - set(transferable)))
            model.load_state_dict(transferable, strict=False)
            reinitialised = dropped
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(resolved_device).eval()
        return cls(
            model=model,
            device=resolved_device,
            class_names=names,
            base_state_digest=digest,
            reinitialised=reinitialised,
        )

    @staticmethod
    def _build(num_classes: int) -> Any:
        import torch

        from .yolo_head import YOLOXHead
        from .yolo_pafpn import YOLOPAFPN
        from .yolox import YOLOX

        channels = list(BACKBONE_CHANNELS)
        backbone = YOLOPAFPN(MODEL_DEPTH, MODEL_WIDTH, in_channels=channels, act=ACTIVATION)
        head = YOLOXHead(num_classes, MODEL_WIDTH, in_channels=channels, act=ACTIVATION)
        model = YOLOX(backbone, head)
        # Upstream `init_yolo`: the BatchNorm defaults YOLOX trains with, not PyTorch's.
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eps = BN_EPS
                module.momentum = BN_MOMENTUM
        # Upstream `Exp.get_model` also does this, and it matters: it sets every classification and
        # objectness bias to -log((1 - p) / p) for p = HEAD_PRIOR_PROB, so a freshly initialised head
        # starts by predicting "almost certainly nothing here" instead of a coin flip at all 8400
        # anchors. Omitting it lets the objectness term saturate within a few steps of a re-headed
        # fine-tune and the model collapses to one class at score 1.0 everywhere. For the COCO path
        # the checkpoint overwrites these biases immediately, so it is a no-op there.
        model.head.initialize_biases(HEAD_PRIOR_PROB)
        return model

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> YoloxDetectionPipeline:
        """Rebuild an adapted pipeline from an exported artifact, in a process that never fine-tuned.

        The artifact records its own format tag, class vocabulary and the digest of the base weights
        it started from; a mismatch raises rather than silently loading a different model.
        """
        import torch

        artifact_path = Path(path)
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if payload.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {payload.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if payload.get("model_id") != MODEL_ID or payload.get("model_revision") != MODEL_REVISION:
            raise ValueError(
                f"artifact was built on {payload.get('model_id')}@{payload.get('model_revision')}, "
                f"package pins {MODEL_ID}@{MODEL_REVISION}"
            )
        names = tuple(payload["class_names"])
        model = cls._build(len(names))
        model.load_state_dict(payload["state_dict"], strict=True)
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(resolved_device).eval()
        return cls(
            model=model,
            device=resolved_device,
            class_names=names,
            source=f"artifact:{artifact_path.name}",
            base_state_digest=payload.get("base_state_digest"),
            adapted=True,
        )

    # -- inference -----------------------------------------------------------------------------

    def detect(
        self,
        image: Image.Image,
        *,
        threshold: float = DETECTION_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> dict[str, Any]:
        """Detect objects on one image; boxes are xyxy pixel coordinates in the input image."""
        rgb, checked, checked_nms = _check_inputs(image, threshold, nms_threshold)
        detections = self._run(rgb, checked, checked_nms)
        if len(detections) > MAX_DETECTIONS:
            raise RuntimeError(
                f"backend returned {len(detections)} detections > anchor count {MAX_DETECTIONS}"
            )
        for det in detections:
            if (
                set(det) != {"box", "label", "score"}
                or len(det["box"]) != 4
                or det["label"] not in self.class_names
            ):
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")
        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "threshold": checked,
            "nms_threshold": checked_nms,
            "width": rgb.width,
            "height": rgb.height,
            "class_names": list(self.class_names),
            "adapted": self.adapted,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def _run(self, image: Image.Image, threshold: float, nms_threshold: float) -> list[dict[str, Any]]:
        import torch

        from .ops import postprocess

        chw, ratio = preprocess(image)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        was_training = self.model.training
        self.model.eval()
        # no_grad rather than inference_mode: upstream `postprocess` writes the xyxy conversion back
        # into its argument in place, which an inference-mode tensor refuses. Upstream's own
        # tools/demo.py uses no_grad for the same reason.
        with torch.no_grad():
            raw = self.model(tensor)
        kept = postprocess(
            raw, len(self.class_names), conf_thre=threshold, nms_thre=nms_threshold, class_agnostic=False
        )[0]
        if was_training:
            self.model.train()
        if kept is None:
            return []
        detections = []
        for x0, y0, x1, y1, objectness, class_conf, class_id in kept.tolist():
            detections.append(
                {
                    "box": [x0 / ratio, y0 / ratio, x1 / ratio, y1 / ratio],
                    "label": self.class_names[int(class_id)],
                    "score": float(objectness * class_conf),
                }
            )
        return detections

    def detect_many(
        self,
        images: Sequence[Image.Image],
        *,
        threshold: float = EVAL_DETECTION_THRESHOLD,
        nms_threshold: float = EVAL_NMS_THRESHOLD,
    ) -> list[list[dict[str, Any]]]:
        """Detections for several images, one ``detect`` per image, defaulting to the *evaluation*
        thresholds — average precision rewards recall, so scoring at the demo thresholds understates
        it. Batching is deliberately not implemented: see the model card's out-of-scope section."""
        return [
            self.detect(image, threshold=threshold, nms_threshold=nms_threshold)["detections"]
            for image in images
        ]

    # -- adaptation ----------------------------------------------------------------------------

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int = DEFAULT_SEED,
        freeze_backbone: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded SimOTA fine-tune on ``records``; returns the run manifest and per-epoch losses.

        The loss and the label assignment are upstream's: the vendored ``YOLOXHead`` in training mode
        computes SimOTA matching, the IoU/objectness/classification terms and their weighting. What
        this method owns is the bounded loop around it — the target-tensor conversion, batching, the
        optimiser, the seed and the ceilings. ``use_l1`` stays off: upstream only enables the extra L1
        box term for the last 15 of 300 epochs, and switching it on for a 3-epoch tutorial would
        change the loss scale for no benefit.

        ``freeze_backbone`` (on by default) trains the detection head only and leaves the CSPDarknet
        backbone and PAFPN neck exactly as the COCO checkpoint left them. That is the honest default
        for a bounded tutorial on CPU: it is several times faster per step, it cannot damage the
        pretrained features with a handful of gradient steps on a small set, and the features a
        COCO-trained backbone already has are what makes a few-epoch adaptation work at all. Pass
        ``freeze_backbone=False`` for a full fine-tune when you have the data and the compute for it.

        Mutates this pipeline in place (``adapted`` becomes True) and leaves the model in eval mode.
        """
        import torch

        manifest = validate_dataset(records, self.class_names, epochs=epochs)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if not isinstance(learning_rate, int | float) or isinstance(learning_rate, bool):
            raise ValueError(f"learning_rate must be a number, got {learning_rate!r}")
        if not 0.0 < float(learning_rate) <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate!r}")

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        images, targets = self._as_batch_tensors(records)
        images, targets = images.to(self.device), targets.to(self.device)

        for parameter in self.model.backbone.parameters():
            parameter.requires_grad = not freeze_backbone
        self.model.train()
        if freeze_backbone:
            # Keep the frozen BatchNorm statistics the checkpoint was trained with: in train() mode a
            # BatchNorm updates its running mean/var even with requires_grad=False, so 27 steps of a
            # tutorial batch would quietly rewrite the backbone's normalisation.
            self.model.backbone.eval()
        self.model.head.use_l1 = False
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=float(learning_rate), momentum=0.9, weight_decay=5e-4)
        history: list[dict[str, Any]] = []
        n = len(records)
        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_losses: list[dict[str, float]] = []
            for start in range(0, n, batch_size):
                index = torch.as_tensor(order[start : start + batch_size].copy(), dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                losses = self.model(images[index].to(self.device), targets[index].to(self.device))
                losses["total_loss"].backward()
                optimizer.step()
                epoch_losses.append(
                    {
                        key: float(value.detach()) if torch.is_tensor(value) else float(value)
                        for key, value in losses.items()
                    }
                )
            row = {
                "epoch": epoch + 1,
                "steps": len(epoch_losses),
                **{
                    key: round(float(np.mean([loss[key] for loss in epoch_losses])), 5)
                    for key in epoch_losses[0]
                },
            }
            if not math.isfinite(row["total_loss"]):
                raise RuntimeError(f"epoch {epoch + 1} produced a non-finite loss: {row}")
            history.append(row)
            if progress is not None:
                progress(row)
        self.model.eval()
        self.adapted = True
        return {
            "dataset": manifest,
            "history": history,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "seed": seed,
            "optimizer": "SGD(momentum=0.9, weight_decay=5e-4)",
            "freeze_backbone": freeze_backbone,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "use_l1": False,
            "loss": "upstream YOLOXHead SimOTA (IoU + objectness + classification)",
            "device": self.device,
            "class_names": list(self.class_names),
            "reinitialised_tensors": list(self.reinitialised),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def _as_batch_tensors(self, records: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
        """Letterbox every record and build the head's own label tensor.

        The head reads ``labels[b, :, 0]`` as a class index and ``labels[b, :, 1:5]`` as (cx, cy, w, h)
        in letterboxed-canvas pixels, counting valid rows as those whose five values do not sum to
        zero — so rows must start at index 0 and stay contiguous.
        """
        import torch

        batch = torch.zeros(len(records), 3, INPUT_SIZE[0], INPUT_SIZE[1])
        labels = torch.zeros(len(records), MAX_LABELS_PER_IMAGE, 5)
        for index, record in enumerate(records):
            chw, ratio = preprocess(validate_image(record["image"]))
            batch[index] = torch.from_numpy(chw)
            for row, (box, label) in enumerate(zip(record["boxes"], record["labels"], strict=True)):
                x0, y0, x1, y1 = (float(v) * ratio for v in box)
                labels[index, row] = torch.tensor(
                    [
                        float(self.class_names.index(label)),
                        (x0 + x1) / 2,
                        (y0 + y1) / 2,
                        x1 - x0,
                        y1 - y0,
                    ]
                )
        return batch, labels

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        threshold: float = EVAL_DETECTION_THRESHOLD,
        nms_threshold: float = EVAL_NMS_THRESHOLD,
        iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
        max_detections: int = MAX_EVAL_DETECTIONS,
    ) -> dict[str, Any]:
        """Score a held-out set: average precision plus the request that produced the detections."""
        predictions = self.detect_many(
            [record["image"] for record in records], threshold=threshold, nms_threshold=nms_threshold
        )
        raw_counts = [len(dets) for dets in predictions]
        # COCO's per-image cap, applied to the score-ordered detections before matching.
        predictions = [dets[:max_detections] for dets in predictions]
        metrics = average_precision(predictions, records, self.class_names, iou_thresholds=iou_thresholds)
        return {
            **metrics,
            "threshold": _check_threshold(threshold, "threshold"),
            "nms_threshold": _check_threshold(nms_threshold, "nms_threshold"),
            "max_detections": max_detections,
            "detections_before_cap": raw_counts,
            "adapted": self.adapted,
            "class_names": list(self.class_names),
            "implementation": (
                "package-local average_precision: COCO matching and 101-point interpolation, without "
                "pycocotools' area ranges, detection cap or crowd handling — not comparable to "
                "published COCO numbers"
            ),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # -- artifact ------------------------------------------------------------------------------

    def save_artifact(self, path: str | Path, *, notes: str | None = None) -> dict[str, Any]:
        """Write the adapted weights and their provenance as one artifact; return its descriptor.

        A single ``torch.save`` of tensors and plain metadata — no archive, no pickled objects beyond
        what ``weights_only=True`` accepts on reload, so there is no extraction step to make safe.
        """
        import torch

        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "upstream_code_revision": UPSTREAM_CODE_REVISION,
            "model_key": MODEL_KEY,
            "class_names": list(self.class_names),
            "depth": MODEL_DEPTH,
            "width": MODEL_WIDTH,
            "input_size": list(INPUT_SIZE),
            "adapted": self.adapted,
            "base_state_digest": self.base_state_digest,
            "notes": notes or "",
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
        }
        torch.save(payload, artifact_path)
        return {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
            "format": ARTIFACT_FORMAT,
            "class_names": list(self.class_names),
            "tensors": len(payload["state_dict"]),
            "base_state_digest": self.base_state_digest,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
