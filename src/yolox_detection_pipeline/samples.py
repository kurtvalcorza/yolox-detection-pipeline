"""Deterministic in-code sample data: the COCO demonstration scene and the adaptation dataset.

Nothing here is downloaded and nothing needs torch — Pillow and numpy only — so the tutorial's
default path has no dataset dependency and the same generators are exercised by the repository's unit
tests and by the smoke run whose numbers the model card quotes.

Everything is drawn at ``INPUT_SIZE`` (640x640), which is also what keeps the letterbox a no-op on the
default path (see ``pipeline.preprocess``).

Two separate label vocabularies live here and must not be conflated:

* ``tutorial_scene`` returns references drawn from the checkpoint's own 80 **COCO** classes; it
  demonstrates the pretrained model and is not training data.
* ``sign_dataset`` returns records labelled with ``SIGN_CLASSES``, a three-class vocabulary that does
  **not** exist in COCO. It is the adaptation dataset, and a model fine-tuned on it answers in those
  three names only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCENE_SIZE = (640, 640)

# The adaptation vocabulary. Deliberately not COCO names: "stop sign" exists in COCO, "yield-sign" and
# "speed-limit-sign" do not, and the hyphenated spellings keep the two vocabularies visually distinct
# in output. A model fine-tuned on this dataset predicts only these three.
SIGN_CLASSES: tuple[str, ...] = ("stop-sign", "yield-sign", "speed-limit-sign")


def _font(size: int) -> Any:
    return ImageFont.load_default(size=size)


def _street_background(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    draw.rectangle([0, 0, width, height], fill=(232, 236, 240))
    draw.rectangle([0, int(height * 0.73), width, height], fill=(120, 124, 128))
    draw.rectangle([0, int(height * 0.73), width, int(height * 0.73) + 8], fill=(240, 240, 240))


def _octagon(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, font: Any) -> list[float]:
    points = [
        (cx + r * math.cos(math.pi / 8 + i * math.pi / 4), cy + r * math.sin(math.pi / 8 + i * math.pi / 4))
        for i in range(8)
    ]
    draw.polygon(points, fill=(196, 30, 34), outline=(255, 255, 255))
    text = "STOP"
    draw.text(
        (cx - draw.textlength(text, font=font) / 2, cy - r * 0.27), text, fill=(255, 255, 255), font=font
    )
    # A regular octagon inscribed in a circle of radius r reaches only r*cos(pi/8) ~ 0.924r along
    # either axis, never r. Returning the circumscribing square would make the reference box 17%
    # larger in area than the shape it names, which caps the achievable IoU and deflates every AP
    # number computed against it. The extent is derived from the same vertices that were drawn.
    half = r * math.cos(math.pi / 8)
    return [cx - half, cy - half, cx + half, cy + half]


def _traffic_light(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float]) -> list[float]:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(40, 42, 46))
    lamp_h = (y1 - y0) / 3
    for i, colour in enumerate([(220, 50, 40), (232, 190, 60), (70, 190, 90)]):
        top = y0 + 10 + i * lamp_h
        draw.ellipse([x0 + 8, top, x1 - 8, top + lamp_h - 12], fill=colour)
    draw.rectangle([(x0 + x1) / 2 - 6, y1, (x0 + x1) / 2 + 6, 480], fill=(110, 110, 114))
    return [x0, y0, x1, y1]


def _analogue_clock(
    draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], font: Any
) -> list[float]:
    """A clock face with tick marks and hour numerals.

    The numerals matter: a plain white disc with two hands is not recognised as a clock by this
    checkpoint, and worse, it makes the drawn ball elsewhere in the scene *become* the clock. The
    repository's release verification records that probe.
    """
    x0, y0, x1, y1 = box
    cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
    draw.ellipse([x0, y0, x1, y1], fill=(252, 252, 250), outline=(24, 24, 24), width=7)
    for i in range(12):
        angle = -math.pi / 2 + i * math.pi / 6
        draw.line(
            [
                cx + (r - 14) * math.cos(angle),
                cy + (r - 14) * math.sin(angle),
                cx + (r - 6) * math.cos(angle),
                cy + (r - 6) * math.sin(angle),
            ],
            fill=(30, 30, 30),
            width=3,
        )
    for hour, step in ((12, 9), (3, 0), (6, 3), (9, 6)):
        angle = -math.pi / 2 + step * math.pi / 6
        tx, ty = cx + (r - 30) * math.cos(angle), cy + (r - 30) * math.sin(angle)
        draw.text((tx - 9, ty - 11), str(hour), fill=(24, 24, 24), font=font)
    draw.line(
        [cx, cy, cx + (r - 34) * math.cos(-math.pi / 3), cy + (r - 34) * math.sin(-math.pi / 3)],
        fill=(20, 20, 20),
        width=7,
    )
    draw.line(
        [cx, cy, cx + (r - 18) * math.cos(math.pi / 8), cy + (r - 18) * math.sin(math.pi / 8)],
        fill=(20, 20, 20),
        width=4,
    )
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(20, 20, 20))
    return [x0, y0, x1, y1]


def _football(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float]) -> list[float]:
    x0, y0, x1, y1 = box
    cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
    draw.ellipse([x0, y0, x1, y1], fill=(250, 250, 250), outline=(30, 30, 30), width=3)
    pentagon = [
        (
            cx + 0.36 * r * math.cos(-math.pi / 2 + i * 2 * math.pi / 5),
            cy + 0.36 * r * math.sin(-math.pi / 2 + i * 2 * math.pi / 5),
        )
        for i in range(5)
    ]
    draw.polygon(pentagon, fill=(28, 28, 28))
    for px, py in pentagon:
        draw.line([px, py, cx + (px - cx) * 2.6, cy + (py - cy) * 2.6], fill=(28, 28, 28), width=3)
    return [x0, y0, x1, y1]


def _bench(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float]) -> list[float]:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y0 + 16], fill=(150, 105, 60))
    draw.rectangle([x0, y0 + 26, x1, y0 + 40], fill=(150, 105, 60))
    draw.rectangle([x0 + 8, y0 + 16, x0 + 18, y1], fill=(120, 84, 48))
    draw.rectangle([x1 - 18, y0 + 16, x1 - 8, y1], fill=(120, 84, 48))
    return [x0, y0, x1, y1]


def tutorial_scene() -> tuple[Image.Image, dict[str, list[list[float]]]]:
    """The COCO demonstration scene: five drawn objects and their reference boxes, keyed by COCO label.

    These are drawn references on a rendered picture, not a labelled photographic dataset: they are
    enough for a per-object ``box_iou`` sanity check and nothing more.
    """
    image = Image.new("RGB", SCENE_SIZE, (232, 236, 240))
    draw = ImageDraw.Draw(image)
    _street_background(draw, *SCENE_SIZE)
    references: dict[str, list[list[float]]] = {}
    draw.rectangle([114, 270, 126, 480], fill=(110, 110, 114))
    references["stop sign"] = [_octagon(draw, 120, 200, 70, _font(22))]
    references["traffic light"] = [_traffic_light(draw, (300, 120, 356, 250))]
    references["clock"] = [_analogue_clock(draw, (460, 130, 600, 270), _font(18))]
    references["sports ball"] = [_football(draw, (240, 520, 320, 600))]
    references["bench"] = [_bench(draw, (440, 500, 610, 580))]
    return image, references


def blank_scene() -> Image.Image:
    """A featureless white image: the degenerate input every detector should be asked about."""
    return Image.new("RGB", SCENE_SIZE, (255, 255, 255))


def noise_scene(seed: int = 0) -> Image.Image:
    """Uniform RGB noise: structure-free input, for the same reason as ``blank_scene``."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (*SCENE_SIZE[::-1], 3), dtype=np.uint8))


def _yield_sign(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> list[float]:
    points = [(cx, cy + r), (cx - r * 0.95, cy - r * 0.75), (cx + r * 0.95, cy - r * 0.75)]
    draw.polygon(points, fill=(255, 255, 255), outline=(198, 32, 36))
    inner = [(cx, cy + r * 0.62), (cx - r * 0.62, cy - r * 0.5), (cx + r * 0.62, cy - r * 0.5)]
    draw.line([*inner, inner[0]], fill=(198, 32, 36), width=int(max(4, r * 0.22)))
    return [cx - r * 0.95, cy - r * 0.75, cx + r * 0.95, cy + r]


def _speed_limit_sign(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, limit: int) -> list[float]:
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=(255, 255, 255),
        outline=(198, 32, 36),
        width=int(max(4, r * 0.2)),
    )
    font = _font(int(max(12, r * 0.9)))
    text = str(limit)
    draw.text((cx - draw.textlength(text, font=font) / 2, cy - r * 0.55), text, fill=(30, 30, 30), font=font)
    return [cx - r, cy - r, cx + r, cy + r]


def sign_dataset(
    n_images: int = 40,
    *,
    seed: int = 0,
    max_objects: int = 3,
) -> list[dict[str, Any]]:
    """A deterministic labelled dataset over ``SIGN_CLASSES`` for the bounded fine-tune.

    Each record is ``{"image": PIL.Image, "boxes": [[x0, y0, x1, y1], ...], "labels": [name, ...]}`` —
    the record shape ``validate_dataset`` and ``finetune`` accept, and the shape a BYOD caller must
    produce from their own labelled images. Signs are placed on a non-overlapping grid so the boxes
    are exact by construction rather than approximate.

    It is synthetic drawn data, so a model fine-tuned on it learns to find *these renderings*. That is
    the point of a bounded tutorial adaptation and the reason its metrics are not a claim about real
    traffic signs.
    """
    if not 1 <= n_images <= 500:
        raise ValueError(f"n_images must be in 1..500, got {n_images}")
    if not 1 <= max_objects <= 6:
        raise ValueError(f"max_objects must be in 1..6, got {max_objects}")
    rng = np.random.default_rng(seed)
    width, height = SCENE_SIZE
    slots = [(x, y) for y in (150, 400) for x in (140, 360, 560)]
    records: list[dict[str, Any]] = []
    for _ in range(n_images):
        image = Image.new("RGB", SCENE_SIZE, (232, 236, 240))
        draw = ImageDraw.Draw(image)
        # A varied but never-black background, so the network cannot key on a constant canvas.
        tint = rng.integers(200, 245, 3)
        draw.rectangle([0, 0, width, height], fill=tuple(int(v) for v in tint))
        draw.rectangle([0, int(height * 0.72), width, height], fill=(118, 122, 126))
        n_objects = int(rng.integers(1, max_objects + 1))
        chosen = rng.permutation(len(slots))[:n_objects]
        boxes: list[list[float]] = []
        labels: list[str] = []
        for slot_index in chosen:
            cx, cy = slots[int(slot_index)]
            cx += float(rng.integers(-28, 29))
            cy += float(rng.integers(-28, 29))
            radius = float(rng.integers(46, 71))
            # Keep the whole sign inside the canvas: a clipped shape would make its reference box
            # loose, and a loose reference box turns every IoU and AP number below into noise.
            cx = min(max(cx, radius + 6), width - radius - 6)
            cy = min(max(cy, radius + 6), height - radius - 100)
            kind = SIGN_CLASSES[int(rng.integers(0, len(SIGN_CLASSES)))]
            draw.rectangle(
                [cx - 5, cy + radius * 0.6, cx + 5, min(height - 1, cy + radius + 90)], fill=(112, 112, 116)
            )
            if kind == "stop-sign":
                box = _octagon(draw, cx, cy, radius, _font(int(max(11, radius * 0.34))))
            elif kind == "yield-sign":
                box = _yield_sign(draw, cx, cy, radius)
            else:
                box = _speed_limit_sign(draw, cx, cy, radius, int(rng.choice([30, 50, 60, 80])))
            boxes.append(
                [
                    float(max(0.0, box[0])),
                    float(max(0.0, box[1])),
                    float(min(width, box[2])),
                    float(min(height, box[3])),
                ]
            )
            labels.append(kind)
        records.append({"image": image, "boxes": boxes, "labels": labels})
    return records
