"""The tutorial's pre-registered selection and fresh-test rules (docs/yolox-preregistration.md, 2026-10-18).

`PREREGISTERED_SETTINGS` lists every setting the pre-registration fixes. A run is `preregistered` only when all
of them hold, the validation grid is complete and ran with them, the model in hand is the selected system, and the
generated fresh pool matches its committed digest; only such a run scores the fresh test. Every other run is
exploratory: validation and the labelled exposed development test only, written to `*_exploratory_<n>` files. A
repeat of a pre-registered scoring in the same runtime is a reproduction and writes `*_reproduction_<n>` files; it
never overwrites the first one. A runtime cannot see other runtimes: docs/release-verification.md says which hosted
run is the evaluation of record (`RECORDED_EVALUATION`, None until that run exists).

Shared design for the YOLOX tutorials (rows 116 and 117): only `VALIDATION_POOL`, `FRESH_TEST_POOL` and the
digests differ between the two repositories.
"""

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import (
    COCO_IOU_THRESHOLDS,
    EVAL_DETECTION_THRESHOLD,
    EVAL_NMS_THRESHOLD,
    MAX_EVAL_DETECTIONS,
    ap_from_matches,
    prepared_matches,
)

# ---- pools (drawn by samples.sign_dataset; digests computed under the pinned Pillow 11.3.0) ------------------------
VALIDATION_POOL = {"n": 40, "seed": 1161}
FRESH_TEST_POOL = {"n": 200, "seed": 1160}
TRAINING_DIGEST = "60b0aaeea62cf9bb0498af03c1b99b43c644a4a09fee5e196e4d69dc9aef94bc"
EXPOSED_DEV_TEST_DIGEST = "8729a7c00ce71282f406a899e88bc63b713795932240257de6545d82cb34eef8"
VALIDATION_DIGEST = "a75c4381677c522178acfd3b16125c064bbc5b9514fce7a5d9d71f90d8e35a68"
FRESH_TEST_DIGEST = "6900526c80ac2fbfd983b114d803512cbafca271250cab22ecd428fcbe9415ac"

# ---- grid and selection --------------------------------------------------------------------------------------------
GRID: tuple[dict[str, Any], ...] = (
    {"name": "C0", "epochs": 6, "learning_rate": 1e-3},
    {"name": "C1", "epochs": 10, "learning_rate": 1e-3},
    {"name": "C2", "epochs": 8, "learning_rate": 2e-3},
)
DEFAULT_CONFIG = "C0"
RECORDED_SELECTION = "C0"  # what a grid-off (CPU) run trains; C0 until a hosted record selects otherwise
TIE_ORDER = ("C0", "C2", "C1")  # exact ties go to the smaller config (fewer epochs)
TIE_TOLERANCE = 1e-9
MIN_AP_GAIN = 0.03  # a challenger must beat C0's validation AP@[.50:.95] by this much
MAX_AP50_DROP = 0.02  # ... and lose at most this much validation AP50
KEEP_BASELINE_GAIN = 0.03  # the adapted model replaces the pretrained baseline only above this validation AP gain
BASELINE_SYSTEM = "baseline"

# ---- reading rule --------------------------------------------------------------------------------------------------
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261018
IMPROVED_MIN_GAIN = 0.03

PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": "sample",
    "n_images": 40,
    "dataset_seed": 0,
    "holdout": 0.25,
    "split_seed": 0,
    "training_digest": TRAINING_DIGEST,
    "validation_pool": dict(VALIDATION_POOL),
    "validation_digest": VALIDATION_DIGEST,
    "fresh_test_pool": dict(FRESH_TEST_POOL),
    "fresh_test_digest": FRESH_TEST_DIGEST,
    "batch_size": 2,
    "adapt_seed": 0,
    "freeze_backbone": True,
    "eval_threshold": EVAL_DETECTION_THRESHOLD,
    "eval_nms_threshold": EVAL_NMS_THRESHOLD,
    "max_detections": MAX_EVAL_DETECTIONS,
    "iou_thresholds": [float(t) for t in COCO_IOU_THRESHOLDS],
    "grid": [dict(c) for c in GRID],
    "run_grid": True,
    "min_ap_gain": MIN_AP_GAIN,
    "max_ap50_drop": MAX_AP50_DROP,
    "keep_baseline_gain": KEEP_BASELINE_GAIN,
}

# The hosted T4 run that is the evaluation of record (date, notebook blob, numbers); None until it exists.
RECORDED_EVALUATION: dict[str, Any] | None = None

TEST_LOOK_PREREGISTERED = "pre-registered evaluation (this runtime)"
TEST_LOOK_REPRODUCTION = "reproduction"
TEST_LOOK_EXPLORATORY = "not scored (exploratory)"
TEST_LOOK_MASKED = "not scored (fresh test masked for a development check)"


def record_digest(record: Mapping[str, Any]) -> str:
    """sha256 over one record's decoded pixels (mode, size, RGB bytes)."""
    image = record["image"]
    digest = hashlib.sha256(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def layout_key(record: Mapping[str, Any]) -> str:
    """The record's box layout (labels and boxes rounded to 0.1 px), for the layout-disjointness check."""
    parts = [f"{label}@{','.join(f'{v:.1f}' for v in box)}" for box, label in zip(record["boxes"], record["labels"], strict=True)]
    return ";".join(sorted(parts))


def pool_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """sha256 over the ordered per-record pixel digests and box layouts."""
    lines = [f"{i}:{record_digest(r)}:{layout_key(r)}" for i, r in enumerate(records)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def preregistration_differences(current: Mapping[str, Any]) -> list[str]:
    """Every pre-registered setting whose current value differs (a missing setting differs)."""
    missing = object()
    return [key for key, value in PREREGISTERED_SETTINGS.items() if current.get(key, missing) != value]


def config_record(
    config: Mapping[str, Any],
    run: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    *,
    digests: Mapping[str, str],
    fingerprint: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """One grid row, validation only: the settings it ran with and its validation scores."""
    row: dict[str, Any] = {"name": config["name"], "config": dict(config), "digests": dict(digests)}
    if error is not None or run is None or validation is None:
        row.update({"status": "failed", "error": error or "no result"})
        return row
    row.update(
        {
            "status": "complete",
            "settings": {k: run[k] for k in ("epochs", "learning_rate", "batch_size", "seed", "freeze_backbone")},
            "val_ap": float(validation["ap"]),
            "val_ap50": float(validation["ap50"]),
            "final_total_loss": float(run["history"][-1]["total_loss"]) if run.get("history") else None,
            "fingerprint": fingerprint,
        }
    )
    return row


def grid_check(
    grid: Mapping[str, Mapping[str, Any]] | None,
    settings: Mapping[str, Any],
    digests: Mapping[str, str],
) -> dict[str, Any]:
    """Is the grid complete, run with the current fixed settings, on the current training and validation data?"""
    problems: list[str] = []
    grid = grid or {}
    for config in GRID:
        row = grid.get(config["name"])
        if row is None:
            problems.append(f"{config['name']}: not run")
            continue
        if row.get("status") != "complete":
            problems.append(f"{config['name']}: {row.get('status')} ({row.get('error')})")
            continue
        expected = {
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "batch_size": settings.get("batch_size"),
            "seed": settings.get("adapt_seed"),
            "freeze_backbone": settings.get("freeze_backbone"),
        }
        if row.get("settings") != expected:
            problems.append(f"{config['name']}: ran with {row.get('settings')}, the pre-registered settings imply {expected}")
        if dict(row.get("digests", {})) != dict(digests):
            problems.append(f"{config['name']}: ran on different training or validation data")
    extra = sorted(set(grid) - {c["name"] for c in GRID})
    if extra:
        problems.append(f"configs outside the pre-registered grid: {extra}")
    return {"complete": not problems, "problems": problems}


def select_config(grid: Mapping[str, Mapping[str, Any]], baseline_val_ap: float) -> dict[str, Any]:
    """The pre-registered choice (validation only): C0 unless a challenger qualifies; then baseline vs adapted."""
    done = {name: row for name, row in grid.items() if row.get("status") == "complete"}
    if not done:
        raise ValueError("no grid config completed")
    base_name = DEFAULT_CONFIG if DEFAULT_CONFIG in done else min(done, key=TIE_ORDER.index)
    base = done[base_name]
    gains = {
        name: row["val_ap"] - base["val_ap"]
        for name, row in done.items()
        if name != base_name
        and row["val_ap"] - base["val_ap"] >= MIN_AP_GAIN - TIE_TOLERANCE
        and base["val_ap50"] - row["val_ap50"] <= MAX_AP50_DROP + TIE_TOLERANCE
    }
    if gains:
        best = max(gains.values())
        tied = [n for n, g in gains.items() if best - g <= TIE_TOLERANCE]
        config = min(tied, key=TIE_ORDER.index)
        reason = (
            f"{config} beat {base_name} by {gains[config]:.4f} validation AP (margin {MIN_AP_GAIN}) with AP50 at most "
            f"{MAX_AP50_DROP} lower"
        )
    else:
        config = base_name
        reason = f"no challenger beat {base_name} by {MIN_AP_GAIN} validation AP with AP50 at most {MAX_AP50_DROP} lower"
    adapted_gain = done[config]["val_ap"] - baseline_val_ap
    keeps_adapted = adapted_gain >= KEEP_BASELINE_GAIN - TIE_TOLERANCE
    return {
        "selected_config": config,
        "selected_system": config if keeps_adapted else BASELINE_SYSTEM,
        "reason": reason,
        "adapted_gain_over_baseline": adapted_gain,
        "keep_baseline_rule": (
            f"{config} beats the pretrained baseline by {adapted_gain:.4f} validation AP "
            f"({'>=' if keeps_adapted else '<'} {KEEP_BASELINE_GAIN}): "
            + ("the adapted model is the selected system" if keeps_adapted else "the baseline is kept (a negative result)")
        ),
        "scores": {n: {"val_ap": r["val_ap"], "val_ap50": r["val_ap50"]} for n, r in done.items()},
        "fingerprint": done[config].get("fingerprint"),
    }


def paired_bootstrap(
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired image-level bootstrap of AP@[.50:.95] (selected - baseline), 95 % percentile interval.

    `selected` and `baseline` are `match_detections` tables over the same images; each resample draws images with
    replacement and recomputes both APs with the same multiplicities, so an image moves with all its boxes."""
    n = selected["n_images"]
    if baseline["n_images"] != n or selected["references"] != baseline["references"]:
        raise ValueError("the two systems must be scored on the same images in the same order")
    prep_sel, prep_base = prepared_matches(selected), prepared_matches(baseline)
    point_sel = ap_from_matches(selected, prepared=prep_sel)
    point_base = ap_from_matches(baseline, prepared=prep_base)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        weights = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        diffs[b] = ap_from_matches(selected, weights, prepared=prep_sel)["ap"] - ap_from_matches(baseline, weights, prepared=prep_base)["ap"]
    low, high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "n_images": n,
        "resampling_unit": "image",
        "n_boot": n_boot,
        "seed": seed,
        "selected_ap": point_sel["ap"],
        "baseline_ap": point_base["ap"],
        "primary_ap_difference": point_sel["ap"] - point_base["ap"],
        "primary_ap_ci95": [low, high],
        "ci_half_width": (high - low) / 2,
        "bootstrap_se": float(diffs.std(ddof=1)),
        "secondary_ap50_difference": point_sel["ap50"] - point_base["ap50"],
    }


def reading_verdict(boot: Mapping[str, Any] | None, *, same_model: bool) -> str:
    """The pre-registered verdict, from the primary AP difference and its CI only."""
    if same_model:
        return "same model (adaptation not selected)"
    low, high = boot["primary_ap_ci95"]
    if low > 0 and boot["primary_ap_difference"] >= IMPROVED_MIN_GAIN:
        return "improved"
    if high < 0:
        return "worse"
    return "no detectable difference"


def scoring_mode(
    *, differences: Sequence[str], grid: Mapping[str, Any], state_matches: bool, fresh_digest_ok: bool | None, masked: bool
) -> str:
    """'preregistered' only when every check holds; 'masked' for a development check; else 'exploratory'.

    `fresh_digest_ok` is None when the fresh pool was not generated (masked, or an earlier check already failed)."""
    if differences or not grid.get("complete") or not state_matches:
        return "exploratory"
    if masked:
        return "masked"
    return "preregistered" if fresh_digest_ok else "exploratory"


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write via a temporary file and os.replace: a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def read_json_or_none(path: str | Path) -> Any:
    """The parsed file, or None when it is missing, empty or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class LookLedger:
    """Counts fresh-test scorings in this runtime, crash-safely.

    A look is recorded only by `record()`, after the report is safely on disk. Whether the pre-registered slot is
    spent is decided from the outputs as well as the ledger: a readable main report means every later scoring is a
    reproduction, even if the ledger was lost; a missing or unreadable main report is (re)written unless the ledger
    proves a completed look. A corrupt or empty ledger reads as empty and never crashes a run.
    """

    def __init__(self, path: str | Path, outputs_dir: str | Path = "outputs", stem: str = "yolox_detection") -> None:
        self.path = Path(path)
        self.outputs = Path(outputs_dir)
        self.stem = stem

    def _load(self) -> dict[str, int]:
        data = read_json_or_none(self.path)
        if not isinstance(data, dict):
            return {}
        return {k: int(v) for k, v in data.items() if isinstance(v, int | float) and not isinstance(v, bool)}

    def _save(self, counts: Mapping[str, int]) -> None:
        atomic_write_text(self.path, json.dumps(dict(counts)))

    @property
    def main_report(self) -> Path:
        return self.outputs / f"{self.stem}_evaluation_report.json"

    def _highest(self, kind: str) -> int:
        pattern = re.compile(rf"^{re.escape(self.stem)}_evaluation_report_{kind}_(\d+)\.json$")
        found = [
            int(m.group(1))
            for f in (self.outputs.glob("*.json") if self.outputs.is_dir() else [])
            if (m := pattern.match(f.name))
        ]
        return max(found, default=0)

    def plan(self, mode: str) -> tuple[str, str]:
        """(test_look, file suffix) for a scoring of kind `mode`. Records nothing."""
        if mode not in ("preregistered", "exploratory", "masked"):
            raise ValueError(f"unknown scoring mode {mode!r}")
        counts = self._load()
        if mode == "preregistered":
            if read_json_or_none(self.main_report) is not None or counts.get("preregistered_completed", 0) > 0:
                n = max(self._highest("reproduction"), counts.get("reproductions", 0)) + 1
                return TEST_LOOK_REPRODUCTION, f"_reproduction_{n}"
            return TEST_LOOK_PREREGISTERED, ""
        n = max(self._highest(mode), counts.get(mode, 0)) + 1
        return (TEST_LOOK_MASKED if mode == "masked" else TEST_LOOK_EXPLORATORY), f"_{mode}_{n}"

    def record(self, mode: str, suffix: str) -> None:
        """Count a completed scoring; call it only after its report is on disk."""
        counts = self._load()
        if mode == "preregistered":
            if suffix == "":
                counts["preregistered_completed"] = counts.get("preregistered_completed", 0) + 1
            else:
                counts["reproductions"] = max(counts.get("reproductions", 0), int(suffix.rsplit("_", 1)[1]))
        else:
            counts[mode] = max(counts.get(mode, 0), int(suffix.rsplit("_", 1)[1]))
        self._save(counts)

    def completed(self) -> int:
        counts = self._load()
        return counts.get("preregistered_completed", 0) + counts.get("reproductions", 0)
