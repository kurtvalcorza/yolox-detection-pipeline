"""Score-once, grid and reading-rule tests for the 2026-10-18 review fixes (row 116 R116-M1, R116-M2, R116-m4, R116-m6,
R116-S1; shared YOLOX design for rows 116 and 117).

The generated notebook's own Sections 6-9 are executed with a stand-in detector built on the `_StubPipe` of
tests/test_review_fixes.py: no weights and no torch call. The stand-in's average precision is computed by the real
`average_precision` on stand-in detections. None of this is model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib
import itertools
import json
import os
import subprocess
import types
from pathlib import Path

import pytest

from test_review_fixes import NOTEBOOK, PACKAGE, PIPELINE_CLASS, _code, _namespace, _StubPipe

pipeline_module = importlib.import_module(f"{PACKAGE}.pipeline")
samples_module = importlib.import_module(f"{PACKAGE}.samples")
protocol = importlib.import_module(f"{PACKAGE}.protocol")
ROOT = Path(__file__).resolve().parents[1]
STEM = NOTEBOOK.name.replace("_finetune_colab.ipynb", "")
NEEDLES = {
    6: "validation_records = sign_dataset(",
    7: "BASELINE_FINGERPRINT = ",
    8: "grid_rows, grid_models = {}, {}",
    9: "ledger = LookLedger(",
    10: "new_records = sign_dataset(3, seed=NEW_DATA_SEED)",
}


class _GridStub(_StubPipe):
    """`_StubPipe` plus a state fingerprint, per-config behaviour and real AP on stand-in detections."""

    fail_epochs: set[int] = set()
    adapted_quality = 1.0  # 1.0: boxes exact; 0.0: the adapted model is no better than the baseline

    def __init__(self, class_names=("a",), adapted=False) -> None:
        super().__init__(class_names, adapted)
        self.trained_with: tuple = ()

    def finetune(self, records, **kwargs):
        EVENTS.append("train")
        if kwargs.get("epochs") in type(self).fail_epochs:
            raise RuntimeError("CUDA out of memory (stand-in)")
        out = super().finetune(records, **kwargs)
        self.trained_with += (kwargs.get("epochs"), kwargs.get("learning_rate"))
        out["history"] = [{"epoch": 1, "total_loss": 2.0}, {"epoch": 2, "total_loss": 1.0}]
        return out

    def state_fingerprint(self):
        return hashlib.sha256(f"{self.adapted}:{self.trained_with}:{self.finetune_calls}".encode()).hexdigest()

    def _detections(self, record):
        good = self.adapted and type(self).adapted_quality > 0.5
        dets = []
        for box, label in zip(record["boxes"], record["labels"], strict=True):
            shift = 0.0 if good else (box[2] - box[0]) * 0.6
            dets.append({"label": label, "score": 0.9 if good else 0.4, "box": [box[0] + shift, box[1], box[2] + shift, box[3]]})
        return dets

    def evaluate(self, records, keep_matches=False, **_):
        predictions = [self._detections(r) for r in records]
        out = pipeline_module.average_precision(predictions, records, self.class_names)
        out["implementation"] = "stand-in"
        if keep_matches:
            out["matches"] = pipeline_module.match_detections(predictions, records, self.class_names)
        return out


EVENTS: list[str] = []


def _torch(available: bool, name: str = "Tesla T4"):
    """A torch stand-in with the CUDA query and the amendment-2 determinism switches (calls are logged in EVENTS)."""
    state = {"det": False, "warn_only": False}

    def use_deterministic_algorithms(flag, warn_only=False):
        state.update(det=bool(flag), warn_only=bool(warn_only))
        EVENTS.append("deterministic")

    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: available, get_device_name=lambda index=0: name,
                                   manual_seed_all=lambda seed: None),
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False, benchmark=True)),
        use_deterministic_algorithms=use_deterministic_algorithms,
        are_deterministic_algorithms_enabled=lambda: state["det"],
        is_deterministic_algorithms_warn_only_enabled=lambda: state["warn_only"],
        manual_seed=lambda seed: None,
    )


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell(notebook: dict, section: int) -> str:
    (source,) = [s for s in _code(notebook) if NEEDLES[section] in s]
    return source


@pytest.fixture
def harness(notebook, tmp_path, monkeypatch):
    _GridStub.fail_epochs, _GridStub.adapted_quality = set(), 1.0
    monkeypatch.delenv("DIMER_MASK_FRESH_TEST", raising=False)
    ns = _namespace(tmp_path, monkeypatch)
    ns.update({k: v for k, v in vars(protocol).items() if not k.startswith("__")})
    ns[PIPELINE_CLASS] = _GridStub
    ns["paired_bootstrap"] = functools.partial(protocol.paired_bootstrap, n_boot=200)
    # The record-less path (shared with row 117 until its record exists); the record tests below set it.
    ns["RECORDED_EVALUATION"] = None
    ns["torch"] = _torch(True, "Tesla T4")
    calls: list[int] = []
    real_sign = samples_module.sign_dataset

    def sign_dataset(n, *, seed=0, **kw):
        calls.append(seed)
        return real_sign(n, seed=seed, **kw)

    ns["sign_dataset"] = sign_dataset
    ns["_seeds"] = calls

    def run(sections=(6, 7, 8, 9), **edits):
        for section in sections:
            source = _cell(notebook, section)
            for old, new in edits.items():
                if old in source:
                    source = source.replace(old, new)
            exec(source, ns)  # noqa: S102 - the notebook's own cells
        return ns

    return run, ns, tmp_path / "outputs"


def _report(tmp_path: Path, suffix: str = "") -> dict:
    return json.loads((tmp_path / f"{STEM}_evaluation_report{suffix}.json").read_text(encoding="utf-8"))


# ---- the pre-registered path, and its reproduction ------------------------------------------------------------------


def test_default_gpu_run_scores_the_fresh_test_once_then_labels_a_rerun_a_reproduction(harness) -> None:
    run, ns, tmp = harness
    run()
    first = _report(tmp)
    assert first["test_look"] == protocol.TEST_LOOK_PREREGISTERED and first["preregistered"] is True
    assert first["settings_differing"] == [] and first["grid"]["complete"] is True
    assert first["fresh_test"]["verdict"] == "improved" and first["fresh_test"]["bootstrap"]["n_images"] == 200
    assert protocol.FRESH_TEST_POOL["seed"] in ns["_seeds"]
    before = (tmp / f"{STEM}_evaluation_report.json").read_bytes()
    run(sections=(9,))
    again = _report(tmp, "_reproduction_1")
    assert again["test_look"] == "reproduction" and again["fresh_test"] is not None
    assert (tmp / f"{STEM}_evaluation_report.json").read_bytes() == before
    assert {p.name for p in (tmp / "grid" / "run_1").iterdir()} == {"C0.json", "C1.json", "C2.json"}
    assert first["device"] == {"class": "cuda-t4", "name": "Tesla T4"}


@pytest.mark.parametrize(
    ("edit", "differing"),
    [
        ({"BATCH_SIZE = 2": "BATCH_SIZE = 1"}, "batch_size"),
        ({"\nSEED = 0  #": "\nSEED = 1  #"}, "split_seed"),
        ({"N_IMAGES = 40": "N_IMAGES = 36"}, "n_images"),
        ({"DATASET_SEED = 0": "DATASET_SEED = 2"}, "dataset_seed"),
        ({"FREEZE_BACKBONE = True": "FREEZE_BACKBONE = False"}, "freeze_backbone"),
        ({"HOLDOUT = 0.25": "HOLDOUT = 0.3"}, "holdout"),
    ],
)
def test_a_changed_setting_is_exploratory_and_never_scores_the_fresh_test(harness, edit, differing) -> None:
    run, ns, tmp = harness
    run(**edit)
    report = _report(tmp, "_exploratory_1")
    assert report["test_look"] == protocol.TEST_LOOK_EXPLORATORY and report["fresh_test"] is None
    assert differing in report["settings_differing"] and report["preregistered"] is False
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"]
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()


def test_every_fixed_setting_is_in_the_preregistered_check() -> None:
    for key in ("training_digest", "validation_digest", "fresh_test_digest", "batch_size", "adapt_seed", "freeze_backbone",
                "grid", "run_grid", "eval_threshold", "max_detections", "min_ap_gain", "keep_baseline_gain"):
        changed = dict(protocol.PREREGISTERED_SETTINGS, **{key: "changed"})
        assert protocol.preregistration_differences(changed) == [key]
    assert protocol.preregistration_differences({}) == list(protocol.PREREGISTERED_SETTINGS)


def test_recorded_evaluation_is_the_amendment_2_hosted_t4_record() -> None:
    """The 2026-10-18 Colab CLI T4 run of 4429e0e (docs/execution-evidence/2026-10-18-4429e0e/), as Sections 8 and 9 printed it."""
    record = protocol.RECORDED_EVALUATION
    assert record is not None and record is not protocol.ROUND2_RECORDED_EVALUATION
    assert record["fresh_pool"] == protocol.FRESH_TEST_POOL == {"n": 200, "seed": 1162}
    assert record["date"] == "2026-10-18" and record["device_class"] == "cuda-t4"
    assert record["commit"] == "4429e0e50372f14355fb23d801a754a57e3b364e"
    assert record["notebook_blob"] == "a9dd3990a1d8b18be6fbcc06ee218c19e0025b7c"
    assert record["selected_config"] == record["selected_system"] == "C1"
    assert record["validation_ap"] == {"baseline": 0.1767, "C0": 0.7037, "C1": 0.8937, "C2": 0.8296}
    assert record["grid_hash"] == "184785885a816d4becfec2789c796d112be29dbe5f5b4760f2ab7fdb39753793"
    assert record["fresh_test"] == {
        "n_images": 200, "n_references": 386, "selected_ap": 0.8933, "baseline_ap": 0.1317,
        "ap_difference": 0.7616, "ci95": [0.7381, 0.782], "verdict": "improved",
    }
    assert (ROOT / record["evidence"]).is_dir()


def test_round2_record_is_unchanged_history() -> None:
    """The round-2 Colab CLI T4 run (docs/execution-evidence/2026-10-18-ad795d2/), values as Section 9 printed them."""
    record = protocol.ROUND2_RECORDED_EVALUATION
    assert record["fresh_pool"] == protocol.EXPOSED_ROUND2_POOL == {"n": 200, "seed": 1160}
    assert record is not None and record["date"] == "2026-10-18" and record["device_class"] == "cuda-t4"
    assert record["commit"].startswith("ad795d2") and record["notebook_blob"].startswith("5fe2ef5d")
    assert record["selected_config"] == record["selected_system"] == "C1"
    assert record["validation_ap"] == {"baseline": 0.1767, "C0": 0.6966, "C1": 0.8554, "C2": 0.7122}
    assert record["fresh_test"] == {
        "n_images": 200, "n_references": 399, "selected_ap": 0.8573, "baseline_ap": 0.1363,
        "ap_difference": 0.7211, "ci95": [0.6983, 0.7435], "verdict": "improved",
    }
    assert (ROOT / record["evidence"]).is_dir()


def test_cpu_default_grid_off_trains_the_recorded_selection_and_is_exploratory(harness) -> None:
    run, ns, tmp = harness
    ns["torch"] = _torch(False)
    run()
    report = _report(tmp, "_exploratory_1")
    assert report["fresh_test"] is None and "run_grid" in report["settings_differing"]
    assert report["test_look"] == protocol.TEST_LOOK_EXPLORATORY and report["preregistered"] is False
    # The recorded selection follows the amendment-2 record (4429e0e selected C1, as ad795d2 did): ten epochs at 1e-3.
    assert list(ns["grid_rows"]) == [protocol.RECORDED_SELECTION] == ["C1"]
    assert protocol.RECORDED_EVALUATION["selected_config"] == protocol.RECORDED_SELECTION
    assert protocol.ROUND2_RECORDED_EVALUATION["selected_config"] == protocol.RECORDED_SELECTION
    assert ns["grid_rows"]["C1"]["settings"]["epochs"] == 10 and ns["grid_rows"]["C1"]["settings"]["learning_rate"] == 1e-3


def test_a_partial_grid_is_exploratory_and_not_a_look(harness) -> None:
    run, ns, tmp = harness
    _GridStub.fail_epochs = {8}
    run()
    report = _report(tmp, "_exploratory_1")
    assert report["grid"]["complete"] is False and any("C2: failed" in p for p in report["grid"]["problems"])
    assert report["fresh_test"] is None and report["grid_rows"]["C2"]["status"] == "failed"
    assert protocol.LookLedger(tmp / "work" / "look_ledger.json", tmp, STEM).completed() == 0


def test_a_changed_model_state_is_exploratory(harness) -> None:
    run, ns, tmp = harness
    run(sections=(6, 7, 8))
    ns["adapter"].finetune([], epochs=1)
    run(sections=(9,))
    assert _report(tmp, "_exploratory_1")["fresh_test"] is None
    assert "the model in hand is not the selected system of this Section 8 run" in _report(tmp, "_exploratory_1")["reasons"]


@pytest.mark.parametrize("failure", ["raise", "digest"])
def test_a_failed_generation_or_digest_check_is_a_friendly_refusal_and_not_a_look(harness, capsys, failure) -> None:
    run, ns, tmp = harness
    real_sign, real_digest = ns["sign_dataset"], ns["pool_digest"]
    if failure == "raise":
        ns["sign_dataset"] = lambda n, *, seed=0, **kw: (_ for _ in ()).throw(ValueError("stand-in")) if seed == protocol.FRESH_TEST_POOL["seed"] else real_sign(n, seed=seed, **kw)
    else:
        ns["pool_digest"] = lambda records: "0" * 64 if len(records) == protocol.FRESH_TEST_POOL["n"] else real_digest(records)
    run()
    out = capsys.readouterr().out
    assert "does not match its committed digest" in out and "Traceback" not in out
    report = _report(tmp, "_exploratory_1")
    assert report["fresh_test"] is None and "the fresh pool failed its digest check" in report["reasons"]
    ns["sign_dataset"], ns["pool_digest"] = real_sign, real_digest
    run(sections=(9,))
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED


def test_an_interrupted_report_write_is_not_a_look(harness) -> None:
    run, ns, tmp = harness
    real = ns["atomic_write_json"]

    def interrupted(path, payload):
        if "evaluation_report" in Path(path).name:
            raise KeyboardInterrupt
        return real(path, payload)

    ns["atomic_write_json"] = interrupted
    with pytest.raises(KeyboardInterrupt):
        run()
    assert not list(tmp.glob(f"{STEM}_evaluation_report*"))
    ns["atomic_write_json"] = real
    run(sections=(9,))
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED


@pytest.mark.parametrize("ledger_text", ["", "{not json", "[1, 2]", '{"preregistered_completed": "x"}', '{"preregistered_completed": NaN}', '{"preregistered_completed": Infinity}'])
def test_a_corrupt_or_empty_ledger_never_crashes_or_relabels(harness, ledger_text) -> None:
    run, ns, tmp = harness
    (tmp / "work").mkdir(parents=True)
    (tmp / "work" / "look_ledger.json").write_text(ledger_text, encoding="utf-8")
    run()
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED
    (tmp / "work" / "look_ledger.json").write_text(ledger_text, encoding="utf-8")
    run(sections=(9,))
    assert _report(tmp, "_reproduction_1")["test_look"] == "reproduction"


def test_an_existing_main_report_file_spends_the_slot_even_when_unreadable(harness) -> None:
    """Amendment 1 (A3), reviewer probe P3 inverted: report emptied and ledger corrupted after a look -> reproduction."""
    run, ns, tmp = harness
    run()
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED
    (tmp / f"{STEM}_evaluation_report.json").write_text("", encoding="utf-8")
    (tmp / "work" / "look_ledger.json").write_text("{bad", encoding="utf-8")
    run(sections=(9,))
    assert _report(tmp, "_reproduction_1")["test_look"] == "reproduction"
    assert (tmp / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8") == ""


def test_a_masked_run_never_generates_the_fresh_test(harness, monkeypatch) -> None:
    run, ns, tmp = harness
    monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    run()
    report = _report(tmp, "_masked_1")
    assert report["test_look"] == protocol.TEST_LOOK_MASKED and report["masked"] is True
    assert report["preregistered"] is False and report["preregistered_settings_hold"] is True
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"] and report["fresh_test"] is None


def test_the_keep_baseline_rule_selects_the_baseline_and_reports_same_model(harness) -> None:
    run, ns, tmp = harness
    _GridStub.adapted_quality = 0.0
    run()
    report = _report(tmp)
    assert report["selection"]["selected_system"] == protocol.BASELINE_SYSTEM
    assert report["fresh_test"]["verdict"] == "same model (adaptation not selected)" and report["fresh_test"]["bootstrap"] is None


def test_rerunning_the_finetune_cell_trains_fresh_models(harness) -> None:
    run, ns, tmp = harness
    run(sections=(6, 7, 8))
    first = ns["adapter"]
    run(sections=(8,), **{"FREEZE_BACKBONE = True": "FREEZE_BACKBONE = False"})
    assert ns["adapter"] is not first and first.finetune_calls == 1 and ns["adapter"].finetune_calls == 1
    assert ns["run"]["freeze_backbone"] is False and isinstance(ns["FINETUNE_SECONDS"], float)


# ---- selection, reading rule, tie rule ------------------------------------------------------------------------------


def _rows(**scores):
    return {n: {"status": "complete", "val_ap": ap, "val_ap50": ap50, "fingerprint": n} for n, (ap, ap50) in scores.items()}


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"C0": (0.50, 0.90), "C1": (0.529, 0.95), "C2": (0.52, 0.95)}, "C0"),  # below the 0.03 margin
        ({"C0": (0.50, 0.90), "C1": (0.54, 0.87), "C2": (0.52, 0.95)}, "C0"),  # AP50 more than 0.02 lower
        ({"C0": (0.50, 0.90), "C1": (0.54, 0.89), "C2": (0.52, 0.95)}, "C1"),
        ({"C0": (0.50, 0.90), "C1": (0.54, 0.90), "C2": (0.56, 0.90)}, "C2"),  # both qualify: the larger gain
        ({"C0": (0.50, 0.90), "C1": (0.55, 0.90), "C2": (0.55, 0.90)}, "C2"),  # tie: the smaller config
    ],
)
def test_selection_rule(scores, expected) -> None:
    assert protocol.select_config(_rows(**scores), baseline_val_ap=0.1)["selected_config"] == expected


def test_keep_baseline_rule_margin() -> None:
    rows = _rows(C0=(0.30, 0.9), C1=(0.30, 0.9), C2=(0.30, 0.9))
    assert protocol.select_config(rows, baseline_val_ap=0.28)["selected_system"] == protocol.BASELINE_SYSTEM
    assert protocol.select_config(rows, baseline_val_ap=0.27)["selected_system"] == "C0"


@pytest.mark.parametrize(
    ("ci", "diff", "verdict"),
    [([0.01, 0.2], 0.1, "improved"), ([0.001, 0.05], 0.02, "no detectable difference"), ([-0.2, -0.01], -0.1, "worse"), ([-0.1, 0.1], 0.0, "no detectable difference")],
)
def test_reading_verdict(ci, diff, verdict) -> None:
    assert protocol.reading_verdict({"primary_ap_ci95": ci, "primary_ap_difference": diff}, same_model=False) == verdict


def test_constant_scores_give_one_ap_whatever_the_image_order() -> None:
    """R116-S1: 2 hits and 2 false positives, all at score 0.5, used to give AP50 0.2525-0.505 by image order."""
    records = [{"boxes": [[0, 0, 10, 10]], "labels": ["a"]} for _ in range(4)]
    dets = [[{"label": "a", "score": 0.5, "box": [0, 0, 10, 10]}], [{"label": "a", "score": 0.5, "box": [0, 0, 10, 10]}],
            [{"label": "a", "score": 0.5, "box": [50, 50, 60, 60]}], [{"label": "a", "score": 0.5, "box": [50, 50, 60, 60]}]]
    values = {round(pipeline_module.average_precision([dets[i] for i in order], [records[i] for i in order], ["a"])["ap50"], 9)
              for order in itertools.permutations(range(4))}
    assert len(values) == 1
    result = pipeline_module.average_precision(dets, records, ["a"])
    assert result["tied_score_groups"] == 1 and "order-independent" in result["tie_rule"]


def test_distinct_scores_keep_the_per_detection_curve_and_the_bootstrap_reweights_images() -> None:
    records = [{"boxes": [[0, 0, 10, 10]], "labels": ["a"]} for _ in range(3)]
    dets = [[{"label": "a", "score": 0.9, "box": [0, 0, 10, 10]}], [{"label": "a", "score": 0.8, "box": [40, 40, 50, 50]}],
            [{"label": "a", "score": 0.7, "box": [0, 0, 10, 10]}]]
    result = pipeline_module.average_precision(dets, records, ["a"], iou_thresholds=(0.5,))
    assert result["ap50"] == pytest.approx((1.0 * 34 + (2 / 3) * 33) / 101)  # precision 1 up to recall 1/3, then 2/3
    matches = pipeline_module.match_detections(dets, records, ["a"], iou_thresholds=(0.5,))
    assert pipeline_module.ap_from_matches(matches, [1, 0, 1])["ap50"] == pytest.approx(1.0)
    boot = protocol.paired_bootstrap(matches, matches, n_boot=50)
    assert boot["primary_ap_difference"] == 0.0 and boot["primary_ap_ci95"] == [0.0, 0.0]


# ---- pins, atomic writes, provenance --------------------------------------------------------------------------------


def test_pool_digests_match_the_pins_and_the_pools_are_disjoint() -> None:
    seed0 = samples_module.sign_dataset(40, seed=0)
    train, exposed = pipeline_module.split_records(seed0, holdout=0.25, seed=0)
    val = samples_module.sign_dataset(protocol.VALIDATION_POOL["n"], seed=protocol.VALIDATION_POOL["seed"])
    fresh = samples_module.sign_dataset(protocol.FRESH_TEST_POOL["n"], seed=protocol.FRESH_TEST_POOL["seed"])
    assert protocol.pool_digest(train) == protocol.TRAINING_DIGEST
    assert protocol.pool_digest(exposed) == protocol.EXPOSED_DEV_TEST_DIGEST
    assert protocol.pool_digest(val) == protocol.VALIDATION_DIGEST
    assert protocol.pool_digest(fresh) == protocol.FRESH_TEST_DIGEST
    pools = [seed0, val, fresh, samples_module.sign_dataset(3, seed=99)]
    for a, b in itertools.combinations(pools, 2):
        assert not {protocol.record_digest(r) for r in a} & {protocol.record_digest(r) for r in b}
        assert not {protocol.layout_key(r) for r in a} & {protocol.layout_key(r) for r in b}


def test_atomic_write_replaces_whole_files_only(tmp_path: Path) -> None:
    target = tmp_path / "r.json"
    protocol.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"]
    assert protocol.read_json_or_none(tmp_path / "missing.json") is None


def test_notebook_stamp_names_a_commit_that_contains_the_carried_modules(notebook: dict) -> None:
    generated = notebook["metadata"]["dimer"]["generated_from"]
    revision = generated["revision"]
    for rel in generated["modules"]:
        try:
            committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{rel}"])
        except (OSError, subprocess.CalledProcessError):
            if os.environ.get("CI"):
                pytest.fail(f"the stamped commit {revision} is not in this checkout; CI must fetch full history")
            pytest.skip("the stamped commit is not in this clone (shallow checkout)")
        assert committed.decode("utf-8").replace("\r\n", "\n") == (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n"), rel
    assert notebook["metadata"]["dimer"]["notebook_kind"] == "TUTORIAL"


# ---- round 2: reviewer probes P1-P6 inverted, and F7 ------------------------------------------------------------


@pytest.mark.parametrize(("available", "name", "expected"), [(False, "cpu", "cpu"), (True, "NVIDIA L4", "cuda-other"), (True, "NVIDIA A100-SXM4-40GB", "cuda-other")])
def test_only_a_t4_can_score_the_fresh_test_whatever_run_grid_says(harness, available, name, expected) -> None:
    """Amendment 1 (A1), reviewer probe P1 inverted: RUN_GRID="on" on a CPU or a non-T4 GPU is exploratory."""
    run, ns, tmp = harness
    ns["torch"] = _torch(available, name)
    run(**{'RUN_GRID = "auto"': 'RUN_GRID = "on"'})
    report = _report(tmp, "_exploratory_1")
    assert report["fresh_test"] is None and report["device"]["class"] == expected
    assert "device_class" in report["settings_differing"] and report["grid"]["complete"] is True
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"]
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()


def test_exploratory_reruns_never_overwrite_earlier_grid_rows(harness) -> None:
    """Reviewer probe P4 inverted: one grid folder per Section 8 execution."""
    run, ns, tmp = harness
    run()
    before = (tmp / "grid" / "run_1" / "C0.json").read_bytes()
    run(sections=(8, 9), **{"FREEZE_BACKBONE = True": "FREEZE_BACKBONE = False"})
    assert (tmp / "grid" / "run_1" / "C0.json").read_bytes() == before
    assert json.loads((tmp / "grid" / "run_2" / "C0.json").read_text(encoding="utf-8"))["settings"]["freeze_backbone"] is False


class _Interrupting(_GridStub):
    interrupt_epochs: set[int] = set()

    def finetune(self, records, **kwargs):
        if kwargs.get("epochs") in type(self).interrupt_epochs:
            raise KeyboardInterrupt
        return super().finetune(records, **kwargs)


@pytest.mark.parametrize("epochs", [6, 10, 8])  # C0, C1, C2
def test_an_interrupted_section8_rerun_leaves_nothing_section9_can_score(harness, epochs) -> None:
    """Amendment 1 (A2), reviewer probe P6 inverted: interrupt during C0, C1 or C2 after an exploratory run."""
    run, ns, tmp = harness
    ns[PIPELINE_CLASS] = _Interrupting
    _Interrupting.interrupt_epochs = set()
    run(**{"FREEZE_BACKBONE = True": "FREEZE_BACKBONE = False"})
    assert _report(tmp, "_exploratory_1")["fresh_test"] is None
    _Interrupting.interrupt_epochs = {epochs}
    with pytest.raises(KeyboardInterrupt):
        run(sections=(8,))
    _Interrupting.interrupt_epochs = set()
    assert ns["selection"] is None and ns["adapter"] is None and ns["grid_status"]["complete"] is False
    with pytest.raises(RuntimeError, match="Section 8 did not finish"):
        run(sections=(9,))
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"]
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()
    run(sections=(8, 9))  # a complete re-run restores the pre-registered path
    assert _report(tmp)["test_look"] == protocol.TEST_LOOK_PREREGISTERED


def test_stale_selection_or_rows_left_in_memory_are_not_trusted(harness) -> None:
    """A2 variants: Section 9 recomputes the grid check and the selection-to-row fingerprint match."""
    run, ns, tmp = harness
    run(sections=(6, 7, 8))
    ns["grid_status"] = {"complete": True, "problems": []}
    del ns["grid_rows"]["C2"]
    run(sections=(9,))
    report = _report(tmp, "_exploratory_1")
    assert report["fresh_test"] is None and any("C2: not run" in r for r in report["reasons"])
    run(sections=(8,))
    ns["selection"] = dict(ns["selection"], fingerprint="stale")
    run(sections=(9,))
    assert _report(tmp, "_exploratory_2")["fresh_test"] is None


def test_keep_baseline_sections_10_and_11_use_the_selected_baseline(harness, notebook, capsys) -> None:
    """F7: when the rule keeps the baseline, Section 10 says so and runs the baseline; Section 11 exports it."""
    run, ns, tmp = harness
    _GridStub.adapted_quality = 0.0
    run(sections=(6, 7, 8, 9, 10))
    out = capsys.readouterr().out
    assert "kept the pretrained baseline" in out and ns["selected_pipe"] is ns["baseline_pipe"]
    section11 = next(s for s in _code(notebook) if "reloaded = YoloxDetectionPipeline.load_artifact(artifact_path)" in s or "load_artifact(artifact_path)" in s)
    assert "selected_pipe.save_artifact(" in section11 and "adapter.save_artifact(" not in section11
    assert "selected_val['ap']" in section11


def test_device_class_helper() -> None:
    assert protocol.device_class(_torch(False)) == "cpu"
    assert protocol.device_class(_torch(True, "Tesla T4")) == "cuda-t4"
    assert protocol.device_class(_torch(True, "NVIDIA L4")) == "cuda-other"
    assert protocol.PREREGISTERED_SETTINGS["device_class"] == "cuda-t4"


# ---- with the evaluation of record set (2026-10-18) -----------------------------------------------------------------


def test_with_the_record_the_first_run_in_a_new_runtime_is_a_reproduction_and_never_writes_the_main_files(harness, capsys) -> None:
    run, ns, tmp = harness
    ns["RECORDED_EVALUATION"] = protocol.RECORDED_EVALUATION  # the module value: the amendment-2 record
    run()
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()
    assert not (tmp / f"{STEM}_fresh_test_per_image.json").exists()
    first = _report(tmp, "_reproduction_1")
    assert first["test_look"] == protocol.TEST_LOOK_REPRODUCTION == "reproduction" and first["preregistered"] is True
    assert first["fresh_test"] is not None and first["recorded_evaluation"] == protocol.RECORDED_EVALUATION
    assert first["fresh_test_pool"] == {"n": 200, "seed": 1162}
    comparison = first["record_comparison"]
    assert comparison["record"]["commit"].startswith("4429e0e") and comparison["matches"] is False
    assert "selected_ap" in comparison["differing"]  # the stand-in detector is not the record
    out = capsys.readouterr().out
    assert "reproduction of the pre-registered evaluation of 2026-10-18 (commit 4429e0e): differs from the record in" in out
    assert (tmp / f"{STEM}_fresh_test_per_image_reproduction_1.json").is_file()
    run(sections=(9,))
    assert _report(tmp, "_reproduction_2")["test_look"] == "reproduction"
    assert not (tmp / f"{STEM}_evaluation_report.json").exists()


def test_with_the_record_a_changed_setting_is_still_exploratory(harness) -> None:
    run, ns, tmp = harness
    ns["RECORDED_EVALUATION"] = protocol.RECORDED_EVALUATION
    run(**{"BATCH_SIZE = 2": "BATCH_SIZE = 1"})
    report = _report(tmp, "_exploratory_1")
    assert report["test_look"] == protocol.TEST_LOOK_EXPLORATORY and report["fresh_test"] is None
    assert "record_comparison" not in report and not list(tmp.glob(f"{STEM}_evaluation_report_reproduction_*.json"))


def test_ledger_plan_with_a_record_never_returns_the_main_slot(tmp_path) -> None:
    ledger = protocol.LookLedger(tmp_path / "work" / "ledger.json", tmp_path, STEM)
    assert ledger.plan("preregistered") == (protocol.TEST_LOOK_PREREGISTERED, "")
    assert ledger.plan("preregistered", protocol.ROUND2_RECORDED_EVALUATION) == ("reproduction", "_reproduction_1")
    ledger.record("preregistered", "_reproduction_1")
    assert ledger.plan("preregistered", protocol.ROUND2_RECORDED_EVALUATION) == ("reproduction", "_reproduction_2")
    assert ledger.plan("exploratory", protocol.ROUND2_RECORDED_EVALUATION) == (protocol.TEST_LOOK_EXPLORATORY, "_exploratory_1")


def test_compare_with_record_reports_a_match_and_each_differing_field() -> None:
    record = protocol.ROUND2_RECORDED_EVALUATION
    fresh = {
        "selected": {"ap": 0.85731}, "baseline": {"ap": 0.13628}, "verdict": "improved",
        "bootstrap": {"primary_ap_difference": 0.72114, "primary_ap_ci95": [0.69834, 0.74349]},
    }
    selection = {"selected_config": "C1", "selected_system": "C1"}
    same = protocol.compare_with_record(record, fresh, selection)
    assert same["matches"] is True and same["differing"] == []
    other = protocol.compare_with_record(record, dict(fresh, selected={"ap": 0.85}), {"selected_config": "C0", "selected_system": "C0"})
    assert other["matches"] is False and other["differing"] == ["selected_config", "selected_system", "selected_ap"]
    assert protocol.compare_with_record(None, fresh, selection) is None
    # The amendment-2 record: a run that prints its numbers (to 4 decimals) matches it.
    fresh3 = {
        "selected": {"ap": 0.89331}, "baseline": {"ap": 0.13168}, "verdict": "improved",
        "bootstrap": {"primary_ap_difference": 0.76163, "primary_ap_ci95": [0.73812, 0.78198]},
    }
    match3 = protocol.compare_with_record(protocol.RECORDED_EVALUATION, fresh3, selection)
    assert match3["matches"] is True and match3["record"]["commit"].startswith("4429e0e")
    assert protocol.compare_with_record(protocol.RECORDED_EVALUATION, fresh, selection)["matches"] is False
    assert protocol.compare_with_record(record, None, selection) is None


def test_the_saved_per_image_tables_rerun_the_bootstrap_exactly(harness) -> None:
    run, ns, tmp = harness
    run()
    report = _report(tmp)
    saved = json.loads((tmp / f"{STEM}_fresh_test_per_image.json").read_text(encoding="utf-8"))
    assert report["fresh_test"]["per_image_file"].endswith(f"{STEM}_fresh_test_per_image.json")
    rows = saved["per_image"]
    assert len(rows) == 200 and sum(r["references"] for r in rows) == report["fresh_test"]["selected"]["n_references"]
    assert saved["bootstrap"] == {"n_boot": 200, "seed": protocol.BOOTSTRAP_SEED, "resampling_unit": "image"}
    selected = protocol.matches_from_json(saved["matches"]["selected"])
    baseline = protocol.matches_from_json(saved["matches"]["baseline"])
    again = protocol.paired_bootstrap(selected, baseline, n_boot=200)
    assert again["primary_ap_ci95"] == report["fresh_test"]["bootstrap"]["primary_ap_ci95"]
    assert again["primary_ap_difference"] == report["fresh_test"]["bootstrap"]["primary_ap_difference"]


# ---- round 3: amendment 2 (deterministic GPU training, fresh test 3) ----------------------------------------------


def test_determinism_is_switched_on_before_any_training_and_recorded(harness) -> None:
    run, ns, tmp = harness
    EVENTS.clear()
    run()
    assert EVENTS[0] == "deterministic" and EVENTS.index("deterministic") < EVENTS.index("train")
    report = _report(tmp)
    assert report["determinism"] == protocol.DETERMINISM_EXPECTED == protocol.PREREGISTERED_SETTINGS["determinism"]
    assert report["determinism"]["warn_only"] is False and report["determinism"]["cublas_workspace_config"] == ":4096:8"
    assert ns["torch"].backends.cudnn.deterministic is True and ns["torch"].backends.cudnn.benchmark is False


def test_changed_determinism_flags_are_exploratory(harness, monkeypatch) -> None:
    run, ns, tmp = harness
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    run()
    report = _report(tmp, "_exploratory_1")
    assert "determinism" in report["settings_differing"] and report["fresh_test"] is None


def test_an_op_without_a_deterministic_kernel_stops_the_grid(harness) -> None:
    run, ns, tmp = harness

    class _NoKernel(_GridStub):
        def finetune(self, records, **kwargs):
            raise RuntimeError("index_add_cuda_ does not have a deterministic implementation, but you set "
                               "'torch.use_deterministic_algorithms(True)'")

    ns[PIPELINE_CLASS] = _NoKernel
    with pytest.raises(RuntimeError, match="no deterministic CUDA kernel; STOP"):
        run()
    assert not list(tmp.glob(f"{STEM}_evaluation_report*"))


def test_two_identical_runs_print_the_same_grid_hash(harness, capsys) -> None:
    run, ns, tmp = harness
    run(sections=(6, 7, 8))
    first = ns["GRID_HASH"]
    run(sections=(8,))
    assert ns["GRID_HASH"] == first and len(first) == 64
    assert capsys.readouterr().out.count(f"grid hash {first}") == 2
    assert protocol.grid_hash({"C0": {"val_ap": 1.0, "finetune_seconds": 1.0}}) == protocol.grid_hash({"C0": {"val_ap": 1.0, "finetune_seconds": 9.0}})
    assert protocol.grid_hash({"C0": {"loss_curve": [2.0, 1.0]}}) != protocol.grid_hash({"C0": {"loss_curve": [2.0, 1.1]}})


def test_masked_determinism_check_prints_the_hash_and_never_generates_fresh_test_3(harness, monkeypatch, capsys) -> None:
    run, ns, tmp = harness
    monkeypatch.setenv("DIMER_MASK_FRESH_TEST", "1")
    run()
    report = _report(tmp, "_masked_1")
    assert report["grid_hash"] == ns["GRID_HASH"] and f"grid hash {ns['GRID_HASH']}" in capsys.readouterr().out
    assert protocol.FRESH_TEST_POOL["seed"] not in ns["_seeds"] and report["fresh_test"] is None


def test_the_first_fresh_test_3_scoring_is_the_preregistered_evaluation(harness) -> None:
    run, ns, tmp = harness
    ns["RECORDED_EVALUATION"] = None  # the record-less state the amendment-2 record (4429e0e) itself ran in
    run()
    report = _report(tmp)
    assert report["test_look"] == protocol.TEST_LOOK_PREREGISTERED and report["fresh_test_pool"] == {"n": 200, "seed": 1162}
    assert protocol.FRESH_TEST_POOL["seed"] == 1162 and 1162 in ns["_seeds"] and 1160 not in ns["_seeds"]


def test_fresh_test_3_is_pinned_and_disjoint_from_every_earlier_pool() -> None:
    fresh3 = samples_module.sign_dataset(200, seed=1162)
    assert protocol.pool_digest(fresh3) == protocol.FRESH_TEST_DIGEST
    exposed2 = samples_module.sign_dataset(200, seed=1160)
    assert protocol.pool_digest(exposed2) == protocol.EXPOSED_ROUND2_DIGEST
    earlier = samples_module.sign_dataset(40, seed=0) + samples_module.sign_dataset(40, seed=1161) + exposed2 + samples_module.sign_dataset(3, seed=99)
    assert not {protocol.record_digest(r) for r in fresh3} & {protocol.record_digest(r) for r in earlier}
    assert not {protocol.layout_key(r) for r in fresh3} & {protocol.layout_key(r) for r in earlier}

