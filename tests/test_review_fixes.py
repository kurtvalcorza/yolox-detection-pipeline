"""Regression tests for the 2026-10-05 notebook review findings (YXD-M1..M3, YXD-m1..m3).

They need only CI's lightweight dependencies (numpy, Pillow, pytest): the BYOD readers run on real drawn images, and the
generated notebook's own learner cells are executed with a stand-in detector (no weights, no torch call). None of this
is model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import re
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"yolox_fix_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
PACKAGE = TEMPLATE["package"]
PREFIX = "YXX" if "yolox_x" in PACKAGE else "YXD"
pipeline_module = importlib.import_module(f"{PACKAGE}.pipeline")
samples_module = importlib.import_module(f"{PACKAGE}.samples")
protocol_module = importlib.import_module(f"{PACKAGE}.protocol")
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
PIPELINE_CLASS = TEMPLATE["pipeline_class"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


def _cell(notebook: dict, needle: str) -> str:
    (source,) = [s for s in _code(notebook) if needle in s]
    return source


class _StubPipe:
    """Stand-in detector: deterministic scores, an artifact that round-trips through a JSON file."""

    def __init__(self, class_names=("a",), adapted=False) -> None:
        self.class_names, self.adapted = tuple(class_names), adapted
        self.device, self.source, self.reinitialised = "cpu", "stand-in", ()
        self.finetune_calls = 0

    @classmethod
    def from_pretrained(cls, weights_dir=None, class_names=("a",), seed=0, **_):
        return cls(class_names)

    @classmethod
    def load_artifact(cls, path):
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(meta["class_names"], adapted=meta["adapted"])

    def evaluate(self, records):
        value = 0.9 if self.adapted else 0.2
        return {"ap": value, "ap50": value, "per_class_ap50": {}, "n_images": len(records)}

    def finetune(self, records, **kwargs):
        self.finetune_calls += 1
        self.adapted = True
        return {"history": [{"total_loss": 2.0}, {"total_loss": 1.0}], "epochs": kwargs.get("epochs"), "optimizer": "SGD", "loss": "simota", "use_l1": False, "freeze_backbone": kwargs.get("freeze_backbone"), "trainable_parameters": 1, "total_parameters": 2, "batch_size": kwargs.get("batch_size"), "learning_rate": kwargs.get("learning_rate"), "seed": kwargs.get("seed")}

    def save_artifact(self, path, notes=""):
        Path(path).write_text(json.dumps({"class_names": list(self.class_names), "adapted": self.adapted}), encoding="utf-8")
        return {"format": "stand-in", "path": str(path)}

    def detect(self, image, threshold=0.3, nms_threshold=0.3):
        return {"detections": [{"label": self.class_names[0], "score": 0.5, "box": [0.0, 0.0, 4.0, 4.0]}], "class_names": list(self.class_names)}


def _namespace(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    ns = {k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")}
    ns.update({k: v for k, v in vars(samples_module).items() if not k.startswith("__")})
    ns.update({k: v for k, v in vars(protocol_module).items() if not k.startswith("__")})
    ns["selection"] = {"selected_config": "C0"}
    ns.update(json=json, io=io, Path=Path, Image=Image, OUTPUTS=tmp_path, WEIGHTS_DIR=tmp_path / "w", threshold=0.3, nms_threshold=0.3, EPOCHS=1, HOLDOUT=0.25, SEED=0, BATCH_SIZE=2, LEARNING_RATE=1e-3, FREEZE_BACKBONE=True, NOTEBOOK_SOURCE={"repository_revision": "r"}, pipe=_StubPipe(("stop sign",)))
    ns[PIPELINE_CLASS] = _StubPipe
    return ns


def _byod_dir(tmp_path: Path, n: int, *, missing: str | None = None, escape: bool = False) -> Path:
    directory = tmp_path / "byod"
    directory.mkdir()
    entries = []
    for record in samples_module.sign_dataset(n, seed=3):
        name = f"img{len(entries):02d}.png"
        if name != missing:
            record["image"].save(directory / name)
        entries.append({"file": name, "boxes": [list(map(float, b)) for b in record["boxes"]], "labels": list(record["labels"])})
    if escape:
        entries[0]["file"] = "../outside.png"
    (directory / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")
    return directory


def _byod_cell(notebook: dict, **changes: str) -> str:
    source = _cell(notebook, "USE_BYOD_DATASET = False")
    for old, new in changes.items():
        source = source.replace(old, new)
    return source


# ---- M1: isolated runtime, no restart, gate 6 annotated -------------------------------------------------------------


def test_M1_no_kernel_install_no_restart_and_a_hash_lock(notebook: dict) -> None:
    assert "Restart the runtime" not in NOTEBOOK.read_text(encoding="utf-8")
    sources = _code(notebook)
    assert not any("[sys.executable, '-m', 'pip'" in s for s in sources)
    kernel = [s for s in sources if "# dimer: kernel cell" in s]
    assert len(kernel) == 2
    install = next(s for s in kernel if "LOCK_TEXT = r'''" in s)
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    build.check_lock(build._pins(ROOT, TEMPLATE), (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8"))
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "needs re-qualification with `restarted: false`" in record


# ---- M2: a real BYOD dataset path ------------------------------------------------------------------------------------


def test_M2_read_detection_records_reads_a_labelled_directory_and_refuses_by_name(tmp_path: Path) -> None:
    records = pipeline_module.read_detection_records(_byod_dir(tmp_path, 3))
    assert [r["id"] for r in records] == ["img00.png", "img01.png", "img02.png"]
    assert pipeline_module.validate_dataset(records, sorted({label for r in records for label in r["labels"]}))["verdict"] == "accepted"
    other = tmp_path / "missing"
    other.mkdir()
    with pytest.raises(FileNotFoundError, match=r"image 'img01\.png' not found"):
        pipeline_module.read_detection_records(_byod_dir(other, 3, missing="img01.png"))
    third = tmp_path / "escape"
    third.mkdir()
    with pytest.raises(ValueError, match=r"must be relative to"):
        pipeline_module.read_detection_records(_byod_dir(third, 3, escape=True))


def test_M2_byod_dataset_branch_adapts_reloads_and_writes_a_result_with_no_code_edits(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    directory = _byod_dir(tmp_path, 10)
    source = _byod_cell(notebook, **{"USE_BYOD_DATASET = False": "USE_BYOD_DATASET = True", 'BYOD_DATASET_DIR = ""': f"BYOD_DATASET_DIR = {str(directory)!r}"})
    exec(source, ns)  # noqa: S102 - the notebook's own BYOD cell
    result = json.loads((tmp_path / "byod_yolox_result.json").read_text(encoding="utf-8"))
    assert result["reload_check"]["identical"] is True
    assert result["baseline"]["ap"] == 0.2 and result["adapted"]["ap"] == 0.9
    assert result["split"]["held_out"] >= 2 and result["class_names"] == sorted(result["class_names"])


def test_M2_byod_dataset_below_the_minimum_or_incompatible_is_refused(notebook: dict, tmp_path: Path, monkeypatch, capsys) -> None:
    """R116-m7 (DAT20): a refused BYOD dataset prints one message naming the rule, writes nothing and raises nothing."""
    ns = _namespace(tmp_path, monkeypatch)
    small = _byod_dir(tmp_path, 4)
    exec(_byod_cell(notebook, **{"USE_BYOD_DATASET = False": "USE_BYOD_DATASET = True", 'BYOD_DATASET_DIR = ""': f"BYOD_DATASET_DIR = {str(small)!r}"}), ns)  # noqa: S102
    assert re.search(r"BYOD dataset refused: .*at least 8 are needed.*earlier sections are kept", capsys.readouterr().out)
    other = tmp_path / "bad"
    other.mkdir()
    bad = _byod_dir(other, 9)
    exec(_byod_cell(notebook, **{"USE_BYOD_DATASET = False": "USE_BYOD_DATASET = True", 'BYOD_DATASET_DIR = ""': f"BYOD_DATASET_DIR = {str(bad)!r}", 'BYOD_CLASS_NAMES = ""': 'BYOD_CLASS_NAMES = "only-this"'}), ns)  # noqa: S102
    assert re.search(r"BYOD dataset refused: .*label .* is not in class_names", capsys.readouterr().out)
    exec(_byod_cell(notebook, **{"USE_BYOD_DATASET = False": "USE_BYOD_DATASET = True", 'BYOD_DATASET_DIR = ""': "BYOD_DATASET_DIR = 'no-such-dir'"}), ns)  # noqa: S102
    assert "is not a directory" in capsys.readouterr().out
    assert not (tmp_path / "byod_yolox_result.json").exists()


# ---- M3: the guided layer --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    ["## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model/System → Output", "## Roadmap", "<strong>Glossary</strong>", "**Predict before running:**", "**What to notice", "<summary>Check your reasoning</summary>", "## 14. Activity: change one thing — unfreeze the backbone", "## Troubleshooting", "## Conclusion (your notes)"],
)
def test_M3_guided_layer_marker_is_present(notebook: dict, marker: str) -> None:
    assert marker in _markdown(notebook)


def test_M3_infrastructure_cells_are_collapsed_and_the_miss_is_not_announced(notebook: dict) -> None:
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    first_learner = next(i for i, c in enumerate(code) if c["source"].startswith("import hashlib"))
    assert first_learner >= 10
    for cell in code[:first_learner]:
        assert cell["metadata"].get("cellView") == "form", cell["source"][:60]
    assert "**Expect one honest failure.**" not in _markdown(notebook)
    assert _markdown(notebook).count("**Predict before running:**") >= 6


# ---- m1: no template braces, a documented upstream TODO only ---------------------------------------------------------


def test_m1_no_template_braces_and_only_the_documented_upstream_todo(notebook: dict) -> None:
    assert "{{" not in _markdown(notebook)
    hits = [(i, line) for i, s in enumerate(_code(notebook)) for line in s.splitlines() if re.search(r"\b(TODO|TBD|FIXME)\b", line)]
    assert [line.strip() for _i, line in hits] == ["# TODO: the string might change, consider a better way"]
    assert "the string might change, consider a better way" in (ROOT / "docs" / "UPSTREAM.md").read_text(encoding="utf-8")


# ---- m2: re-running the fine-tune cell trains fresh models: tests/test_score_once.py ---------------------------


# ---- m3: BYOD image by path, named decode errors, exactly one upload -------------------------------------------------


def test_m3_byod_image_by_path_runs_without_colab_and_corrupt_files_are_named(notebook: dict, tmp_path: Path, monkeypatch, capsys) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    Image.new("RGB", (64, 48), "white").save(tmp_path / "mine.png")
    exec(_byod_cell(notebook, **{"USE_BYOD_IMAGE = False": "USE_BYOD_IMAGE = True", 'BYOD_IMAGE_PATH = ""': "BYOD_IMAGE_PATH = 'mine.png'"}), ns)  # noqa: S102
    assert ns["byod_image"].size == (64, 48) and "not-measurable" in capsys.readouterr().out
    with pytest.raises(ValueError, match=r"^broken\.png: not an image Pillow can decode"):
        pipeline_module.load_byod_image("broken.png", b"not an image")


@pytest.mark.parametrize("uploads", [{}, {"a.png": b"x", "b.png": b"y"}])
def test_m3_cancelled_or_multi_file_upload_stops_with_the_rule(notebook: dict, tmp_path: Path, monkeypatch, uploads: dict, capsys) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: uploads
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)
    exec(_byod_cell(notebook, **{"USE_BYOD_IMAGE = False": "USE_BYOD_IMAGE = True"}), ns)  # noqa: S102
    assert f"BYOD image refused: Upload exactly one image (received {len(uploads)})" in capsys.readouterr().out


# ---- m3 (status half): STATUS.md, README and the record agree on the restarted hosted run ---------------------------


def test_m3_status_and_record_agree_the_restarted_hosted_run_is_not_one_pass_evidence() -> None:
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8").split("## Release status", 1)[1]
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status
    for text in (status, readme):
        assert "Kaggle T4" in text and "manual restart" in text and "not one-pass" in text
    gate6 = next(line for line in record.splitlines() if line.startswith("| 6 |"))
    assert "| **open**" in gate6 and "**met**" not in gate6
    assert "PASS after a manual restart" in record
