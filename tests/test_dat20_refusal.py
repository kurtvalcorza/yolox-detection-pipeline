"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and stops Run all.

Two layers: the Section 13 cell raises `BYODRefusal` (checked by executing the committed cell with the
real loaders), and the isolated-cell router relays it as `IsolatedCellRefusal` with no traceback (checked
by running the generator's router against a real worker process; POSIX only, as on Colab).
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
from multiprocessing.connection import Connection

import pytest

from test_review_fixes import NOTEBOOK, ROOT, _byod_cell, _byod_dir, _namespace

REFUSAL_CELL = "yolox_detection-54"
IMAGE_ON = {"USE_BYOD_IMAGE = False": "USE_BYOD_IMAGE = True"}
DATASET_ON = {"USE_BYOD_DATASET = False": "USE_BYOD_DATASET = True"}


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _refused(notebook, ns, **changes) -> str:
    with pytest.raises(ValueError) as caught:
        exec(_byod_cell(notebook, **changes), ns)  # noqa: S102 - the notebook's own BYOD cell
    assert caught.value.dimer_refusal is True and caught.value.__cause__ is None
    rendered = caught.value._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    return rendered[0]


def test_scripted_hosted_input_a_file_that_is_not_an_image(notebook, tmp_path, monkeypatch):
    ns = _namespace(tmp_path, monkeypatch)
    (tmp_path / "hostname").write_text("colab-vm\n", encoding="utf-8")
    line = _refused(
        notebook,
        ns,
        **IMAGE_ON,
        **{'BYOD_IMAGE_PATH = ""': f"BYOD_IMAGE_PATH = {str(tmp_path / 'hostname')!r}"},
    )
    assert line.startswith("Refused: hostname: not an image Pillow can decode"), line


def test_a_directory_as_the_image_path_and_a_missing_dataset_dir(notebook, tmp_path, monkeypatch):
    ns = _namespace(tmp_path, monkeypatch)
    line = _refused(
        notebook, ns, **IMAGE_ON, **{'BYOD_IMAGE_PATH = ""': f"BYOD_IMAGE_PATH = {str(tmp_path)!r}"}
    )
    assert line.startswith(f"Refused: {tmp_path}: BYOD_IMAGE_PATH is not a file"), line
    line = _refused(
        notebook, ns, **DATASET_ON, **{'BYOD_DATASET_DIR = ""': "BYOD_DATASET_DIR = 'no-such-dir'"}
    )
    assert line.startswith("Refused: no-such-dir: BYOD_DATASET_DIR is not a directory"), line


def test_a_dataset_refusal_removes_an_earlier_byod_result(notebook, tmp_path, monkeypatch):
    ns = _namespace(tmp_path, monkeypatch)
    (tmp_path / "byod_yolox_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "byod-adapter.safetensors").write_bytes(b"earlier")
    ns["byod_adapted"] = {"ap": 0.9}
    small = _byod_dir(tmp_path, 4)
    line = _refused(
        notebook, ns, **DATASET_ON, **{'BYOD_DATASET_DIR = ""': f"BYOD_DATASET_DIR = {str(small)!r}"}
    )
    assert line.startswith("Refused: byod: the BYOD dataset has 4 records"), line
    assert (
        not (tmp_path / "byod_yolox_result.json").exists()
        and not (tmp_path / "byod-adapter.safetensors").exists()
    )
    assert "byod_adapted" not in ns


def test_genuine_error_keeps_its_type(notebook, tmp_path, monkeypatch):
    ns = _namespace(tmp_path, monkeypatch)
    (tmp_path / "mine.png").write_bytes(b"x")

    def broken(name, data):
        raise RuntimeError("internal bug")

    ns["load_byod_image"] = broken
    with pytest.raises(RuntimeError, match="internal bug"):
        exec(_byod_cell(notebook, **IMAGE_ON, **{'BYOD_IMAGE_PATH = ""': "BYOD_IMAGE_PATH = 'mine.png'"}), ns)  # noqa: S102


def test_refusal_cell_id(notebook):
    ids = [c["id"] for c in notebook["cells"] if "class BYODRefusal" in "".join(c["source"])]
    assert ids == [REFUSAL_CELL]


def _router() -> dict:
    tree = ast.parse((ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_ISOLATED_ROUTER"
    )
    parts = [
        c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)
    ]
    source = next(s for s in parts if "class IsolatedRuntime" in s)
    source = source[source.index("_WORKER_SOURCE = ") :]
    ns = {"os": os, "signal": signal, "subprocess": subprocess, "sys": sys, "Connection": Connection}
    exec(source[: source.index("\ndef _is_user_cell")], ns)  # noqa: S102 - the router's worker and kernel classes
    return ns


@pytest.mark.skipif(
    os.name == "nt", reason="the router passes file descriptors to the worker (POSIX, as on Colab)"
)
def test_router_relays_a_refusal_as_one_line_and_keeps_tracebacks_for_errors(capfd):
    router = _router()
    runtime = router["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        cell = (
            "class BYODRefusal(ValueError):\n    dimer_refusal = True\n\n"
            "raise BYODRefusal('hostname: not an image')\n"
        )
        with pytest.raises(router["IsolatedCellRefusal"]) as caught:
            runtime.run(cell)
        assert caught.value._render_traceback_() == ["Refused: hostname: not an image"]
        assert "Traceback" not in capfd.readouterr().err
        runtime.run("survivor = 1")  # the worker namespace survives the refusal
        with pytest.raises(router["IsolatedCellError"]):
            runtime.run("raise KeyError('a bug')")
        assert "Traceback" in capfd.readouterr().err
    finally:
        runtime.close()
