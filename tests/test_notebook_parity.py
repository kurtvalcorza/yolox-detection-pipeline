"""NOTEBOOK_SPEC 2.0 parity tests (PAR1–PAR4) for the standalone tutorial notebook.

The notebook carries the whole package — eight vendored upstream YOLOX modules plus this
repository's `samples.py` and `pipeline.py` — one tagged cell per module in dependency order. These
tests fail whenever a carried cell, the inline manifest or the inline pins diverge from the
repository at HEAD.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
PKG_DIR = ROOT / "src" / TEMPLATE["package"]
MANIFEST = ROOT / "weights" / TEMPLATE["weights_key"] / "dimer-base-manifest.json"
REWRITES = TEMPLATE.get("rewrites", build.DEFAULT_REWRITES)


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} not generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def context() -> dict:
    return build.load_context(ROOT, TEMPLATE)


def _cells(notebook: dict, cell_type: str) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == cell_type]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _tagged(notebook: dict) -> list[dict]:
    return [
        c for c in _cells(notebook, "code") if c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    ]


def test_par1_every_module_is_carried_in_dependency_order(notebook: dict, context: dict) -> None:
    tagged = _tagged(notebook)
    names = [c["metadata"]["dimer"]["embedded_module"] for c in tagged]
    assert names == context["module_rels"], "carried modules differ from the generator's dependency order"
    assert len(names) == len(TEMPLATE["modules"]) == 10


def test_par1_each_carried_cell_equals_its_repository_module(notebook: dict, context: dict) -> None:
    texts = {m: (PKG_DIR / m).read_text(encoding="utf-8") for m in context["modules"]}
    expected = build.apply_rewrites(texts, REWRITES)
    for cell, module in zip(_tagged(notebook), context["modules"], strict=True):
        assert _source(cell).rstrip("\n") + "\n" == expected[module], (
            f"embedded {module} drifted from src/; regenerate the notebook"
        )


def test_par1_each_carried_cell_records_its_own_module_digest(notebook: dict, context: dict) -> None:
    for cell in _tagged(notebook):
        meta = cell["metadata"]["dimer"]
        assert meta["module_sha256"] == context["per_module_sha256"][meta["embedded_module"]]


def test_par1_rewrite_rules_are_the_only_difference(context: dict) -> None:
    """Exactly one line changes across the whole package, and it is the weights-directory rewrite."""
    texts = {m: (PKG_DIR / m).read_text(encoding="utf-8") for m in context["modules"]}
    rewritten = build.apply_rewrites(dict(texts), REWRITES)
    changed: list[tuple[str, str]] = []
    for module, original in texts.items():
        for a, b in zip(original.splitlines(), rewritten[module].splitlines(), strict=False):
            if a != b:
                changed.append((a, b))
    # The declared rewrites, plus the package-relative imports the generator strips.
    rewrite_lines = [(a, b) for a, b in changed if "standalone rewrite" in b and "__file__" in a]
    import_lines = [(a, b) for a, b in changed if a.lstrip().startswith("from .")]
    assert len(rewrite_lines) == len(REWRITES)
    assert changed and len(rewrite_lines) + len(import_lines) == len(changed), changed
    for _original, replaced in import_lines:
        assert "standalone rewrite" in replaced


def test_par1_no_package_relative_import_survives(notebook: dict) -> None:
    """In the notebook every module's names are already kernel globals, so a surviving `from .x`
    would be a NameError at run time."""
    for cell in _tagged(notebook):
        for line in _source(cell).splitlines():
            assert not line.lstrip().startswith("from ."), line


def test_par2_inline_manifest_matches_the_repository(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    assert inline, "model cell must carry MANIFEST = {...}"
    assert json.loads(inline.group(1)) == json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_par2_inline_pins_match_pyproject(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    assert pins_block, "install cell must carry PINS = [...]"
    assert re.findall(r"'([^']+)'", pins_block.group(1)) == build._pins(ROOT)


def test_par2_metadata_declares_the_standalone_contract(notebook: dict, context: dict) -> None:
    meta = notebook["metadata"]["dimer"]
    assert meta["standalone"] is True
    assert meta["notebook_spec"] == build.NOTEBOOK_SPEC
    assert meta["notebook_profile"] == TEMPLATE["profile"] == "E2E"
    generated = meta["generated_from"]
    assert generated["module"] == f"src/{TEMPLATE['package']}/pipeline.py"
    assert generated["modules"] == context["module_rels"]
    assert generated["module_sha256"] == context["module_sha256"]


def test_par3_generator_check_is_clean(notebook: dict) -> None:
    # The recorded revision is a provenance label carried through the check (see build_notebook.py
    # --check); content drift is what fails this comparison.
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, TEMPLATE, recorded))
    current = NOTEBOOK.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    assert current == rendered, "notebook is stale; run python tools/build_notebook.py"


def test_st1_primary_path_has_no_repository_dependency(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    # The *act* of cloning, not the word: the carried `stage_missing_files` docstring legitimately
    # explains what happens on a fresh clone of the repository.
    assert not re.search(r"\bgit\b[^\n]*\bclone\b", code)
    assert "raw.githubusercontent.com" not in code
    assert "pip install -e" not in code
    assert f"import {TEMPLATE['package']}" not in code
    assert f"from {TEMPLATE['package']}" not in code


def test_st1_the_only_github_reference_is_the_pinned_release_asset(notebook: dict) -> None:
    """This profile's weights *are* a GitHub release asset, so `github.com` cannot simply be banned
    in code. What must not appear is a path that fetches repository source."""
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    urls = re.findall(r"https://github\.com/[^\s'\"]+", code)
    assert urls, "the release asset URL must be carried"
    for url in urls:
        assert "/releases/download/" in url, url
        # Either the resolved URL or the f-string that builds it from the pinned constants.
        assert "Megvii-BaseDetection/YOLOX" in url or "{MODEL_ID}" in url, url
    assert any("Megvii-BaseDetection/YOLOX" in url for url in urls), urls


def test_e2e_default_path_actually_adapts(notebook: dict) -> None:
    """RUN7/FT2: an E2E notebook whose workflow includes adaptation must really run it in the
    default path, not behind a default-off flag."""
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    assert "run = adapter.finetune(" in code
    finetune_line = next(line for line in code.splitlines() if "adapter.finetune(" in line)
    assert not finetune_line.lstrip().startswith(("if ", "#")), finetune_line
    for gate in ("USE_BYOD_IMAGE = False", "USE_BYOD_DATASET = False"):
        assert gate in code, f"{gate} must be gated off by default"


def test_e2e_baseline_precedes_adaptation(notebook: dict) -> None:
    """A before/after claim needs the before to be measured first, on the same held-out records."""
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    assert code.index("baseline = adapter.evaluate(held_out)") < code.index("run = adapter.finetune(")
    assert code.index("run = adapter.finetune(") < code.index("adapted = adapter.evaluate(held_out)")


def test_e2e_artifact_is_exported_and_reloaded(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    assert "adapter.save_artifact(" in code
    assert "YoloxDetectionPipeline.load_artifact(" in code
    assert "assert abs(reloaded_metrics['ap'] - adapted['ap']) < 1e-9" in code
