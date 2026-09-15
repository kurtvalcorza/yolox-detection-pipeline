"""The vendored-upstream boundary.

`src/yolox_detection_pipeline/` holds eight modules of upstream YOLOX beside this package's own code.
These tests keep that boundary legible: the vendored files stay byte-identical to the pinned upstream
revision apart from the three declared edits, they import nothing the pinned runtime does not carry,
and the package's identity constants keep naming the revision they came from.

The network-backed check (`tools/vendor_upstream.py --check`, which re-fetches upstream and diffs) is
a CI step rather than a unit test so the suite stays offline.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from yolox_detection_pipeline.pipeline import UPSTREAM_CODE_REVISION

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "yolox_detection_pipeline"
DOCS = ROOT / "docs" / "UPSTREAM.md"

VENDORED = (
    "network_blocks.py",
    "darknet.py",
    "yolo_pafpn.py",
    "losses.py",
    "yolo_head.py",
    "yolox.py",
    "coco_classes.py",
    "ops.py",
)
OWN = ("pipeline.py", "samples.py", "__init__.py")


def _vendor_tool():
    spec = importlib.util.spec_from_file_location("vendor_upstream", ROOT / "tools" / "vendor_upstream.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vendor = _vendor_tool()


def test_every_vendored_module_exists() -> None:
    for name in VENDORED:
        assert (PKG / name).is_file(), name


def test_the_vendor_tool_and_the_tests_agree_on_the_file_list() -> None:
    assert set(vendor.VENDORED) == set(VENDORED)


def test_the_package_pins_the_revision_the_tool_vendored_from() -> None:
    assert UPSTREAM_CODE_REVISION == vendor.UPSTREAM_REV
    assert len(UPSTREAM_CODE_REVISION) == 40


def test_upstream_doc_records_the_revision_and_the_licence_file() -> None:
    text = DOCS.read_text(encoding="utf-8")
    assert UPSTREAM_CODE_REVISION in text
    assert "Apache-2.0" in text
    assert (PKG / "LICENSE-YOLOX").is_file()
    assert "Apache License" in (PKG / "LICENSE-YOLOX").read_text(encoding="utf-8")


def test_upstream_doc_records_a_digest_for_every_vendored_file() -> None:
    text = DOCS.read_text(encoding="utf-8")
    for name in VENDORED:
        assert f"`src/yolox_detection_pipeline/{name}`" in text, name


def test_every_declared_edit_is_applied_exactly_once() -> None:
    for name, old, new, _why in vendor.EDITS:
        source = (PKG / name).read_text(encoding="utf-8")
        assert old not in source, f"{name}: pre-edit text is still present"
        assert source.count(new) == 1, f"{name}: post-edit text should appear exactly once"


def test_vendored_code_imports_nothing_outside_the_pinned_runtime() -> None:
    """The point of vendoring: no cv2, no loguru, no thop, no `yolox` distribution."""
    allowed = {
        "torch",
        "torchvision",
        "numpy",
        "math",
        "warnings",
        "random",
        "copy",
        "functools",
        "os",
        "sys",
    }
    for name in VENDORED:
        tree = ast.parse((PKG / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.partition(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                roots = [(node.module or "").partition(".")[0]]
            for root in roots:
                assert root in allowed, f"{name} imports {root!r}, which is not in the pinned runtime"


def test_vendored_code_never_imports_the_yolox_distribution() -> None:
    for name in VENDORED:
        source = (PKG / name).read_text(encoding="utf-8")
        assert "from yolox" not in source, name
        assert "import yolox" not in source, name


def test_vendored_relative_imports_resolve_inside_this_package() -> None:
    stems = {Path(name).stem for name in (*VENDORED, *OWN)}
    for name in VENDORED:
        tree = ast.parse((PKG / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1, f"{name}: only single-level relative imports are embeddable"
                assert node.module in stems, f"{name}: `from .{node.module}` names no module here"


def test_package_code_does_not_reach_into_the_upstream_download_helper() -> None:
    """`yolox/models/build.py` downloads a checkpoint with no digest check; it is deliberately not
    vendored, and nothing here should reimplement it."""
    assert not (PKG / "build.py").exists()
    for name in OWN:
        source = (PKG / name).read_text(encoding="utf-8")
        assert "load_state_dict_from_url" not in source, name
        assert "torch.hub" not in source, name


def test_ruff_excludes_exactly_the_vendored_files() -> None:
    """A lint autofix on vendored code would fork it from upstream and break the vendoring check."""
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name in VENDORED:
        assert f'"src/yolox_detection_pipeline/{name}"' in config, name
    for name in OWN:
        assert f'"src/yolox_detection_pipeline/{name}"' not in config, f"{name} must stay linted"


@pytest.mark.parametrize("name", VENDORED)
def test_vendored_modules_are_importable(name: str) -> None:
    module = importlib.import_module(f"yolox_detection_pipeline.{Path(name).stem}")
    assert module.__file__ is not None
