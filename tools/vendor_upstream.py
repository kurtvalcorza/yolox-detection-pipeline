#!/usr/bin/env python3
"""Vendor the YOLOX model code this package runs, from a pinned upstream revision.

YOLOX is not distributed as a usable library for this purpose. The PyPI ``yolox`` sdist is the
official Megvii source, but its ``install_requires`` pins ``onnx==1.8.1``, ``onnxruntime==1.8.0``
and ``onnx-simplifier==0.3.5`` and asks for a bare ``torch>=1.7``/``torchvision``, so installing it
would churn or break a pinned runtime; and ``--no-deps`` does not help, because
``yolox.models.yolo_head`` imports ``yolox.utils``, whose ``__init__`` pulls in ``cv2``, ``loguru``
and ``thop``. The inference and SimOTA-training path itself needs nothing but torch, torchvision and
numpy, so it is vendored here instead.

The eight modules are copied from the pinned revision with their upstream file names and their
upstream package-relative imports intact. Exactly three edits are applied, each declared in EDITS
and each required to match exactly once, so a silent divergence is impossible. ``docs/UPSTREAM.md``
records the revision, the per-file upstream SHA-256 and the diff of every edit.

Usage:
    python tools/vendor_upstream.py            # (re)write the vendored modules and docs/UPSTREAM.md
    python tools/vendor_upstream.py --check     # exit 1 if the working tree differs from upstream+edits
    python tools/vendor_upstream.py --from DIR  # read upstream from a local checkout instead of GitHub
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "https://github.com/Megvii-BaseDetection/YOLOX"
UPSTREAM_REV = "419778480ab6ec0590e5d3831b3afb3b46ab2aa3"
UPSTREAM_TAG = "0.3.0"
RAW = "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/{rev}/{path}"

PACKAGE = "yolox_detection_pipeline"

# Upstream path -> vendored file name. Upstream names are kept so the vendored tree reads as what it
# is, and so the upstream `from .losses import ...` / `from .network_blocks import ...` lines keep
# working unchanged in the flat package.
VERBATIM: dict[str, str] = {
    "yolox/models/network_blocks.py": "network_blocks.py",
    "yolox/models/darknet.py": "darknet.py",
    "yolox/models/yolo_pafpn.py": "yolo_pafpn.py",
    "yolox/models/losses.py": "losses.py",
    "yolox/models/yolo_head.py": "yolo_head.py",
    "yolox/models/yolox.py": "yolox.py",
    "yolox/data/datasets/coco_classes.py": "coco_classes.py",
}

# Concatenated, in this order, into the derived ops.py.
DERIVED_OPS: tuple[str, ...] = ("yolox/utils/compat.py", "yolox/utils/boxes.py")

VENDORED = (*VERBATIM.values(), "ops.py")

# (vendored file, old, new, why) -- applied in order; each must match exactly once.
EDITS: tuple[tuple[str, str, str, str], ...] = (
    (
        "yolo_head.py",
        "import math\nfrom loguru import logger\n",
        "import math\nimport warnings\n",
        "drop the loguru dependency; its only use is the one message on the CUDA-OOM fallback path "
        "rewritten below",
    ),
    (
        "yolo_head.py",
        "from yolox.utils import bboxes_iou, meshgrid\n",
        "from .ops import bboxes_iou, meshgrid\n",
        "both helpers are vendored into ops.py; importing yolox.utils would pull in cv2, loguru, "
        "thop and tabulate through its __init__",
    ),
    (
        "yolo_head.py",
        "                    logger.error(\n",
        "                    warnings.warn(\n",
        "same message on the same code path, through the standard library instead of loguru",
    ),
)

OPS_HEADER = '''"""Box helpers and the NMS post-processor, vendored from upstream ``yolox/utils``.

Concatenated from ``yolox/utils/compat.py`` and ``yolox/utils/boxes.py`` at the revision recorded in
``docs/UPSTREAM.md``, minus their shebang lines. The upstream ``yolox/utils/__init__.py`` re-exports
modules that import ``cv2``, ``loguru``, ``thop`` and ``tabulate``; only the helpers below are on the
inference or training path, so vendoring them keeps this package's dependency set at torch,
torchvision, numpy and Pillow.
"""

'''

UPSTREAM_MD_TAIL = """
`ops.py` is the concatenation of the two `yolox/utils` files above, minus their shebang lines, under
a new module docstring. Every other file is byte-identical to upstream apart from the edits below,
including its upstream package-relative imports (`from .losses import IOUloss`,
`from .network_blocks import BaseConv, DWConv`, ...), which resolve unchanged because the modules keep
their upstream names in this flat package.

## Edits

"""

NOT_VENDORED = """
## Deliberately not vendored

- `yolox/models/build.py` — `create_yolox_model` fetches a checkpoint with
  `torch.hub.load_state_dict_from_url` and no digest check. This package stages the checkpoint from
  the pinned release asset and verifies its SHA-256 against the committed manifest first.
- `yolox/exp/` — the experiment/configuration machinery. The two values this profile needs from it
  (`depth = 0.33`, `width = 0.50` for YOLOX-S) are module constants in `pipeline.py`, and the
  `test_size`/`test_conf`/`nmsthre` defaults from `yolox/exp/yolox_base.py` are recorded there too.
- `yolox/data/` (beyond the COCO class list), `yolox/evaluators/`, `yolox/core/` and the rest of
  `yolox/utils/` — the training harness, COCO evaluator, visualiser and distributed helpers, which
  bring in `cv2`, `loguru`, `thop`, `tabulate`, `pycocotools` and `tensorboard`. The bounded
  fine-tuning loop and the COCO-style evaluator this package needs are implemented in `pipeline.py`
  against the vendored head instead.
- The upstream `ValTransform`/`preproc` letterbox, which resizes with `cv2.INTER_LINEAR`.
  `pipeline.py` reproduces it with Pillow; see `docs/WEIGHTS.md` for the one case where the two can
  differ and why the default 640x640 sample path is unaffected.
"""


def fetch(path: str, local: Path | None) -> bytes:
    if local is not None:
        return (local / path).read_bytes()
    with urllib.request.urlopen(RAW.format(rev=UPSTREAM_REV, path=path), timeout=60) as response:
        return response.read()


def build(local: Path | None) -> tuple[dict[str, bytes], list[tuple[str, str, str, int]]]:
    """Return {vendored name: bytes} and the provenance rows for docs/UPSTREAM.md."""
    rows: list[tuple[str, str, str, int]] = []
    texts: dict[str, str] = {}

    for upstream_path, name in VERBATIM.items():
        raw = fetch(upstream_path, local)
        if b"\x00" in raw:
            raise SystemExit(f"NUL byte in {upstream_path}")
        rows.append((upstream_path, name, hashlib.sha256(raw).hexdigest(), len(raw)))
        texts[name] = raw.decode("utf-8")

    parts = [OPS_HEADER]
    for upstream_path in DERIVED_OPS:
        raw = fetch(upstream_path, local)
        rows.append((upstream_path, "ops.py", hashlib.sha256(raw).hexdigest(), len(raw)))
        body = raw.decode("utf-8")
        parts.append("".join(line for line in body.splitlines(keepends=True) if not line.startswith("#!")))
    texts["ops.py"] = "".join(parts)

    for name, old, new, _why in EDITS:
        count = texts[name].count(old)
        if count != 1:
            raise SystemExit(
                f"edit for {name} matched {count} times, expected exactly 1 — upstream changed at "
                f"{UPSTREAM_REV}? offending text:\n{old!r}"
            )
        texts[name] = texts[name].replace(old, new)

    out = {}
    for name, text in texts.items():
        data = text.encode("utf-8")
        if b"\x00" in data:
            raise SystemExit(f"NUL byte in generated {name}")
        out[name] = data
    return out, rows


def upstream_md(rows: list[tuple[str, str, str, int]]) -> bytes:
    lines = [
        "# Vendored upstream YOLOX code",
        "",
        f"- **Source:** {UPSTREAM_REPO}",
        f"- **Revision:** `{UPSTREAM_REV}` (tag `{UPSTREAM_TAG}`)",
        f"- **Licence:** Apache-2.0 — `src/{PACKAGE}/LICENSE-YOLOX` is upstream's own LICENSE file",
        "- **Regenerate / verify:** `python tools/vendor_upstream.py [--check]`",
        "",
        "The eight modules below are upstream YOLOX, not DIMER code. They are kept close to verbatim so",
        "a reviewer can re-fetch the revision and diff. `tools/vendor_upstream.py --check` fails if the",
        "working tree stops matching upstream-plus-declared-edits, and runs in CI.",
        "",
        "Why the code is vendored rather than installed: the PyPI `yolox` sdist is the official Megvii",
        "source, but its `install_requires` pins `onnx==1.8.1`, `onnxruntime==1.8.0` and",
        "`onnx-simplifier==0.3.5` alongside a bare `torch>=1.7`/`torchvision`, so installing it would",
        "churn or break a pinned runtime. `pip install --no-deps yolox` does not avoid this either:",
        "`yolox.models.yolo_head` imports `yolox.utils`, whose `__init__` re-exports modules needing",
        "`cv2`, `loguru` and `thop`. The path this package actually runs needs only torch, torchvision",
        "and numpy.",
        "",
        "## Files",
        "",
        "| Upstream path | Vendored as | Upstream SHA-256 | Bytes |",
        "|---|---|---|---|",
    ]
    for upstream_path, name, digest, size in rows:
        lines.append(f"| `{upstream_path}` | `src/{PACKAGE}/{name}` | `{digest}` | {size} |")
    lines.append(UPSTREAM_MD_TAIL.rstrip("\n"))
    lines.append("")
    for index, (name, old, new, why) in enumerate(EDITS, start=1):
        lines += [
            f"{index}. **`{name}`** — {why}.",
            "",
            "   ```diff",
            *[f"   -{line}" for line in old.rstrip("\n").splitlines()],
            *[f"   +{line}" for line in new.rstrip("\n").splitlines()],
            "   ```",
            "",
        ]
    lines.append(NOT_VENDORED.strip("\n"))
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="exit 1 on any difference; write nothing")
    parser.add_argument("--from", dest="local", type=Path, default=None, help="local upstream checkout root")
    args = parser.parse_args(argv)

    pkg_dir = args.repo / "src" / PACKAGE
    docs = args.repo / "docs" / "UPSTREAM.md"
    files, rows = build(args.local)
    files_out = {pkg_dir / name: data for name, data in files.items()}
    files_out[docs] = upstream_md(rows)

    if args.check:
        stale = []
        for path, data in files_out.items():
            # Compare on LF: a Windows checkout with core.autocrlf rewrites text files to CRLF.
            current = path.read_bytes().replace(b"\r\n", b"\n") if path.is_file() else b""
            if current != data:
                stale.append(path.relative_to(args.repo).as_posix())
        if stale:
            print(
                "STALE vendored upstream files (run tools/vendor_upstream.py): " + ", ".join(sorted(stale)),
                file=sys.stderr,
            )
            return 1
        print(
            f"OK: {len(files_out)} vendored files match upstream {UPSTREAM_REV[:12]} plus {len(EDITS)} edits"
        )
        return 0

    for path, data in files_out.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f"wrote {len(files_out)} files from upstream {UPSTREAM_REV[:12]} plus {len(EDITS)} declared edits")
    for upstream_path, name, digest, size in rows:
        print(f"  {upstream_path:45s} -> {name:20s} {digest[:16]} {size:>7d} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
