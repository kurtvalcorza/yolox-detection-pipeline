# Vendored upstream YOLOX code

- **Source:** https://github.com/Megvii-BaseDetection/YOLOX
- **Revision:** `419778480ab6ec0590e5d3831b3afb3b46ab2aa3` (tag `0.3.0`)
- **Licence:** Apache-2.0 — `src/yolox_detection_pipeline/LICENSE-YOLOX` is upstream's own LICENSE file
- **Regenerate / verify:** `python tools/vendor_upstream.py [--check]`

The eight modules below are upstream YOLOX, not DIMER code. They are kept close to verbatim so
a reviewer can re-fetch the revision and diff. `tools/vendor_upstream.py --check` fails if the
working tree stops matching upstream-plus-declared-edits, and runs in CI.

Why the code is vendored rather than installed: the PyPI `yolox` sdist is the official Megvii
source, but its `install_requires` pins `onnx==1.8.1`, `onnxruntime==1.8.0` and
`onnx-simplifier==0.3.5` alongside a bare `torch>=1.7`/`torchvision`, so installing it would
churn or break a pinned runtime. `pip install --no-deps yolox` does not avoid this either:
`yolox.models.yolo_head` imports `yolox.utils`, whose `__init__` re-exports modules needing
`cv2`, `loguru` and `thop`. The path this package actually runs needs only torch, torchvision
and numpy.

## Files

| Upstream path | Vendored as | Upstream SHA-256 | Bytes |
|---|---|---|---|
| `yolox/models/network_blocks.py` | `src/yolox_detection_pipeline/network_blocks.py` | `f490240a674799f1b548ac4ca571c7f3e59d4e75efee8ebeb1e24e6e4e10d2a4` | 6092 |
| `yolox/models/darknet.py` | `src/yolox_detection_pipeline/darknet.py` | `368c9db45166e863691cf7460b314d047928f90e8744c492a673b6eff9b809f3` | 6019 |
| `yolox/models/yolo_pafpn.py` | `src/yolox_detection_pipeline/yolo_pafpn.py` | `8571af557f6caacda61e54813be4be2d73c4a0889e1f876ec5586057eb2eebde` | 3530 |
| `yolox/models/losses.py` | `src/yolox_detection_pipeline/losses.py` | `7fa685fb4653bab69d5880c3f749edeb8a4cabfe71fadf0acc5151c2eaa75702` | 1677 |
| `yolox/models/yolo_head.py` | `src/yolox_detection_pipeline/yolo_head.py` | `9fcdcf859b91c587d11926c0a8e3e423abdaa53504eb8ad2a17379ee405378bd` | 23339 |
| `yolox/models/yolox.py` | `src/yolox_detection_pipeline/yolox.py` | `3b88f3b19e27232ab5442b4d2c53b20d3f1e40f6f093871d6742e694368f6da0` | 1364 |
| `yolox/data/datasets/coco_classes.py` | `src/yolox_detection_pipeline/coco_classes.py` | `b38193c481a73f1f674cedab9e551b15b39b1a7aaed3e09e16505362cc54ad51` | 1296 |
| `yolox/utils/compat.py` | `src/yolox_detection_pipeline/ops.py` | `a8f0dec9e0566cf0a09a211bc08ce47bc7abd6dd75d239ee85d8c0c5fe3e07ce` | 310 |
| `yolox/utils/boxes.py` | `src/yolox_detection_pipeline/ops.py` | `65d8341d67ec35d65f4a73ba8480ce536a8bb38f4b4088f99db82bc812ec9bcb` | 4471 |

`ops.py` is the concatenation of the two `yolox/utils` files above, minus their shebang lines, under
a new module docstring. Every other file is byte-identical to upstream apart from the edits below,
including its upstream package-relative imports (`from .losses import IOUloss`,
`from .network_blocks import BaseConv, DWConv`, ...), which resolve unchanged because the modules keep
their upstream names in this flat package.

## Edits

1. **`yolo_head.py`** — drop the loguru dependency; its only use is the one message on the CUDA-OOM fallback path rewritten below.

   ```diff
   -import math
   -from loguru import logger
   +import math
   +import warnings
   ```

2. **`yolo_head.py`** — both helpers are vendored into ops.py; importing yolox.utils would pull in cv2, loguru, thop and tabulate through its __init__.

   ```diff
   -from yolox.utils import bboxes_iou, meshgrid
   +from .ops import bboxes_iou, meshgrid
   ```

3. **`yolo_head.py`** — same message on the same code path, through the standard library instead of loguru.

   ```diff
   -                    logger.error(
   +                    warnings.warn(
   ```

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
