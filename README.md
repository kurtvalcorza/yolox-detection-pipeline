# yolox-detection-pipeline

DIMER inference and bounded fine-tuning pipeline for **YOLOX-S**, the small anchor-free detector from
`Megvii-BaseDetection/YOLOX`, pinned to the `yolox_s.pth` asset of upstream release `0.1.1rc0`.

Two capabilities over one checkpoint:

- **Detection** — the 80 COCO classes, score-ordered xyxy pixel boxes, caller-owned confidence and NMS
  thresholds.
- **Adaptation** — a bounded SimOTA fine-tune that re-heads the detector onto your own class
  vocabulary, scores it against a held-out split with COCO-style average precision, and exports a
  single reloadable artifact.

```python
from yolox_detection_pipeline import YoloxDetectionPipeline
from yolox_detection_pipeline.samples import SIGN_CLASSES, sign_dataset, tutorial_scene

pipe = YoloxDetectionPipeline.from_pretrained(allow_download=True)   # verifies SHA-256 first
scene, references = tutorial_scene()
print(pipe.detect(scene, threshold=0.3, nms_threshold=0.3)["detections"])

records = sign_dataset(40, seed=0)                                    # your own records work too
adapter = YoloxDetectionPipeline.from_pretrained(class_names=SIGN_CLASSES, seed=0)
adapter.finetune(records[:30], epochs=6)
print(adapter.evaluate(records[30:])["ap50"])
adapter.save_artifact("outputs/yolox-detection-adapter.pt")
```

## Why the model code is vendored

YOLOX is not installable into a pinned runtime. The PyPI `yolox` sdist is the official Megvii source,
but its `install_requires` pins `onnx==1.8.1`, `onnxruntime==1.8.0` and `onnx-simplifier==0.3.5`
beside a bare `torch>=1.7`/`torchvision`, and `pip install --no-deps yolox` does not help either:
`yolox.models.yolo_head` imports `yolox.utils`, whose `__init__` re-exports modules needing `cv2`,
`loguru` and `thop`.

The eight modules this package actually runs are therefore copied from upstream commit
`419778480ab6ec0590e5d3831b3afb3b46ab2aa3` with **three declared edits**, keeping their upstream file
names and upstream relative imports. `docs/UPSTREAM.md` records the revision, a SHA-256 per upstream
file and a diff of every edit; `python tools/vendor_upstream.py --check` re-fetches upstream and fails
on any divergence, and runs in CI. The result is a runtime of `torch`, `torchvision`, `numpy` and
`pillow` — no `transformers`, no `huggingface-hub`, no OpenCV, and no `yolox` distribution.

## Provenance

YOLOX publishes **no Hugging Face model repository**, so the pin is a release tag plus a digest rather
than a Hub revision:

| | |
|---|---|
| Upstream | `Megvii-BaseDetection/YOLOX` |
| Revision | `0.1.1rc0` (immutable GitHub release tag) |
| Asset | `yolox_s.pth`, 72,089,125 bytes |
| SHA-256 | `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30` |
| Licence | Apache-2.0 |
| Published | 2021-08-18 |

The checkpoint is a **pickle**, so the order matters: `verify_snapshot` re-hashes the file and raises
on a mismatch *before* `torch.load(..., weights_only=True)` is called, and `from_pretrained` refuses to
import torch at all until the snapshot verifies. Upstream's own downloader
(`torch.hub.load_state_dict_from_url`, no digest check) is deliberately not vendored. See
[`docs/WEIGHTS.md`](docs/WEIGHTS.md).

## Two facts about the input pipeline

- **Channel order is BGR**, raw 0–255, with no `/255` and no mean/std normalisation — upstream feeds
  `cv2.imread` output straight to the network. Feeding RGB is not a no-op: on this repository's
  tutorial scene it loses the bench entirely and takes the stop sign from 0.962 to 0.839.
- **The letterbox pads with grey 114**, aspect ratio preserved, image at the top-left. It is
  reproduced with Pillow rather than OpenCV; the resize is skipped outright when the input is already
  640×640, so every sample here is unaffected by the kernel difference.

The package owns both, so a caller never has to.

## Tutorial

[`tutorials/yolox_detection_finetune_colab.ipynb`](tutorials/yolox_detection_finetune_colab.ipynb) is
a standalone NOTEBOOK_SPEC 2.0 `E2E` notebook: it carries the package verbatim, needs no clone and no
DIMER service, and its default `Run all` path really adapts the model — COCO detection on a drawn
scene, then validate → split → baseline → bounded fine-tune → evaluate → new-data inference → export →
fresh reload. See [`tutorials/README.md`](tutorials/README.md).

## Development

```bash
python -m pip install -e . --no-deps
PYTHONPATH=src pytest -q -o addopts= tests      # offline; no checkpoint required
ruff check src tests tools
python tools/vendor_upstream.py --check         # vendored code still matches upstream
python tools/build_notebook.py --check          # notebook still matches the package
PYTHONPATH=src python tools/validate_release_assets.py
```

## Release status

**Candidate.** The package, its tests and the notebook are complete, and the notebook has been
executed top-to-bottom once in a fresh local CPU kernel. That is pre-flight evidence, not
supported-runtime evidence: promotion to Release-grade requires a clean-room execution recorded in
[`docs/release-verification.md`](docs/release-verification.md).

## Licences

Repository code: Apache-2.0 (`LICENSE`). Vendored upstream YOLOX model code: Apache-2.0, upstream's
own licence text at `src/yolox_detection_pipeline/LICENSE-YOLOX`. Model weights: Apache-2.0 as
published upstream, redistributed unmodified and not committed to this repository.
