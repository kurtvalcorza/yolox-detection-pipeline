# yolox-detection-pipeline

DIMER inference and bounded fine-tuning pipeline for **YOLOX-S**, the small anchor-free detector from
`Megvii-BaseDetection/YOLOX`, pinned to the `yolox_s.pth` asset of upstream release `0.1.1rc0`.

Two capabilities over one checkpoint:

- **Detection** — the 80 COCO classes, score-ordered xyxy pixel boxes, caller-owned confidence and NMS
  thresholds.
- **Adaptation** — a bounded SimOTA fine-tune that re-heads the detector onto your own class
  vocabulary, chooses its configuration on a validation pool by a pre-registered grid
  ([`docs/yolox-preregistration.md`](docs/yolox-preregistration.md)), scores a fresh test with COCO-style
  average precision only under the pre-registered settings, and exports a single reloadable artifact.

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
adapter.save_artifact("outputs/yolox-detection-adapter.safetensors")  # + yolox-detection-adapter.json sidecar
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
| Served file | `yolox_s.safetensors` (36,014,492 bytes, SHA-256 `8fa0c1135952a70148c3e3114ac1ddae2d399600053841b89738dc3694e3aaf0`), the safetensors conversion |
| Licence | Apache-2.0 |
| Published | 2021-08-18 |

The checkpoint is a **pickle**, so the order matters: `verify_snapshot` re-hashes the file and raises
on a mismatch *before* `torch.load(..., weights_only=True)` reads it, once, to write the pinned
safetensors conversion `yolox_s.safetensors`. The model is served only from that file
(`load_state_dict(strict=True)` after its own digest check), and `from_pretrained` refuses to import torch
at all until the snapshot or the conversion verifies. Upstream's own downloader
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
supported-runtime evidence. The recorded Kaggle T4 run of an earlier blob needed a manual restart
after its install cell, so it is not one-pass evidence either; the regenerated notebook (isolated
`uv` environment, no restart) — blob `fe56cb6950f3` (commit `2c3d8bc`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-10 (Colab CLI 0.7.4 sequential execution, 24/24 code cells, 103.3 s; on the 10-image split now called the exposed development test, because the defaults were chosen on it: baseline AP 0.2083 / AP50 0.2222 → adapted AP 0.7766 / AP50 1.0; fresh reload identical) — is the current one-pass record; status stays Candidate. The pre-registered **evaluation of record** ran on 2026-10-18: blob `5fe2ef5d4628` (commit `ad795d2`) on a fresh Colab Tesla T4 (Colab CLI 0.7.4 sequential execution, 25/25 code cells, 0 errors, 134.6 s). The validation grid selected C1 (10 epochs at 1e-3; validation AP C0 0.6966 / C1 0.8554 / C2 0.7122). On the 200-image fresh test, C1 scored AP@[.50:.95] 0.8573 against the re-headed baseline's 0.1363: +0.7211, 95 % CI [0.6983, 0.7435], verdict *improved*. The baseline's classification head is untrained, so a large gain was expected. The fresh test is drawn by the same generator, so it measures in-distribution generalisation, not detection on real images. Under amendment 2 (deterministic GPU training, fresh test 3), the **amendment-2 evaluation of record** ran on 2026-10-18: blob `a9dd3990a1d8` (commit `4429e0e`) on a fresh Colab Tesla T4 (Colab CLI 0.7.4, 25/25 code cells, 0 errors, 143.6 s), after two masked hosted runs printed the same grid hash. The grid selected C1 again (validation AP C0 0.7037 / C1 0.8937 / C2 0.8296). On fresh test 3 (200 images, seed 1162), C1 scored AP@[.50:.95] 0.8933 against the baseline's 0.1317: +0.7616, 95 % CI [0.7381, 0.782], verdict *improved*. A reproduction on 2026-10-18 (blob `6b295979c64f`, commit `f0d021e`, fresh Colab Tesla T4, 25/25 code cells, 0 errors, 151.8 s) labelled itself *reproduction*, printed the same grid hash and matched the record exactly. The round-2 record above stays as history. The current notebook is blob `b564dcae6b02` (main `21fefe0`, after PR #10): on 2026-10-25 it ran on fresh Colab Tesla T4 VMs, and the default path (25/25 code cells, 0 errors, 129.3 s) matched the amendment-2 record on fresh test 3 (C1 AP 0.8933 vs baseline 0.1317), exported the adapter as `.safetensors` plus its `.json` sidecar, reloaded to identical scores and refused both tampered artifacts; the Section 14 activity (`FREEZE_BACKBONE = False`) also passed. The serving change of 2026-10-24 (code `73eef81`) regenerated the notebook as blob `0c4430e821f5`, which loads the model from the pinned safetensors conversion `yolox_s.safetensors`. On 2026-10-24 that exact blob (commit `85f4838`) ran on fresh Colab Tesla T4 VMs: the default path (25/25 code cells, 0 errors, 167.3 s) matched the amendment-2 record on fresh test 3 (C1 AP 0.8933 vs baseline 0.1317), exported the adapter as `.safetensors` plus its `.json` sidecar, reloaded to identical scores and refused both tampered artifacts; the Section 14 activity (`FREEZE_BACKBONE = False`) also passed. The previous notebook, blob `4f0199a59dcb` (commit `1792c3f`, safetensors export), ran on fresh Colab Tesla T4 VMs on 2026-10-23: the default path (25/25 code cells, 0 errors, 149.9 s) matched the amendment-2 record, exported `yolox-detection-adapter.safetensors` plus its `.json` sidecar, reloaded to identical scores and refused both tampered artifacts; the Section 14 activity (`FREEZE_BACKBONE = False`) also passed. Earlier blobs' runs are history under REL14. Status stays Candidate, pending only the REL12 BYOD journey and the REL2 browser Run all. Promotion to Release-grade
requires a clean-room execution recorded in [`docs/release-verification.md`](docs/release-verification.md).

## Licences

Repository code: Apache-2.0 (`LICENSE`). Vendored upstream YOLOX model code: Apache-2.0, upstream's
own licence text at `src/yolox_detection_pipeline/LICENSE-YOLOX`. Model weights: Apache-2.0 as
published upstream, redistributed unmodified and not committed to this repository.
