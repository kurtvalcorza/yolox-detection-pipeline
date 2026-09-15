# Release verification

This file is the release gate for `tutorials/yolox_detection_finetune_colab.ipynb`. Static validation
(`tools/validate_release_assets.py`), the unit suite and CI are **not** clean-runtime execution
evidence; only an execution recorded under "Recorded executions" is.

## Gates

| # | Gate | State |
|---|---|---|
| 1 | Package unit suite passes offline, without the checkpoint | **met** — 113 tests |
| 2 | Vendored upstream code matches the pinned revision plus the declared edits | **met** — `tools/vendor_upstream.py --check` |
| 3 | Notebook matches the package byte-for-byte | **met** — `tools/build_notebook.py --check` (PAR3) |
| 4 | Static release-asset validation passes | **met** — `tools/validate_release_assets.py` |
| 5 | Notebook executes top-to-bottom in a fresh **local** kernel | **met** — pre-flight only, recorded below |
| 6 | Notebook executes top-to-bottom in a **supported hosted runtime** from a cold start | **open** |
| 7 | Weights uploaded to the DIMER Model Repository | **open** — owner action |

Gate 6 is what separates **Candidate** from **Release-grade**. The local run in gate 5 pre-staged the
checkpoint and used a venv that already held the pins; it therefore exercises the notebook's logic but
not its install cell, its download path, or a cold hosted runtime.

## Smoke evidence

Measured 2026-09-15 on the reference machine (Intel Core Ultra 9 275HX, Windows, Python 3.12.10,
`torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `numpy 2.5.3`, `pillow 11.3.0`) with
`CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`, float32.

### Load and verify

| Step | Result |
|---|---|
| `verify_snapshot` | 0.08 s, 1 file, 72,089,125 bytes |
| `torch.load(weights_only=True)` | succeeds; top-level keys `amp`, `model`, `optimizer`, `start_epoch` |
| `load_state_dict(strict=True)` | all 462 keys matched, none missing or unexpected |
| `from_pretrained` | 2.64 s, 8,968,255 parameters (upstream README: 9.0 M) |

### COCO detection on the drawn tutorial scene

`detect` at the demo thresholds 0.3 / 0.3 → 6 detections in 0.11 s (0.08 s on the second call):

| Label | Score | Box |
|---|---|---|
| stop sign | 0.962 | 53, 134, 187, 266 |
| traffic light | 0.863 | 300, 120, 359, 254 |
| clock | 0.855 | 458, 130, 602, 274 |
| bench | 0.545 | 439, 500, 612, 582 |
| kite | 0.369 | 241, 519, 321, 600 |
| clock | 0.309 | 240, 519, 321, 600 |

`evaluation_report` against the drawn references, matching same-label only:

| Reference | `box_iou` | Matched score |
|---|---|---|
| stop sign | 0.946 | 0.962 |
| traffic light | 0.914 | 0.863 |
| clock | 0.946 | 0.855 |
| bench | 0.959 | 0.545 |
| **sports ball** | **0.000** | — (no `sports ball` detection at any threshold) |

4/5 references matched at IoU ≥ 0.5. At the evaluation thresholds 0.01 / 0.65 the same scene yields 7
detections.

### Degenerate inputs

| Input | Demo pair 0.3/0.3 | Evaluation pair 0.01/0.65 |
|---|---|---|
| Blank white 640×640 | 0 detections | 0 detections |
| Uniform noise (NumPy default generator, seed 0) | 0 detections | 1 (`bear`, 0.073) |

### Channel order

The same letterboxed tensor, fed as RGB instead of upstream's BGR:

| | Detections | Stop sign | Bench | Spurious second clock |
|---|---|---|---|---|
| BGR (upstream) | 6 | 0.962 | 0.545 | 0.309 |
| RGB (the mistake) | 5 | 0.839 | absent | 0.441 |

### End-to-end adaptation

`sign_dataset(40, seed=0)` → 40 images, 82 boxes (`stop-sign` 21, `yield-sign` 34,
`speed-limit-sign` 27), no validation findings. `split_records(holdout=0.25, seed=0)` → 30 train /
10 held out. Re-heading onto `SIGN_CLASSES` took 0.20 s and reinitialised exactly six tensors:
`head.cls_preds.{0,1,2}.{weight,bias}`.

`finetune(epochs=6, batch_size=2, learning_rate=1e-3, seed=0, freeze_backbone=True)` — 25.9 s,
15 steps per epoch, 1,891,096 of 8,938,456 parameters trainable:

| Epoch | total | iou | conf | cls | l1 | num_fg |
|---|---|---|---|---|---|---|
| 1 | 6.28799 | 1.65693 | 2.50648 | 2.12457 | 0.0 | 8.21 |
| 2 | 3.75580 | 1.77725 | 1.19503 | 0.78353 | 0.0 | 7.87 |
| 3 | 2.85105 | 1.43285 | 0.77167 | 0.64653 | 0.0 | 8.07 |
| 4 | 2.38556 | 1.27658 | 0.61288 | 0.49610 | 0.0 | 8.31 |
| 5 | 1.95995 | 1.05263 | 0.48550 | 0.42183 | 0.0 | 8.55 |
| 6 | 1.66196 | 0.87565 | 0.44800 | 0.33831 | 0.0 | 8.79 |

`l1_loss` is 0.0 throughout by design: upstream enables the L1 box term only for the last 15 of 300
epochs, and this bounded run leaves `use_l1` off.

Held-out scores (10 images, evaluation thresholds, COCO 100-detection cap):

| Metric | Baseline | Adapted | Change |
|---|---|---|---|
| AP@[.50:.95] | 0.2083 | **0.8110** | +0.6027 |
| AP50 | 0.2222 | **1.0000** | +0.7778 |
| AP50 `stop-sign` | 0.0 | 1.0 | |
| AP50 `yield-sign` | 0.0 | 1.0 | |
| AP50 `speed-limit-sign` | 0.667 | 1.0 | |

The baseline is non-zero because only the classification branch is re-initialised; the COCO-trained
box and objectness heads already localise a sign-shaped object.

New-data inference on three images from an unseen seed (99), demo confidence 0.3 / NMS 0.45:

| Image | Truth | Detections | Same-label `box_iou` |
|---|---|---|---|
| 0 | yield-sign ×2 | yield-sign 0.911, yield-sign 0.906 | 0.902, 0.887 |
| 1 | speed-limit-sign, yield-sign | yield-sign 0.917, speed-limit-sign 0.887 | 0.886, 0.949 |
| 2 | stop-sign | stop-sign 0.912 | 0.916 |

All five objects found with the correct label and no spurious boxes.

### Artifact

`save_artifact` → 35,998,819 bytes over 462 tensors in 0.08 s. `load_artifact` in a fresh object
(0.08 s) reproduced AP 0.8110 / AP50 1.0000 **exactly**. An artifact with a tampered `format` tag was
refused with `ValueError`.

### Helper sanity

`average_precision` returns 1.0 for exact predictions and 0.0 for no predictions, penalises a
false positive ranked above a hit (AP50 0.5) and correctly does **not** penalise one ranked below all
hits, which is COCO's own semantics.

## The fine-tuning grid behind the defaults

`DEFAULT_EPOCHS = 6`, `DEFAULT_LEARNING_RATE = 1e-3` and `freeze_backbone=True` were chosen by
measurement on this repository's own sample dataset, not by taste. Held-out AP50 after training,
everything else fixed and seeded:

| Epochs | Learning rate | Backbone | Train time | AP50 after | AP after |
|---|---|---|---|---|---|
| 3 | 1e-3 | frozen | 10.6 s | 0.3546 | 0.1088 |
| 3 | 1e-3 | frozen | 9.9 s | 0.3546 | 0.1088 |
| 4 | 2e-3 | frozen | 10.2 s | 0.3096 | 0.1374 |
| **6** | **1e-3** | **frozen** | **20.2 s** | **0.8998** | **0.3870** |
| 6 | 2e-3 | frozen | 18.8 s | 0.6270 | 0.3005 |
| 8 | 2e-3 | frozen | 92.5 s | 0.8404 | 0.4945 |
| 6 | 1e-3 | **unfrozen** | 91.3 s | **0.0000** | **0.0000** |

Two results are load-bearing. The unfrozen row is a **collapse**, not a slow start: a full fine-tune
at this learning rate on 30 small images destroys the COCO-pretrained features the adaptation depends
on, which is why the default freezes the backbone and why the card warns about changing it. And the
two repeated `(3, 1e-3, frozen)` rows agree to every digit, including per-image detection counts,
which is the evidence that the seeding is complete.

That grid ran on 24 images (18 train / 6 held out); raising the sample to 40 images at the chosen
configuration is what produced the AP50 1.0000 / AP 0.8110 recorded above.

### A defect this grid found

The first grid run showed the pre-adaptation baseline moving between 0.4455 and 0.2121 AP50 across
otherwise identical configurations. Cause: the re-headed classification layers were initialised from
torch's **global** RNG, which nothing seeded until `finetune` ran, so the baseline depended on
whatever the ambient RNG state happened to be. `from_pretrained` now takes a `seed` and seeds that
initialisation under a saved-and-restored global RNG state (NOTEBOOK_SPEC 2.0 ENV7). Without the fix
the before/after comparison would have been meaningless.

### A second defect, in the sample data

The stop-sign reference box was originally the octagon's **circumscribing square**, `[cx ± r, cy ± r]`.
A regular octagon inscribed in a circle of radius `r` only reaches `r·cos(π/8) ≈ 0.924 r` along either
axis, so the reference box was 17% larger in area than the shape it named — capping the achievable IoU
and deflating every number computed against it. Fixing the extent to the drawn vertices raised the
tutorial scene's stop-sign `box_iou` from 0.902 to 0.946 and the post-adaptation AP from 0.7044 to
0.8110. A unit test now measures the rendered ink and asserts the reference box hugs it.

## Findings kept, not fixed

These are recorded behaviour, not defects, and should survive future edits:

1. **The drawn football is never detected as a `sports ball`** at any threshold; its box surfaces as
   `kite` 0.369 and a second `clock` 0.309. The evaluation report scores it 0.000 rather than dropping
   the reference.
2. **The clock face needs numerals.** A plain white disc with two hands was not read as a `clock` at
   all during scene probing, and worse, it made the *ball* become the clock (`clock` 0.748 on the
   ball's box). Three scene variants were probed and the numbered-face variant was chosen on hits at
   IoU ≥ 0.5 (4/5, against 3/5 for the plain face).
3. **`bench` scores low** (0.545) despite an almost exact box (IoU 0.959).
4. **Uniform noise yields one `bear` at 0.073** at the evaluation thresholds. Blank input yields
   nothing at either pair.

## Traps recorded for the fleet

- Upstream `postprocess` writes the xyxy conversion back into its argument **in place**, so a tensor
  produced under `torch.inference_mode()` raises `RuntimeError`. Upstream's own `tools/demo.py` uses
  `no_grad` for the same reason; this package does too, and says why at the call site.
- Omitting upstream's `head.initialize_biases(1e-2)` makes a re-headed fine-tune **collapse**: the
  objectness term saturates within a few steps and the model predicts one class at score 1.0 at every
  anchor. It was the cause of the first failed adaptation run here.
- The vendored `yolo_head` emits a `FutureWarning` for `torch.cuda.amp.autocast` under torch 2.14.
  It is upstream's code and is left verbatim rather than fixed, so the vendoring check keeps passing.

## Recorded executions

| Date | Runtime | Notebook | Blob | Result |
|---|---|---|---|---|
| 2026-09-15 | Local Windows CPU kernel (Python 3.12.10, `torch 2.14.0+cu130`, GPU hidden), pre-staged checkpoint, pins already present | `yolox_detection_finetune_colab.ipynb` | blob `004d6442d06c32598fc7a34ed4f405d9d767de72` at commit `716194a` | **PASS** — 22/22 code cells, 28.2 s; reproduced the smoke figures exactly (baseline AP 0.2083 / AP50 0.2222 → adapted AP 0.8110 / AP50 1.0000, loss 6.28799 → 1.66196, COCO IoU 0.946/0.914/0.946/0.000/0.959, artifact 35,998,883 bytes over 462 tensors, fresh reload identical, all five new-data objects correctly labelled) |
| 2026-09-15 | Same runtime, superseded | `yolox_detection_finetune_colab.ipynb` | blob `dd48b2167af1f0e088fddd2de5be6a64a3bcd721` at commit `2f79037` | **PASS** — 22/22 code cells, 39.3 s, identical results. Superseded by the row above after `load_artifact` gained the variant check; kept because the record should show which blob was actually executed at the time. |
| — | Supported hosted runtime, cold start | — | — | **not yet run** (gate 6) |
