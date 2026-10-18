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
| 6 | Notebook executes top-to-bottom in a **supported hosted runtime** from a cold start | **open** — the recorded Kaggle T4 run (2026-09-15) needed one manual restart after the old install cell, so it is workflow history, not one-pass evidence (2026-10-05 review, YXD-M1; state corrected 2026-10-10). The 2026-10-05 note that this gate needs re-qualification with `restarted: false` is answered by the one-pass Colab T4 run of the regenerated notebook recorded below (2026-10-10, blob `fe56cb6950f3`, commit `2c3d8bc`, no restart); the gate stays open until a reviewer/integrator accepts that record. |
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

**Disclosure (2026-10-18, review R116-M1).** This grid chose the defaults on **held-out** AP50 of
`sign_dataset(24, seed=0)` split 18 / 6. The generator is prefix-stable, so those 24 images are the first 24
of today's 40: four of today's ten held-out images were in the grid's held-out set and three were its
training images. Today's 10-image split is therefore the **exposed development test**, reported for
continuity only. Selection now runs on a separate validation pool under `docs/yolox-preregistration.md`,
and only a fresh test is scored as the measurement. On AP@[.50:.95] the leader of this grid was 8 epochs
at 2e-3, not the default; that row is C2 of the new grid.

`DEFAULT_EPOCHS = 6`, `DEFAULT_LEARNING_RATE = 1e-3` and `freeze_backbone=True` were chosen by this
grid, a configuration chosen on its evaluation split. The criterion was AP50 on the six held-out images
after training, everything else fixed and seeded:

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
| 2026-09-15 | Kaggle GPU kernel (`kurtvalcorza/dimer-nb2-yolox-detection-finetune` v1), fresh container assigned **Tesla T4** (driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed in fresh IPython interpreter via `nbclient 0.10.2` | `yolox_detection_finetune_colab.ipynb` | blob `5c09465debca9e836854b91e8b671e0728b03b67` at commit `8bcfda9` | **PASS after a manual restart** (not one-pass evidence; see gate 6) — 22/22 code cells, 182.3 s wall (pass 1: 155.2 s install + restart; pass 2: 27.0 s full run); matched the smoke figures in AP50 and geometry, not in AP@[.50:.95] (GPU run-to-run and code differences; CUDA determinism is not set): baseline AP 0.2083 / AP50 0.2222 → adapted AP 0.8403 / AP50 1.0000, loss 6.2880 → 1.5985, 1,891,096 of 8,938,456 parameters trained, COCO IoU 0.946/0.914/0.946/0.000/0.959, artifact 35,998,883 bytes over 462 tensors, fresh reload identical (reloaded AP 0.8403 / AP50 1.0000), tampered artifact refused, all five new-data sign objects correctly detected (same-label IoU 0.871–0.943). 2 files, 72 MB staged (`yolox_s.pth`). |
| 2026-10-10 (09:52:59 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-2c3d8bc-c7f4`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (32 locked packages, setup 56 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`; default path, every form field at its default | `yolox_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `b420ebea066d`, `module_sha256` `2f7b3af55ee6…`, generator `build_notebook.py/2.1-swc-yolox`, `notebook_spec` 2.2) | blob `fe56cb6950f3bac15677b7f1e2931a665ee60b91` at commit `2c3d8bc62b255eccd82a1b9b62eec57d872933d8` | **PASS** — one pass, no restart, 0 errors; 24/24 code cells in order (cells 4–13, the ten carried modules, print nothing); 103.3 s wall; `yolox_s.pth` 72,089,125 bytes digest-verified; COCO scene 4/5 matched (IoU 0.946/0.914/0.946/0.000/0.959); RGB order 5 detections, stop sign 0.839; blank and noise 0 boxes at demo thresholds; 30/10 split; baseline AP 0.2083 / AP50 0.2222 → adapted AP 0.7766 / AP50 1.0 / AP75 1.0; loss 6.2880 → 1.5954 in 5.9 s, `num_fg` 7.87–8.87, 1,891,096 of 8,938,456 parameters trained; new-data seed 99: all five objects correctly labelled, no spurious boxes; artifact 35,998,883 bytes over 462 tensors, fresh reload identical, tampered artifact refused. Evidence in `docs/execution-evidence/2026-10-10-2c3d8bc/`: executed notebook SHA-256 `3d3b6433ffa2a3767204f511ec4a561cefa56f15514bf7215de49b49a270176f`, `run_summary.json` `dca98837888981aedd968cc70af01fe35e0311e5cac4250e977a1ff9e3298f4b`, `exec.log` `e4a5c5f290d2b2b454a2d22ba0c4b3dc834c4219089f7bf27fdfd3fe46f4c79e`. Not exercised: both BYOD branches, the Section 12 activity, a browser Run all |
| 2026-10-18 (08:49:26 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-ad795d2-c7f4`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (32 locked packages, setup 46 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`, device class `cuda-t4`; default path, every form field at its default | `yolox_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `b242f44c2c37`, `module_sha256` `a5292930cd81…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `5fe2ef5d4628ab20855ffbf011e76c9ed9891077` at commit `ad795d20747cd867e8f935295441e594b0944db3` | **PASS — pre-registered evaluation of record** (first hosted run of the pre-registered settings and amendment 1; first scoring of the 200-image fresh test by any run). One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 134.6 s wall. Grid C0 / C1 / C2 validation AP 0.6966 / 0.8554 / 0.7122 (AP50 1.0 each), C1 selected; fresh test C1 AP 0.8573 vs baseline 0.1363, +0.7211, 95 % CI [0.6983, 0.7435], verdict *improved* (the baseline's class head is untrained, so a large gain was expected); exposed development test C1 AP 0.8516 / AP50 1.0. Details and hashes: "Hosted T4 evaluation of record (2026-10-18)" below; evidence in `docs/execution-evidence/2026-10-18-ad795d2/`. Not exercised: both BYOD branches, the Section 14 activity, a browser Run all |

## 2026-10-05 review fixes — verification pending

The 2026-10-05 review (`docs/reviews/2026-10-05-notebook-review/yolox_detection_finetune_colab_Review.md`) found that the Kaggle T4 row needed one manual restart after the in-kernel install cell, so gate 6 rests on a restarted run. The notebook was then regenerated (`yolox_detection_finetune_colab_Fixes.md`):
- an isolated, hash-locked `uv` environment replaces the in-kernel install and its restart guard (spec 2.2);
- the BYOD dataset branch now reads `annotations.json` from `BYOD_DATASET_DIR` or the upload dialog and runs adaptation, reload and a result JSON;
- the guided layer was added;
- Section 8 starts every run from the verified base.

**The regenerated blob `fe56cb6950f3` (commit `2c3d8bc`) ran once on a fresh Colab Tesla T4 on 2026-10-10 with no restart** (24/24 code cells, 0 errors, 103.3 s; see Recorded executions). The open items are:
- reviewer/integrator acceptance of that one-pass record (gate 6);
- the REL12 BYOD journey.

`STATUS.md`, `README.md` and gate 6 were corrected on 2026-10-10 to match this record: status **Candidate**, gate 6 open (YXD-m3).

## 2026-10-18 review fixes (row 116): pre-registered grid and fresh test

The configuration is now chosen on a validation pool by the grid fixed in
[`yolox-preregistration.md`](yolox-preregistration.md), committed alone before any code. The fresh test
is scored only under those settings. Nothing below is clean-runtime evidence.

**Pools** (`docs/yolox-detection-pools.json`; per-image SHA-256 over decoded pixels under Pillow 11.3.0):

| Pool | Definition | Images | Boxes (stop / yield / speed-limit) | Digest |
|---|---|---|---|---|
| Training | `sign_dataset(40, seed=0)` split, train part | 30 | 61 (16 / 28 / 17) | `60b0aaeea62c…` |
| Exposed development test | same split, held-out part | 10 | 21 (5 / 6 / 10) | `8729a7c00ce7…` |
| Validation | `sign_dataset(40, seed=1161)` | 40 | 79 (26 / 26 / 27) | `a75c4381677c…` |
| Fresh test | `sign_dataset(200, seed=1160)` | 200 | 399 (119 / 145 / 135) | `6900526c80ac…` |

- Every pair of pools, and each pool against Section 10's seed-99 images, shares 0 images and 0 box
  layouts.
- Nearest-neighbour mean absolute pixel difference (64×64 thumbnails), median per image: fresh vs
  training 0.0257, fresh vs validation 0.0284, validation vs training 0.0319. Within training it is
  0.035. The pools are as far apart as the generator's own spread: disjoint images, the same renderer.
- The September grid's 24 images are the first 24 of `sign_dataset(40, seed=0)` (checked by digest).

**CPU development look (SCORE_ONCE_RULE item 8), fresh test MASKED.** 2026-10-18, local Windows CPU
(Intel Core Ultra 9 275HX), CI pins (`torch 2.14.0+cpu`), code at `825771a` with the notebook
regenerated from it. The committed notebook cells were run in-process with
`DIMER_MASK_FRESH_TEST=1` and `RUN_GRID` forced to `on`, so the grid ran on CPU. The report says
`test_look` "not scored (fresh test masked for a development check)", `preregistered_settings_hold`
true, `settings_differing` empty, grid complete. These numbers were not used to change any setting.

| Config | Validation AP@[.50:.95] | Validation AP50 | Final `total_loss` | Fine-tune s (CPU) |
|---|---|---|---|---|
| Baseline (re-headed) | 0.1767 | 0.1851 | — | — |
| C0 6 ep, 1e-3 | 0.7760 | 1.0000 | 1.66196 | 67.3 |
| C1 10 ep, 1e-3 | **0.8871** | 1.0000 | 1.35705 | 71.5 |
| C2 8 ep, 2e-3 | 0.8285 | 1.0000 | 1.75448 | 55.6 |

- The rule selects **C1** on CPU: +0.1112 validation AP over C0 with no AP50 loss. The keep-baseline
  rule keeps the adapted model (+0.7104 over the baseline).
- Exposed development test (descriptive only): baseline 0.2083 / 0.2222, which matches every earlier
  record; C1 0.9050 / 1.0000. C0 reproduced the 2026-09-15 CPU loss exactly (6.28799 → 1.66196).
- New-data images (seed 99), C1: 5 of 5 with the right label, same-label IoU 0.906–0.969. Reload of
  the artifact gave identical validation scores.
- Old → new: the default configuration changes from C0 (chosen on the exposed split) to the
  pre-registered rule's choice. CPU picks C1. The hosted T4 grid decides.

**Evaluation of record:** the Colab CLI T4 run of 2026-10-18 (commit `ad795d2`, blob `5fe2ef5d`); see
"Hosted T4 evaluation of record (2026-10-18)" below. Hosted fresh-test looks: 1. Reproductions: none yet.
`RECORDED_EVALUATION` in `protocol.py` is still `None` (recording it regenerates the notebook), so until then
a later identical hosted run labels itself "pre-registered evaluation (this runtime)", not "reproduction";
this file is what says it is a reproduction.

**Note (2026-10-18, requote):** `RECORDED_EVALUATION` in `protocol.py` now carries this record (date,
commit `ad795d2`, blob `5fe2ef5d`, selected C1, the fresh-test numbers and the validation grid), and the
notebook was regenerated from it. From that revision on, every pre-registered scoring, including the first
one in a new runtime, is labelled "reproduction", writes `_reproduction_<n>` files (never the main
report or results files) and prints whether it matches the record. The paragraph above describes the
revision the record ran.

**Redaction (review R116-m3, own commit).** In `execution-evidence/2026-10-10-2c3d8bc/exec.log`, lines
1 and 191 had the operator's home-directory run path replaced with `<local-run-dir>/`.
- Captured SHA-256: `e4a5c5f290d2b2b454a2d22ba0c4b3dc834c4219089f7bf27fdfd3fe46f4c79e`. This is the
  value `run_summary.json` still records.
- Committed SHA-256: `e042f9783e063c5a8d68890c4be210076ebc6f86e0b9a8e28bf238c4447af351`.

**Tie rule (review R116-S1).** `average_precision` now scores tied scores as one operating point, so
the result does not depend on image order. With distinct scores the numbers are unchanged; every CPU
number above matches the earlier records where the two overlap.

### Round 2 (PR #8 peer review, 2026-10-18)

- `docs/yolox-preregistration-amendment-1.md` was committed alone, before the code. Its changes:
  - only a T4 can score the fresh test (`device_class` is a checked setting);
  - Section 9 recomputes the grid gate and never trusts state left by an interrupted Section 8;
  - an existing main report file spends the pre-registered slot;
  - grid rows are written per run (`outputs/grid/run_<k>/`);
  - Sections 10-11 use the selected system.
- Erratum: |CPU - T4| <= 0.034 and T4 - T4 = 0.064 AP@[.50:.95] for C0 on the exposed development test.
  The "about 0.1 AP" in the pre-registration is wrong. The rule is unchanged.
  - **Correction (2026-10-18, after the T4 record):** this erratum is too strong. It holds only for C0 on
    the exposed development test, across the earlier records. On validation, the T4 record differs from
    the CPU masked check by up to 0.116 AP@[.50:.95]: C0 0.6966 vs 0.7760 (0.079), C1 0.8554 vs 0.8871
    (0.032), C2 0.7122 vs 0.8285 (0.116). The CPU check also ran older code (`825771a`; the T4 run used
    `b242f44`), so these gaps mix device and code effects. The rule is unchanged.
- No fresh-test score exists: the CPU development look was masked, and there are 0 hosted looks.
  `RECORDED_EVALUATION` is still `None`. (State at round 2; superseded by the 2026-10-18 evaluation of record below.)

## Hosted T4 evaluation of record (2026-10-18)

**REL13 label for every number in this section:** Colab CLI 0.7.4, fresh Colab Tesla T4, 2026-10-18,
commit `ad795d20747cd867e8f935295441e594b0944db3`, notebook blob `5fe2ef5d4628ab20855ffbf011e76c9ed9891077`
(embedded `NOTEBOOK_SOURCE.repository_revision` `b242f44`; `b242f44..ad795d2` only regenerates the notebook).
Role: **pre-registered evaluation of record**, the first hosted run of the pre-registered settings and
amendment 1, and the first time any run scored the 200-image fresh test. Later hosted runs with identical
settings are reproductions of this evaluation. Executor: Colab CLI 0.7.4 sequential execution
(`colab exec -f`), not a browser Run all. Forms and uploads were not exercised. Status stays **Candidate**.

**Identity and runtime.**
- The VM ran a byte-for-byte copy of the committed notebook: input SHA-256 `b7bfaf7e…`, git blob
  `5fe2ef5d`, unchanged after the run.
- `inspect_run.py` passed: 56/56 cells with the same ids and order, and source identity OK.
- Hardware T4 (`colab status`). Section 8 printed `device Tesla T4 (cuda-t4)`; `cuda:0`;
  `torch 2.14.0+cu130`.
- TF32: neither the notebook nor the package sets or prints a TF32 flag. The profile is float32, and
  the Tesla T4 (Turing) has no TF32 mode, so TF32 does not apply to this run. That is inferred from the
  hardware, not printed by the run.
- CUDA determinism is not set. C0 and C1 share a seed but already differ at epoch 2 on the T4; on CPU
  their first six epochs are identical.

**Score-once checks (as printed).**
- Section 9 printed `test_look` "pre-registered evaluation (this runtime)" and wrote
  `outputs/yolox_detection_evaluation_report.json`. Settings: batch 2, seed 0, backbone frozen,
  validation pool 40 images / 79 boxes (seed 1161), fresh pool 200 images / 399 boxes, grid complete with
  no problems, device class `cuda-t4`.
- The report and results JSON stayed on the released VM. The `preregistered` boolean and
  `settings_differing` are therefore shown only through that label, not as committed files.

**Baseline (re-headed, before fine-tuning).** The six `head.cls_preds.{0,1,2}.{weight,bias}` tensors
could not transfer from the COCO checkpoint, so the baseline's classification head is untrained. A large
gain was therefore expected (review F8). The informative parts are the size of the gain, the width of
the interval, and that it held on images no choice was made on.
- Validation: AP 0.1767, AP50 0.1851; per-class AP50 stop 0.0554, yield 0.0, speed-limit 0.5.
- Exposed development test: AP 0.2083, AP50 0.2222; per-class AP50 0.0 / 0.0 / 0.6667. This equals
  every earlier record.

**Grid (validation; this chose the configuration).**

| Config | Validation AP@[.50:.95] | AP50 | Gain over C0 | Final `total_loss` | Fine-tune s (T4) | CPU masked check AP (`825771a`) | T4 − CPU |
|---|---|---|---|---|---|---|---|
| C0 6 ep, 1e-3 | 0.6966 | 1.0 | — | 1.87603 | 6.4 | 0.7760 | −0.0794 |
| C1 10 ep, 1e-3 | **0.8554** | 1.0 | +0.1587 | 1.38442 | 7.4 | 0.8871 | −0.0317 |
| C2 8 ep, 2e-3 | 0.7122 | 1.0 | +0.0155 | 2.60175 | 6.0 | 0.8285 | −0.1163 |

- **Selection, re-applied by hand.** C1 gains 0.1587 ≥ 0.03 with AP50 equal, so it qualifies. C2
  gains 0.0155 < 0.03, so it does not. C1 is selected, which matches the printed reason. Keep-baseline:
  C1 − baseline = 0.6787 at printed precision (printed 0.6786 from unrounded values) ≥ 0.03, so the
  adapted C1 is the selected system.
- AP50 is 1.0 for every config, so each class's AP50 is 1.0. The AP50 guard decided nothing.
- The grid's fine-tuning totalled 19.8 s against the pre-registration's planning estimate of about
  24 s. The whole run took 134.6 s wall.
- C2's loss rose in its last two epochs (1.8718 → 2.0892 → 2.6018). Its CPU run kept falling
  (→ 1.7545).

**Against the CPU masked check.** The CPU check also selected C1, so the decision is unchanged. Every
number differs.
- T4 validation AP is lower for all three configs: by 0.079 (C0), 0.032 (C1) and 0.116 (C2).
- On CPU, C2 also qualified (+0.0525) but lost to C1's larger gain. On the T4 it does not qualify.
- The C1 margin over C0 is 0.1587 on the T4 against 0.1112 on CPU. Keep-baseline is 0.6787 against
  0.7104.
- Only 2 of 24 epoch loss lines are identical (C0 and C1 epoch 1). C2 epoch 1 differs in the 4th
  decimal.
- CUDA non-determinism (note added 2026-10-18): on the T4, C0 and C1 share a seed and print the same
  epoch-1 line, but split at epoch 2 (total loss 3.7556 for C0, 3.7558 for C1). On CPU their epochs
  1–6 are identical. CUDA determinism is not enforced, so a hosted reproduction can differ in later
  decimals and, near the 0.03 margin, in the selected config.
- The exposed development test for C1 is 0.8516 on the T4 against 0.9050 on CPU (AP50 1.0 both).
- Fine-tuning took 19.8 s on the T4 against 194.4 s on CPU.
- The CPU check ran an earlier code revision (`825771a`) than this run (`b242f44`, round-2 code
  `51f7e72`). So the T4 − CPU gaps mix device effects with code changes.
- The round-2 erratum (|CPU − T4| ≤ 0.034 for C0 on the exposed test) does not carry over to
  validation: here C0 differs by 0.079 and C2 by 0.116. The pre-registration's "about 0.1 AP" is
  closer for this pool. Neither changes the rule.

**Fresh test (primary endpoint; 200 images, 399 boxes).**

| System | AP@[.50:.95] | AP50 | Per-class AP50 (stop / yield / speed-limit) |
|---|---|---|---|
| Selected (C1) | **0.8573** | 1.0 | 1.0 / 1.0 / 1.0 |
| Pretrained baseline (re-headed) | 0.1363 | 0.1449 | 0.013 / 0.0 / 0.4217 |

- Difference +0.7211; paired image bootstrap 95 % CI [0.6983, 0.7435] (n_boot 10 000, seed 20261018,
  as pre-registered).
- Verdict **improved**: the lower bound 0.6983 is > 0 and the gain is ≥ 0.03.
- Re-derived from the printed values: 0.8573 − 0.1363 = 0.7210 (printed 0.7211, from unrounded
  values).
- Observed CI half-width: 0.0226. This replaces the pre-registration's planning estimate of ±0.07 AP.
  The minimum detectable change (planned about 0.10) is far below the observed gain.
- The bootstrap was not re-run, because the per-image detections stayed on the VM.
  - **Note (2026-10-18, requote):** Section 9 now also writes
    `outputs/yolox_detection_fresh_test_per_image<suffix>.json`. It holds per-image box counts and true
    positives at IoU 0.5 for both systems, plus the match tables the bootstrap resamples, so a later
    reproduction's interval can be re-run from the file. This is display only and changes no score.
- The fresh test measures in-distribution generalisation (same renderer, same shapes). It is not a
  detection benchmark.

**Exposed development test (descriptive only; decides nothing).** Baseline AP 0.2083 / AP50 0.2222;
C1 AP 0.8516 / AP50 1.0000 (+0.6432 / +0.7778).

**Other printed checks.**
- COCO scene: 4/5 matched (IoU 0.946 / 0.914 / 0.946 / 0.000 / 0.959). RGB order: 5 detections,
  stop sign 0.839. Blank and noise: 0 boxes at the demo thresholds.
- New-data seed 99 with C1: 5/5 objects correctly labelled, same-label IoU 0.881–0.942.
- Artifact: 36,000,287 bytes, 462 tensors, SHA-256 `eef542e6…`. The fresh reload reproduced
  validation AP 0.8554 / AP50 1.0, and the tampered artifact was refused.
- BYOD branches off.

**Evidence** (`docs/execution-evidence/2026-10-18-ad795d2/`, `-text`). SHA-256 values were checked
end to end: the VM run directory, the local copy, and the committed blob.
- Executed notebook: `d997ac73b1e6c8e42f95a7d7c130865f3a5f6909decbeb7c0143f5d32cc5f7b3`.
- `run_summary.json`: `e98d4245f3052ae0775b887b8ff35eac0584259373ea1762dc74f5cb6ac5a3aa`.
- `exec.log`: in lines 1 and 253, the operator's home-directory run path was replaced with
  `<local-run-dir>/`.
  - Captured SHA-256: `c8aacc7a67672603f5741704f6cdae30b1e572c065855ccc6a482df05844a876`.
  - Committed SHA-256: `345229e6fb37ca992e082f7270e5b251a90737ad1bc780e22b34598faa2665b7`.
  - `run_summary.json` does not record the log's hash.
