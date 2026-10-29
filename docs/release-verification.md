# Release verification

**Current status (2026-10-24): Candidate.** **Safetensors serving (2026-10-24).** The model is now served from `yolox_s.safetensors`, a safetensors conversion of the upstream `.pth` whose SHA-256 (`8fa0c1135952…`) is pinned in code, instead of from the pickle itself (code commit `73eef81`). The regenerated notebook `yolox_detection_finetune_colab.ipynb` is blob `0c4430e821f5` (commit `85f4838`). Its default path and its Section 14 activity (`FREEZE_BACKBONE = False`) are covered by two hosted Colab T4 runs of that exact blob on 2026-10-24, both one pass with no restart and 0 errors ("Safetensors serving: hosted T4 runs (2026-10-24, commit 85f4838)" below). The remaining gates are REL12 (the BYOD journey) and REL2 (a browser Run all). Every hosted run of an earlier blob in this file stays as history (REL14), not evidence for this one.

The current blob `a644491f` (code commit `71896bc`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line, with no traceback, and stops Run all at that cell (previously the cell printed the refusal and Run all carried on); genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `b564dcae`, are history for it (REL14).

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
| 2026-10-18 (13:40:44 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-4429e0e-c7f4`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (32 locked packages, setup 52 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`, device class `cuda-t4`; determinism flags on (amendment 2); default path, every form field at its default | `yolox_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `ab130f33d564`, `module_sha256` `57f7ef9a454b…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `a9dd3990a1d8b18be6fbcc06ee218c19e0025b7c` at commit `4429e0e50372f14355fb23d801a754a57e3b364e` | **PASS — amendment-2 evaluation of record** (first hosted run under amendment 2 that scores fresh test 3; first scoring of fresh test 3 by any run). One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 143.6 s wall. Grid hash `184785885a81…`, equal to both masked determinism checks; grid C0 / C1 / C2 validation AP 0.7037 / 0.8937 / 0.8296 (AP50 1.0 each), C1 selected; fresh test 3 (seed 1162, 200 images, 386 boxes) C1 AP 0.8933 vs baseline 0.1317, +0.7616, 95 % CI [0.7381, 0.782], verdict *improved*; exposed development test C1 AP 0.9043 / AP50 1.0. Details and hashes: "Amendment-2 evaluation of record: hosted T4 (2026-10-18, commit 4429e0e)" below; evidence in `docs/execution-evidence/2026-10-18-4429e0e/`. Not exercised: both BYOD branches, the Section 14 activity, a browser Run all |
| 2026-10-18 (14:03:56 UTC start) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-f0d021e-c7f4`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (32 locked packages, setup 55 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`, device class `cuda-t4`; determinism flags on (amendment 2); default path, every form field at its default | `yolox_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `b74cdd9ad2be`, `module_sha256` `273937904596…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `6b295979c64fec16a6321e10a46a34971afe1ae4` at commit `f0d021ebbd5765dc45696d8180ec4a5b1971e3f1` | **PASS — reproduction of the pre-registered evaluation** (reproduction 1 of the amendment-2 record `4429e0e`; second hosted look at fresh test 3). One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 151.8 s wall. Section 9 printed `test_look` "reproduction", wrote the `_reproduction_1` files and printed "matches the record". Grid hash `184785885a81…`, equal to the record; every grid row (except fine-tune seconds) and all 24 epoch loss lines identical to the record; C1 selected; fresh test 3 C1 AP 0.8933 vs baseline 0.1317, +0.7616, 95 % CI [0.7381, 0.782], verdict *improved*, identical to the record. Details and hashes: "Amendment-2 reproduction 1: hosted T4 (2026-10-18, commit f0d021e)" below; evidence in `docs/execution-evidence/2026-10-18-f0d021e/`. Not exercised: both BYOD branches, the Section 14 activity, a browser Run all |
| 2026-10-23 (13:59:10 UTC start) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-yolox-1792c3f-c7f4`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (33 locked packages, setup 61 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`, device class `cuda-t4`; determinism flags on; default path, every form field at its default | `yolox_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `b6082b9b3385`, `module_sha256` `2148bba8f961…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `4f0199a59dcbec7b9292824b5846861d168a486e` at commit `1792c3fa79ae767460ca916aba9793a64b0dc499` | **PASS — default path of the safetensors export.** One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 149.9 s wall. Grid ON (`RUN_GRID` "auto" on a GPU), C0 / C1 / C2 trained; grid hash `184785885a81…`, equal to the amendment-2 record; C1 selected; Section 9 printed `test_look` "reproduction" and "matches the record" (fresh test 3 C1 AP 0.8933 vs baseline 0.1317, +0.7616, 95 % CI [0.7381, 0.782], *improved*). Section 11 exported `yolox-detection-adapter.safetensors` (35,895,336 bytes, 462 tensors) plus the sidecar `yolox-detection-adapter.json`, no `.pt`; fresh reload identical (0.8937 / 1.0); flipped weight byte and edited format tag both refused. Details and hashes: "Safetensors export: hosted T4 runs (2026-10-23, commit 1792c3f)" below; evidence in `docs/execution-evidence/2026-10-23-1792c3f/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-23 (14:01:51 UTC start) | Same runner, a second fresh Colab Tesla T4 VM (same session name); setup 55 s; Section 14 activity: the run copy (git blob `94b0b1b7fe0d`) differs from the committed blob only in `FREEZE_BACKBONE = False` | `yolox_detection_finetune_colab.ipynb`, run copy with one form field changed | blob `4f0199a59dcbec7b9292824b5846861d168a486e` at commit `1792c3fa79ae767460ca916aba9793a64b0dc499` | **PASS — Section 14 activity (exploratory).** One pass, no restart, 0 errors; 25/25 code cells in order; 146.5 s wall. Grid ON; all 8,938,456 parameters trained; validation AP C0 0.4197 / C1 0.8104 / C2 0.0726, C1 selected; C1 fine-tune 19.4 s (default 8.1 s). Section 9 printed "fresh test not scored: settings differ from the pre-registration: ['freeze_backbone']". Same `.safetensors` + `.json` export, reload identical (0.8104 / 1.0), both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-23-1792c3f-activity/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-24 (09:13:12 UTC start) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM, committed blob fetched at the commit and checked before the VM was allocated, input unchanged; kernel Python 3.13.16, isolated CPython 3.12.12 (33 locked packages, setup 47 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`, device class `cuda-t4`; determinism flags on; default path, every form field at its default | `yolox_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `73eef81cdbd1`, `module_sha256` `3c01d232eb4b…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `0c4430e821f5fa10a5023b62ffada64bf22e0a3f` at commit `85f48387d3eed88bc4ab76e1fb53eb6cf3fa60e6` | **PASS — default path of the safetensors serving change.** One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 167.3 s wall. Grid ON, C0 / C1 / C2 trained; grid hash `184785885a81…`, equal to the amendment-2 record; C1 selected; Section 9 printed `test_look` "reproduction" and "matches the record" (fresh test 3 C1 AP 0.8933 vs baseline 0.1317, +0.7616, 95 % CI [0.7381, 0.782], *improved*). Section 11 exported `yolox-detection-adapter.safetensors` (35,895,336 bytes, 462 tensors) plus `yolox-detection-adapter.json`; fresh reload identical (0.8937 / 1.0); both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-24-85f4838/`. |
| 2026-10-24 (09:14:52 UTC start) | Same runner, a second fresh Colab Tesla T4 VM; setup 51 s; Section 14 activity: the run copy (git blob `3b34885b62f4`) differs from the committed blob only in `FREEZE_BACKBONE = False` | `yolox_detection_finetune_colab.ipynb`, run copy with one form field changed | blob `0c4430e821f5fa10a5023b62ffada64bf22e0a3f` at commit `85f48387d3eed88bc4ab76e1fb53eb6cf3fa60e6` | **PASS — Section 14 activity (exploratory).** One pass, no restart, 0 errors; 25/25 code cells in order; 152.4 s wall. Grid ON; all 8,938,456 parameters trained; validation AP C0 0.4197 / C1 0.8104 / C2 0.0726, C1 selected; C1 fine-tune 19.4 s (default 8.2 s). Section 9 printed "fresh test not scored: settings differ from the pre-registration: ['freeze_backbone']". Same `.safetensors` + `.json` export, reload identical (0.8104 / 1.0), both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-24-85f4838-activity/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-25 (00:50:05 UTC start) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM, committed blob fetched at the commit and checked before the VM was allocated, input unchanged; kernel Python 3.13.16, isolated CPython 3.12.12 (33 locked packages, setup 43 s), `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda:0`, device class `cuda-t4`; determinism flags on; default path, every form field at its default | `yolox_detection_finetune_colab.ipynb` (`NOTEBOOK_SOURCE.repository_revision` `e165d90cc93b`, `module_sha256` `b18edddc8932…`, generator `build_notebook.py/2.2-swc-yolox`, `notebook_spec` 2.5) | blob `b564dcae6b02d39cb63c20f3530119bbe25e6d76` at commit `21fefe0e0effc9fd664038c8e54316d229394e92` (main, the merge of PR #10) | **PASS — default path of the current blob.** One pass, no restart, 0 errors; 25/25 code cells in order (14 print output); 129.3 s wall. Grid ON, C0 / C1 / C2 trained; grid hash `184785885a81…`, equal to the amendment-2 record; C1 selected; Section 9 printed `test_look` "reproduction" and "matches the record" (fresh test 3 C1 AP 0.8933 vs baseline 0.1317, +0.7616, 95 % CI [0.7381, 0.782], *improved*). Section 11 exported `yolox-detection-adapter.safetensors` (35,895,336 bytes, 462 tensors) plus `yolox-detection-adapter.json`; fresh reload identical (0.8937 / 1.0); both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-25-21fefe0/`. Not exercised: both BYOD branches, a browser Run all |
| 2026-10-25 (00:44:43 UTC start) | Same runner, a second fresh Colab Tesla T4 VM; setup 52 s; Section 14 activity: the run copy (git blob `a02d210f92ee`) differs from the committed blob only in `FREEZE_BACKBONE = False` | `yolox_detection_finetune_colab.ipynb`, run copy with one form field changed | blob `b564dcae6b02d39cb63c20f3530119bbe25e6d76` at commit `21fefe0e0effc9fd664038c8e54316d229394e92` | **PASS — Section 14 activity (exploratory).** One pass, no restart, 0 errors; 25/25 code cells in order; 151.2 s wall. Grid ON; all 8,938,456 parameters trained; validation AP C0 0.4197 / C1 0.8104 / C2 0.0726, C1 selected; C1 fine-tune 18.8 s (default 7.9 s). Section 9 printed "fresh test not scored: settings differ from the pre-registration: ['freeze_backbone']". Same `.safetensors` + `.json` export, reload identical (0.8104 / 1.0), both tamper checks refused. Evidence in `docs/execution-evidence/2026-10-25-21fefe0-activity/`. Not exercised: both BYOD branches, a browser Run all |

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

## Hosted T4 reproduction 116b (2026-10-18, commit d6794c9)

**REL13 label for every number in this section:**
- Runtime: Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-yolox-d6794c9-c7f4`), started 2026-10-18 09:22:48 UTC, 144.7 s wall.
- Code: commit `d6794c9`, notebook blob `bf60acf7`. Settings were the same as the record's: the round-2 settings, amendment 1 and every field at its default.
- Role: **reproduction**, the second hosted look at fresh test 1160. Section 9 printed `test_look` "reproduction" and wrote the `_reproduction_1` files.
- Evidence: `docs/execution-evidence/2026-10-18-d6794c9/`. The executed notebook and `run_summary.json` were copied byte for byte. In `exec.log`, lines 1 and 256 had the local run directory replaced with `<local-run-dir>/`.
  - Captured SHA-256 of `exec.log`: `c3d8ee212880f312a078b2b09b8149ff27967971d2f4862af4b8f225ba81edbe`.
  - Committed SHA-256 of `exec.log`: `e0f886eaf76044c394fff3634524e1a1f53cc07e6f569d962816a3978862a6ec`.
  - `run_summary.json`: `68ed39db…cd92b`. Executed notebook: `160e5879…610f`.

**The selection was NOT reproduced:**

| Config | Validation AP, record `ad795d2` | Validation AP, 116b `d6794c9` | Final `total_loss` record → 116b |
|---|---|---|---|
| C0 | 0.6966 | **0.8624** | 1.87603 → 1.75461 |
| C1 | **0.8554** | 0.8818 | 1.38442 → 1.42115 |
| C2 | 0.7122 | 0.7981 | 2.60175 → 1.59043 |

- 116b selected **C0**: C1 beat it by only 0.0194, which is below the 0.03 margin. The record selected C1.
- The cause is CUDA non-determinism. With the same seed, the loss curves are identical through epoch 2 and differ from epoch 3 on (C0 epoch 3: 2.8195 in the record, 2.8004 in 116b).

**The fresh-test gain was reproduced within its interval.** Section 9 printed the comparison block against the record:
- `selected_config` C1 → C0;
- `selected_ap` 0.8573 → 0.8594;
- `ap_difference` +0.7211 → **+0.7231**;
- `ci95` [0.6983, 0.7435] → [0.6999, 0.7440].

The verdict is "improved" in both runs. The baseline is identical in both: fresh AP 0.1363 / AP50 0.1449, validation 0.1767, exposed development test 0.2083 / 0.2222.

**Consequence.** Kurt ruled on 2026-10-18 that GPU training will be made deterministic and that a third fresh pool and a new evaluation of record will be added. See amendment 2. The record `ad795d2` and this reproduction stay as history.

## Amendment 2 (2026-10-18, round 3): deterministic GPU training and fresh test 3

**Status.** `docs/yolox-preregistration-amendment-2.md` was committed alone, before the code. The round-2
record (`ad795d2`) and its reproduction 116b stay valid as history. Fresh test 2 (seed 1160, digest
`6900526c80ac…`) is now an exposed test, scored by the round-2 record.

**Fresh test 3.** `sign_dataset(200, seed=1162)`: 200 images and 386 boxes (stop 131, yield 126,
speed-limit 129). SHA-256 under Pillow 11.3.0:
`2c81a3b4929a8c74811cf47dfdedf3fe724800eb1e23d5fce884034c8af3c826`. It shares 0 images and 0 box layouts with
any of these pools: training, the exposed development test, validation, fresh test 2 and seed 99. The median
nearest-neighbour MAD is 0.0249–0.0264 against the earlier pools; it is 0.035 within training. It has never
been scored.

**Determinism.**
- The carried protocol sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` when the module is imported (Section 2),
  before any CUDA work.
- Section 8 calls `enable_determinism(torch, SEED)` before training. That sets:
  - cuDNN deterministic, no benchmark;
  - `torch.use_deterministic_algorithms(True)` without `warn_only`;
  - seeded torch, numpy and random.
- Batches are built in-process: no DataLoader, 0 workers.
- The flags are a pre-registered setting and appear in every report. Section 8 prints a grid hash: SHA-256
  of the grid rows without timings, which covers the loss curves, validation AP and the weight fingerprints.
- An op without a deterministic CUDA kernel stops Section 8 with "STOP and report it".
- Static review of the trainable path found no op on PyTorch's list of ops without a deterministic CUDA
  kernel. That path is the head convs and BatchNorm, BCE and IoU losses, boolean-mask indexing (an
  `index_put_` backward, which has a deterministic kernel) and SimOTA's no-grad `topk`/`argmax`/`index_put_`.
  The neck's nearest upsample is frozen, so it has no backward. This is inferred, not run: the masked hosted
  check is the test.

**Evaluation of record under amendment 2:** the Colab CLI T4 run of 2026-10-18 at commit `4429e0e` (blob
`a9dd3990`); see "Amendment-2 evaluation of record" below. It ran after the masked determinism check passed
(two masked hosted T4 runs, identical grid hash, validation AP, selection and loss curves; see "Masked
determinism check" below). Hosted looks at fresh test 3: 2. Reproductions: 1 (commit `f0d021e`, blob `6b295979`,
identical to the record; see "Amendment-2 reproduction 1" below).
*(State at the 4429e0e revision, before the record existed:* `RECORDED_EVALUATION` is `None` and a grid-off
CPU run keeps training the round-2 selection C1. Recording the amendment-2 record in `protocol.py`
regenerates the notebook; until then a later identical hosted run labels itself "pre-registered evaluation
(this runtime)", and this file is what says it is a reproduction.)

**Note (2026-10-18, requote r3):** `RECORDED_EVALUATION` in `protocol.py` now carries this record (date,
commit `4429e0e`, notebook blob `a9dd3990`, selection C1, validation AP, grid hash and the fresh-test-3
numbers), so a later pre-registered scoring of fresh test 3 labels itself `reproduction`, including the first
one in a new runtime, writes `_reproduction_<n>` files and prints whether it matches the record.
`RECORDED_SELECTION` stays C1. The notebook was regenerated; no model was run and nothing was scored.

## Masked determinism check (amendment 2, 2026-10-18)

**REL13 label for this section:**
- Runtime: Colab CLI 0.7.4, two fresh Colab Tesla T4 VMs, one after the other, same session name
  `suite-yolox-9c5f8f2-c7f4` on different VMs (`…usw1b2-3hovuz6v6dd92`, then `…ass1c0-2zg8b0lwuaies`).
- Code: commit `9c5f8f2` on the throwaway branch `fix/yolox-tutorial-review-2026-10-17-maskcheck`, notebook blob
  `2987f610f6d6afd1d3b64f6589500d35b8ce70a9`. That blob is this PR's blob `a9dd3990` plus one first code cell
  (`maskcheck-00`) that sets `DIMER_MASK_FRESH_TEST=1`; the diff is that one 13-line cell. The branch is **never
  merged** and may be deleted after this PR merges.
- Role: **not scored (fresh test masked for a development check)**. With the mask set, Section 9 never calls
  `sign_dataset` for fresh test 3 (the generation is behind `not masked`), so fresh test 3 was **never generated**
  by either run. Neither suite log mentions seed 1162 or the 386 boxes.
- Evidence: the orchestrator's suite logs only. The two runs shared one local output directory, so the second
  run's files overwrote the first's; no evidence files are committed for them.

| Run | Started (UTC) | Suite result | Cells | Grid hash | Validation AP C0 / C1 / C2 | Selected | `test_look` | Fresh test 3 |
|---|---|---|---|---|---|---|---|---|
| 116m1 | 13:27:50 | PASS, 113 s start to release | 26/26 | `184785885a816d4becfec2789c796d112be29dbe5f5b4760f2ab7fdb39753793` | 0.7037 / 0.8937 / 0.8296 | C1 | not scored (fresh test masked …) | never generated |
| 116m2 | 13:29:49 | PASS, 115 s start to release | 26/26 | `184785885a816d4becfec2789c796d112be29dbe5f5b4760f2ab7fdb39753793` | 0.7037 / 0.8937 / 0.8296 | C1 | not scored (fresh test masked …) | never generated |

- The gate held: the two logs are identical apart from the VM name, object addresses and timings. That covers
  the validation AP to every printed digit (C0 0.7037040193718931, C1 0.8937458271675158, C2 0.8296297020369339),
  the final losses, all 24 epoch loss lines, the selection, the determinism flags and the grid hash. The grid
  hash covers the weight fingerprints, so the trained weights were identical too.
- On these T4 runs C0 and C1 share their first six epoch lines exactly, as they did on CPU. In the round-2
  record they diverged from epoch 2.

## Amendment-2 evaluation of record: hosted T4 (2026-10-18, commit 4429e0e)

**REL13 label for every number in this section:**
- Runtime: Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4
  (session `suite-yolox-4429e0e-c7f4`, VM `…usw1b2-1tw1pkiyg6tmc`), started 2026-10-18 13:40:44 UTC, 143.6 s wall.
  Section 8 printed `device Tesla T4 (cuda-t4)` and the determinism flags: `cublas_workspace_config` `:4096:8`,
  `cudnn_deterministic` True, `cudnn_benchmark` False, `deterministic_algorithms` True, `warn_only` False. TF32 is
  not printed; the Tesla T4 (Turing) has no TF32 mode (inferred from the hardware).
- Code: commit `4429e0e`, notebook blob `a9dd3990`, embedded revision `ab130f3` (`ab130f3..4429e0e` only
  regenerates the notebook), `module_sha256` `57f7ef9a…`. Settings: the round-2 settings, amendment 1, amendment 2
  and every field at its default.
- Role: **amendment-2 evaluation of record.** It is the first hosted T4 run under amendment 2 that scores fresh
  test 3 and the first scoring of fresh test 3 by any run. Section 9 printed `test_look` "pre-registered evaluation
  (this runtime)". The record ran after the masked determinism check above passed.
- Evidence: `docs/execution-evidence/2026-10-18-4429e0e/`. The executed notebook and `run_summary.json` were
  copied byte for byte. In `exec.log`, lines 1 and 256 had the local run directory replaced with
  `<local-run-dir>/`; nothing else changed.
  - Captured SHA-256 of `exec.log`: `e5f3a052d7095a71ab1150f4a92bf7b74a78c98b00d92cce16c83795ee3e737c`. This is
    the value of the file in the run directory; `run_summary.json` records only the executed notebook's hash.
  - Committed SHA-256 of `exec.log`: `61193c21a7175836cb2acb469cd303b3c5ca96619ed7bd6a2a265dff00419462`.
  - `run_summary.json`: `027ad148b084e3b50f2ecceedf840c9780801488606850ffda2a5ff944d45331`.
  - Executed notebook: `7b65221312996341d999b55a33a423283a4ac4b5d01bde0119dd25a6a78c3a58` (equal to
    `run_summary.json` `executed_sha256`).
  - The input copy in the run directory is byte-equal to the committed notebook (git blob `a9dd3990`).
  - `inspect_run.py`: 56/56 cells, same ids and order, source identity OK.

**Grid (pre-registered; decided here).** Identical to both masked checks: grid hash `184785885a81…`.

| Config | Validation AP@[.50:.95] | AP50 | Gain over C0 | Final `total_loss` | Fine-tune s (T4) | Round-2 record `ad795d2` AP |
|---|---|---|---|---|---|---|
| C0 (6 ep, 1e-3) | 0.7037 | 1.0 | — | 1.72345 | 7.0 | 0.6966 |
| C1 (10 ep, 1e-3) | **0.8937** | 1.0 | +0.1900 | 1.52160 | 8.2 | 0.8554 |
| C2 (8 ep, 2e-3) | 0.8296 | 1.0 | +0.1259 | 1.69447 | 6.7 | 0.7122 |

- Both challengers qualify (gain ≥ 0.03, AP50 equal). The larger gain wins, so C1 is selected, as in the round-2
  record. Keep-baseline: 0.8937 − 0.1767 = 0.7170 ≥ 0.03, so the adapted C1 is the selected system.
- Grid fine-tuning totalled 21.9 s. Per-class validation AP50 is not printed per config; a mean AP50 of 1.0
  implies 1.0 for every class (arithmetic).
- Validation baseline: AP 0.1767 / AP50 0.1851; per-class AP50 stop 0.0554, yield 0.0, speed-limit 0.5. Equal to
  every earlier record.

**Fresh test 3** (`sign_dataset(200, seed=1162)`, 200 images, 386 boxes, digest-checked before scoring):

| System | AP@[.50:.95] | AP50 | AP50 stop / yield / speed-limit |
|---|---|---|---|
| Selected C1 | 0.8933 | 1.0 | 1.0 / 1.0 / 1.0 |
| Re-headed baseline | 0.1317 | 0.1388 | 0.0219 / 0.0 / 0.3944 |

- Difference +0.7616, paired image-bootstrap 95 % CI [0.7381, 0.782] (n_boot 10000, fixed seed). The lower bound
  is > 0 and the gain is ≥ 0.03, so the verdict is *improved*. The observed half-width is 0.022.
- At IoU 0.5 every box was found on 200 of 200 images by C1 and on 35 of 200 by the baseline.
- The baseline's classification head is untrained (six `head.cls_preds` tensors could not transfer, printed in
  Section 7), so a large gain was expected. All pools come from the same drawing generator, so this measures
  in-distribution generalisation on drawn signs, not detection on real images.

**Exposed tests (decide nothing).**
- Exposed development test (the old 10 images, 21 boxes): baseline AP 0.2083 / AP50 0.2222 (per-class AP50 0.0 /
  0.0 / 0.6667) → C1 0.9043 / 1.0, change +0.6960 / +0.7778.
- Fresh test 2 (seed 1160, exposed, scored by the round-2 record): **not scored by this run**. Its hosted numbers
  stay those of the round-2 record (+0.7211) and reproduction 116b (+0.7231). Fresh test 2 and fresh test 3 are
  different pools, so +0.7616 is not a reproduction of +0.7211.

**Other checks (this run).** COCO scene 4/5 matched (IoU 0.946 / 0.914 / 0.946 / 0.000 / 0.959); RGB order gives
5 detections, stop sign 0.839; blank and noise 0 boxes at the demo thresholds (noise 1 at the evaluation
thresholds, bear 0.073). New-data images: every printed truth label detected, same-label IoU 0.891–0.955.
Artifact 36,000,287 bytes over 462 tensors, SHA-256 `253dd2459f8a…`; the fresh reload reproduces validation AP
0.8937 / AP50 1.0 exactly; the tampered artifact is refused. BYOD off.

**Against the round-2 record.** Same selection (C1); every validation AP is higher (C0 +0.0071, C1 +0.0383,
C2 +0.1174), and the exposed C1 AP rises from 0.8516 to 0.9043. The two runs differ in code (determinism flags,
amendment 2), so the gaps are not attributed to non-determinism alone. The round-2 record `ad795d2` and
reproduction 116b stay as history.

## Amendment-2 reproduction 1: hosted T4 (2026-10-18, commit f0d021e)

**REL13 label for every number in this section:**
- Runtime: Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4
  (session `suite-yolox-f0d021e-c7f4`, VM `…euw4c2-u6t6kjzt05tr`), started 2026-10-18 14:03:56 UTC, 151.8 s wall,
  environment build 55 s. Section 8 printed `device Tesla T4 (cuda-t4)` and the same determinism flags as the record.
- Code: commit `f0d021e`, notebook blob `6b295979`, embedded revision `b74cdd9` (`b74cdd9..f0d021e` only
  regenerates the notebook), `module_sha256` `27393790…`. `4429e0e..f0d021e` carries the record into
  `RECORDED_EVALUATION` and requotes the worked answers; training, scoring and the pools are unchanged. Settings:
  the round-2 settings, amendment 1, amendment 2 and every field at its default.
- Role: **reproduction of the pre-registered evaluation** (reproduction 1 of the amendment-2 record `4429e0e`; the
  second hosted look at fresh test 3). Section 9 printed `test_look` "reproduction" and "reproduction of the
  pre-registered evaluation of 2026-10-18 (commit 4429e0e): matches the record". It wrote
  `yolox_detection_evaluation_report_reproduction_1.json`, `yolox_detection_fresh_test_per_image_reproduction_1.json`
  and (Section 11) `yolox_detection_results_reproduction_1.json`; no unsuffixed report, per-image or results file.
  The adapter keeps its usual name `yolox-detection-adapter.pt` (byte-identical to the record's, SHA-256
  `253dd2459f8a…`). Nothing was changed in response to the record's test number.
- Evidence: `docs/execution-evidence/2026-10-18-f0d021e/`. The executed notebook and `run_summary.json` were
  copied byte for byte. In `exec.log`, lines 1 and 257 had the local run directory replaced with
  `<local-run-dir>/`; nothing else changed.
  - Captured SHA-256 of `exec.log`: `91931c4c55ebc486c90ccac1043f897734a12a30ea8e8becc071f9ff274dce49`.
  - Committed SHA-256 of `exec.log`: `8f75da47b1324acb27e889326b7c529c773e0f775fb221ca175e89ea5bf6d879`.
  - `run_summary.json`: `129b02e1e0aa202b43518a06b05fae61fc01b32d2860ac40e4b147fe799027b1`.
  - Executed notebook: `3f7ff94663d9448fe3575fc4eb6ba62ee1b0c6c7c9af3cfc68ce9936be719650` (equal to
    `run_summary.json` `executed_sha256`).
  - The input copy in the run directory is byte-equal to the committed notebook (git blob `6b295979`).
  - `inspect_run.py`: 56/56 cells, same ids and order, source identity OK.

**Match against the record.** A line diff of the two `exec.log` files (commit ids, VM name, object addresses
and temporary paths normalised) differs only in: environment build 52 → 55 s, worker pid, the embedded revision,
C0 fine-tune 7.0 → 6.9 s, the `test_look` label, the `_reproduction_1` file names, the new "matches the record"
line and the results-file size (27,763 → 27,750 bytes, the shorter label). Everything else is identical:

| Check | Record `4429e0e` | Reproduction `f0d021e` |
|---|---|---|
| Grid hash | `184785885a816d4b…` | `184785885a816d4b…` (equal) |
| Validation AP C0 / C1 / C2 | 0.7037 / 0.8937 / 0.8296 | equal (full precision) |
| Final `total_loss` C0 / C1 / C2 | 1.72345 / 1.52160 / 1.69447 | equal |
| Epoch loss lines (24) | — | all 24 identical |
| Selection | C1 (+0.1900; keep-baseline 0.7170) | C1, same reason |
| Fresh test 3 C1 / baseline AP | 0.8933 / 0.1317 | 0.8933 / 0.1317 |
| Difference, 95 % CI | +0.7616 [0.7381, 0.782] | +0.7616 [0.7381, 0.782] |
| Verdict | *improved* | *improved* |
| Exposed development test C1 AP / AP50 | 0.9043 / 1.0 | 0.9043 / 1.0 |
| Artifact | 36,000,287 B, 462 tensors, `253dd245…` | equal; reload 0.8937 / 1.0 |

`compare_with_record` checks the selection and the fresh-test fields only; the grid hash and loss curves were
compared from the logs. Determinism therefore held on a fourth T4 VM. Status stays **Candidate**.

## Safetensors artifact export (2026-10-23)

Code commit `b6082b9` changes the exported adapter from one `torch.save` pickle (`yolox-detection-adapter.pt`, format
`dimer-yolox-detection-adapter/1`) to format `/2`: the state dict as `yolox-detection-adapter.safetensors` (written with
`safetensors.torch.save_file`, no pickle) and a plain-JSON sidecar `yolox-detection-adapter.json` holding the metadata and
the SHA-256 of the weights file. `load_artifact` checks the sidecar's format, identity, model key and that
SHA-256 before it opens the weights, then applies them with `load_state_dict(strict=True)`; `.pt` artifacts
are refused. Section 11's tamper check now flips one weight byte and edits the sidecar's format tag, and both
are refused. `safetensors==0.8.0` joins the pins and the hash lock.

The regenerated notebook `yolox_detection_finetune_colab.ipynb` is blob `4f0199a59dcbec7b9292824b5846861d168a486e`. **REL14:** every row under
"Recorded executions" ran an earlier blob; those rows, and the artifact sizes and digests quoted above, are
history for the `.pt` export and are not evidence for this blob. The hosted T4 runs of this blob are recorded
in the next section. Unit tests, the validator and the parity check pass on CPU; that is static evidence only.

## Safetensors export: hosted T4 runs (2026-10-23, commit 1792c3f)

**REL13 label for every number in this section:**
- Runtime: Colab CLI sequential execution (`colab exec -f`, not a browser Run all), session
  `suite-yolox-1792c3f-c7f4`, two fresh Colab Tesla T4 VMs (`…usw4a1-1j7yaycfft303` for the default run,
  `…euw4b2-2yf75ad5s3b5o` for the activity run). Kernel Python 3.13.15, isolated CPython 3.12.12, 33 locked
  packages (`safetensors==0.8.0` is the new one), `torch 2.14.0+cu130`, `cuda:0`, device class `cuda-t4`,
  determinism flags on.
- Code: commit `1792c3f`, notebook blob `4f0199a5`, embedded revision `b6082b9` (`b6082b9..1792c3f` regenerates the
  notebook and adds the `safetensors` pin to the model card), `module_sha256` `2148bba8…`.
- Roles: the default run is every field at its default. The activity run is Section 14 ("change one thing —
  unfreeze the backbone"): its run copy differs from the committed blob only in `FREEZE_BACKBONE = False`, so it
  is **exploratory** and Section 9 did not score the fresh test.

**Why each run took about 150 s.** `RUN_GRID` is "auto", which turns the grid on when CUDA is available, so both
runs trained all three configs; nothing was skipped. Section 8 printed `grid ON: training ['C0', 'C1', 'C2']` in
both. The time splits as follows (all printed by the runs):

| | Default | Activity |
|---|---|---|
| Isolated environment build (Section 1) | 61 s | 55 s |
| Fine-tune C0 / C1 / C2 | 6.8 / 8.1 / 6.7 s | 13.8 / 19.4 / 15.5 s |
| Fresh test 3 (200 images, baseline and C1, paired bootstrap) | scored | not scored (settings differ) |
| Wall, whole notebook | 149.9 s | 146.5 s |

The remainder is model staging (72,089,125 bytes, digest-verified), the COCO demonstration, the baseline and
validation scoring, export and reload. With the backbone frozen, 1,891,096 of 8,938,456 parameters train, which is
why the default grid takes about 22 s on a T4.

**Default run.**
- Section 8: grid hash `184785885a816d4becfec2789c796d112be29dbe5f5b4760f2ab7fdb39753793`, equal to the amendment-2
  record `4429e0e` and reproduction 1 `f0d021e`. Validation AP C0 0.7037 / C1 0.8937 / C2 0.8296 (AP50 1.0 each),
  final `total_loss` 1.72345 / 1.5216 / 1.69447. C1 selected ("C1 beat C0 by 0.1900 validation AP"); it beats the
  baseline's 0.1767 by 0.7170, so the adapted model is the selected system.
- Section 9: `test_look` "reproduction"; fresh test 3 (200 images, 386 boxes) C1 AP 0.8933 / AP50 1.0 against the
  baseline's 0.1317 / 0.1388, difference +0.7616, 95 % CI [0.7381, 0.782], verdict *improved*; printed
  "reproduction of the pre-registered evaluation of 2026-10-18 (commit 4429e0e): matches the record". Exposed
  development test C1 AP 0.9043 / AP50 1.0 (baseline 0.2083 / 0.2222). The safetensors change touches export only, so
  these match the record as expected. The suffix counter is per runtime, so this run again wrote
  `_reproduction_1` files.
- Section 11 printed, verbatim:
  - `"path": "outputs/yolox-detection-adapter.safetensors"`, `"bytes": 35895336`,
    `"sha256": "770d0218dfd9e6bff55450e3ad99f3ad7fce50ad230f22e7861dcce4e61116cc"`,
    `"sidecar": "outputs/yolox-detection-adapter.json"`,
    `"sidecar_sha256": "2d08d512fc9119e360f0bf50fa80d7ace47eb166f7e570c6c09ab7b7fb907daa"`,
    `"format": "dimer-yolox-detection-adapter/2"`, `"tensors": 462`;
  - `{'source': 'artifact:yolox-detection-adapter.safetensors', ...}` and
    `{'reloaded_val_ap': 0.8937, 'reloaded_val_ap50': 1.0}`, then "fresh reload reproduces the validation scores of
    the selected system (C1) exactly";
  - `tampered artifact (one weight byte flipped) refused -> artifact weights digest mismatch for
    'tampered-artifact.safetensors': sidecar records 770d0218…16cc, file has a51c9ecd…fee19`;
  - `tampered artifact (format tag edited) refused -> artifact format 'not-a-dimer-artifact/9' !=
    'dimer-yolox-detection-adapter/2'`.
- Section 12 listed `outputs/`: `grid`, `work`, `yolox-detection-adapter.json`, `yolox-detection-adapter.safetensors`
  and the three `_reproduction_1` JSON files. No `.pt` file was written.
- Other checks: COCO scene 4/5 matched (IoU 0.946 / 0.914 / 0.946 / 0.000 / 0.959); RGB order 5 detections, stop
  sign 0.839; blank and noise 0 boxes at the demo thresholds. New-data images: every truth label detected,
  same-label IoU 0.891–0.955. BYOD off.

**Activity run (`FREEZE_BACKBONE = False`).**
- Section 8: all 8,938,456 parameters trained. Validation AP C0 0.4197 / AP50 0.9137, C1 0.8104 / 1.0,
  C2 0.0726 / 0.1583; grid hash `dd489ab92ad1…`. C1 selected ("C1 beat C0 by 0.3907 validation AP"); it beats the
  baseline by 0.6337.
- So, against the default: the selected C1's validation AP fell from 0.8937 to 0.8104 and its fine-tune took 19.4 s
  instead of 8.1 s; C2 fell to 0.0726.
- Section 9 printed `test_look` "not scored (exploratory)" and "fresh test not scored: settings differ from the
  pre-registration: ['freeze_backbone']"; it wrote `_exploratory_1` files. Exposed development test C1 AP 0.8286 /
  AP50 1.0 (descriptive only).
- Section 11: `yolox-detection-adapter.safetensors` (35,895,336 bytes, 462 tensors, SHA-256 `c54c27f75646…`) plus
  `yolox-detection-adapter.json` (SHA-256 `39f0ae25b60e…`), format `dimer-yolox-detection-adapter/2`; reload 0.8104 /
  1.0, "fresh reload reproduces the validation scores of the selected system (C1) exactly"; the flipped weight byte
  was refused with "artifact weights digest mismatch" and the edited format tag with "artifact format
  'not-a-dimer-artifact/9' != 'dimer-yolox-detection-adapter/2'". Section 12 listed no `.pt` file.

**Evidence.** The executed notebooks, `run_summary.json` and `run_copy.ipynb` were copied byte for byte; each
executed notebook was renamed (`_output` dropped, run label added). In each `exec.log`, the two lines that named the
local run directory (default: lines 1 and 260; activity: lines 1 and 229) had it replaced with `<local-run-dir>/`;
nothing else changed. The input copy of the default run is byte-equal to the committed notebook (SHA-256
`52c87be432b0…`, git blob `4f0199a5`).

| File | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `2026-10-23-1792c3f/yolox_detection_finetune_colab_1792c3f_colab-cli-t4_default.ipynb` | `2d1deb269dbf7fb262c71e81520fe1dac140f9826ce7498cfd818f6b1f5ecf84` (= `executed_sha256`) | same |
| `2026-10-23-1792c3f/run_summary.json` | `308bcf7b40ac707690ff0f254a3ba0372045e748486f7c8e84bb5396a1dc8652` | same |
| `2026-10-23-1792c3f/exec.log` | `49dd94d548307280577f9fc0d63a00fd4fa3e7b7c0fb3b3f9e7b0c46e67babab` | `4b892a5060928f2bebfcd3208d755291519e409f5896efc1d550fec2369adb3a` |
| `2026-10-23-1792c3f-activity/yolox_detection_finetune_colab_1792c3f_colab-cli-t4_activity.ipynb` | `352f9f909df5e8d9781d6e87785b23d6d32014df4ba7a8d59b7b9327f056e664` (= `executed_sha256`) | same |
| `2026-10-23-1792c3f-activity/run_copy.ipynb` | `33b16389a6cb973d385a5659e262014303ecc212dd764b34a7e46594f4f74a7b` (= `run_copy_sha256`, git blob `94b0b1b7`) | same |
| `2026-10-23-1792c3f-activity/run_summary.json` | `a99d0524e932676d911beb35c82886261012849a0cd7f971d19e16ff13f8219b` | same |
| `2026-10-23-1792c3f-activity/exec.log` | `4917d2f8328ab6fe4f1fdc2987a155d0793a8df11ec73d35aebd28182d494e3a` | `71d6775e9bcc10613b7a4e90d8473e83eaf9fb94c1b74492fb82a6ee92290b83` |

## Safetensors serving: hosted T4 runs (2026-10-24, commit 85f4838)

**REL13 label for every number in this section:**
- Runtime: Colab CLI sequential execution (`colab exec -f`, not a browser Run all), two fresh Colab Tesla T4 VMs.
  Kernel Python 3.13.16, isolated CPython 3.12.12, 33 locked packages, `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`,
  `cuda:0`, device class `cuda-t4`, determinism flags on.
- Code: commit `85f4838`, notebook blob `0c4430e8`, embedded revision `73eef81` (the safetensors serving change;
  `73eef81..85f4838` regenerates the notebook and records its status), `module_sha256` `3c01d232…`.
- Roles: the default run is every field at its default. The activity run is Section 14 ("change one thing —
  unfreeze the backbone"): its run copy differs from the committed blob only in `FREEZE_BACKBONE = False`, so it is
  **exploratory** and Section 9 did not score the fresh test.

**Model staging.** Both runs printed `{'fetched': ['yolox_s.pth']}`, `{'verified_files': 1, 'revision': '0.1.1rc0'}`
(72,089,125 bytes) and `{'device': 'cuda:0', 'source': 'local-snapshot'}`. The output does not print the safetensors
conversion or its digest; it shows the staged `.pth`, the verification and the load.

| | Default | Activity |
|---|---|---|
| Isolated environment build (Section 1) | 47 s | 51 s |
| Fine-tune C0 / C1 / C2 | 6.9 / 8.2 / 6.7 s | 13.7 / 19.4 / 15.5 s |
| Fresh test 3 (200 images, baseline and C1, paired bootstrap) | scored | not scored (settings differ) |
| Wall, whole notebook | 167.3 s | 152.4 s |

**Default run.**
- Section 8: grid hash `184785885a816d4becfec2789c796d112be29dbe5f5b4760f2ab7fdb39753793`, equal to the amendment-2
  record `4429e0e`, reproduction 1 `f0d021e` and the 2026-10-23 default run. Validation AP C0 0.7037 / C1 0.8937 /
  C2 0.8296 (AP50 1.0 each), final `total_loss` 1.72345 / 1.5216 / 1.69447. C1 selected ("C1 beat C0 by 0.1900
  validation AP"); it beats the baseline's 0.1767 by 0.7170, so the adapted model is the selected system.
  1,891,096 parameters trained; C1 `total_loss` 6.2880 -> 1.5216 over 10 epochs.
- Section 9: `test_look` "reproduction"; fresh test 3 (200 images, 386 boxes) C1 AP 0.8933 / AP50 1.0 against the
  baseline's 0.1317 / 0.1388, `ap_difference` 0.7616, `ci95` [0.7381, 0.782], verdict *improved*; printed
  "reproduction of the pre-registered evaluation of 2026-10-18 (commit 4429e0e): matches the record". Exposed
  development test C1 AP 0.9043 / AP50 1.0 (baseline 0.2083 / 0.2222).
- Section 11 printed `"path": "outputs/yolox-detection-adapter.safetensors"`, `"bytes": 35895336`,
  `"sha256": "770d0218dfd9e6bff55450e3ad99f3ad7fce50ad230f22e7861dcce4e61116cc"`,
  `"sidecar_sha256": "2d08d512fc9119e360f0bf50fa80d7ace47eb166f7e570c6c09ab7b7fb907daa"`,
  `"format": "dimer-yolox-detection-adapter/2"`, `"tensors": 462` (the same digests as the 2026-10-23 default run);
  `{'reloaded_val_ap': 0.8937, 'reloaded_val_ap50': 1.0}` and "fresh reload reproduces the validation scores of the
  selected system (C1) exactly"; the flipped weight byte was refused ("artifact weights digest mismatch") and the
  edited format tag was refused ("artifact format 'not-a-dimer-artifact/9' != 'dimer-yolox-detection-adapter/2'").
- Section 12 listed `outputs/`: `grid`, `work`, `yolox-detection-adapter.json`, `yolox-detection-adapter.safetensors`
  and the three `_reproduction_1` JSON files.
- Other checks: COCO scene 4/5 matched (IoU 0.946 / 0.914 / 0.946 / 0.000 / 0.959); RGB order 5 detections, stop
  sign 0.839; blank 0 boxes at both thresholds, noise 0 at the demo thresholds and 1 (`bear` 0.073) at the evaluation
  thresholds. New-data images: every truth label detected, same-label IoU 0.891–0.955. BYOD off; both refusal probes
  fired.

**Activity run (`FREEZE_BACKBONE = False`).**
- Section 8: all 8,938,456 parameters trained. Validation AP C0 0.4197 / AP50 0.9137, C1 0.8104 / 1.0,
  C2 0.0726 / 0.1583; grid hash `dd489ab92ad1…` (equal to the 2026-10-23 activity run). C1 selected ("C1 beat C0 by
  0.3907 validation AP"); it beats the baseline by 0.6337.
- Against the default: the selected C1's validation AP fell from 0.8937 to 0.8104 and its fine-tune took 19.4 s
  instead of 8.2 s; C2 fell to 0.0726.
- Section 9 printed `test_look` "not scored (exploratory)" and "fresh test not scored: settings differ from the
  pre-registration: ['freeze_backbone']"; it wrote `_exploratory_1` files. Exposed development test C1 AP 0.8286 /
  AP50 1.0 (descriptive only).
- Section 11: `yolox-detection-adapter.safetensors` (35,895,336 bytes, 462 tensors, SHA-256 `bfb9770b9150…`) plus
  `yolox-detection-adapter.json` (SHA-256 `e0f6a71f6d50…`), format `dimer-yolox-detection-adapter/2`; reload 0.8104 /
  1.0, "fresh reload reproduces the validation scores of the selected system (C1) exactly"; both tamper checks refused.

**Evidence.** The executed notebooks, `run_summary.json` and `run_copy.ipynb` were copied byte for byte; each
executed notebook was renamed (`_output` dropped, run label added). In each `exec.log`, the two lines that named the
local run directory had it replaced with `<local-run-dir>/`; nothing else changed (byte-wise, no re-serialisation).
The activity run copy differs from the committed blob in exactly one line, `FREEZE_BACKBONE = True` -> `False`.

| File | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `2026-10-24-85f4838/yolox_detection_finetune_colab_85f4838_colab-cli-t4_default.ipynb` | `51348353be2de56b3e0a042ab9afbbf4b1fe22d2b227782ca03879a4430312f3` (= `executed_sha256`) | same |
| `2026-10-24-85f4838/exec.log` | `a186b11840ee3bc4d48959bd6bd3a0b9fafb0e8a38df258101963c925c3c6bdf` | `7dc675210c902925d1c11874ce382d2284190dc1a09a878a255660011676021a` |
| `2026-10-24-85f4838/run_summary.json` | `b2591119b4f32dff5524cd1078aa2745d0be0faeb9373236d90292f3ea813497` | same |
| `2026-10-24-85f4838-activity/yolox_detection_finetune_colab_85f4838_colab-cli-t4_activity.ipynb` | `212dc06bcf549772f77c88710b6be63ac86b3b54ee2d33e8bddf1a34c9e859fe` (= `executed_sha256`) | same |
| `2026-10-24-85f4838-activity/exec.log` | `ef82752f6f58a01325213243b0872a96bde6d7a6157c24ab181ef97e8a4a1432` | `ae69e73b587395b015fe39c8ab880d68c90fcfc5ab6e87c2dc6effb333330624` |
| `2026-10-24-85f4838-activity/run_summary.json` | `a911f704351cac49a4a2d69d452d9283148e9858f1f860b91d883a3b0a293a45` | same |
| `2026-10-24-85f4838-activity/run_copy.ipynb` | `7d048e6747e0bf578a00647018e0369692babfec8ea534bf15fe3d7c4e69d4b6` (= `run_copy_sha256`, git blob `3b34885b`) | same |

## Hosted T4 runs of the current blob (2026-10-25, commit 21fefe0)

**REL13 label for every number in this section:**
- Runtime: Colab CLI sequential execution (`colab exec -f`, not a browser Run all), two fresh Colab Tesla T4 VMs.
  Kernel Python 3.13.16, isolated CPython 3.12.12, 33 locked packages, `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`,
  `numpy 2.5.3`, `cuda:0`, device class `cuda-t4`, determinism flags on.
- Code: commit `21fefe0` (main, the merge of PR #10), notebook blob `b564dcae`, embedded revision `e165d90` (PR #10's
  merge of the safetensors serving change), `module_sha256` `b18edddc…`. Each `run_summary.json` records `verdict`
  `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true.
- Roles: the default run is every field at its default. The activity run is Section 14 ("change one thing —
  unfreeze the backbone"): its run copy differs from the committed blob only in `FREEZE_BACKBONE = False`, so it is
  **exploratory** and Section 9 did not score the fresh test.

| | Default | Activity |
|---|---|---|
| Isolated environment build (Section 1) | 43 s | 52 s |
| Fine-tune C0 / C1 / C2 | 6.7 / 7.9 / 6.4 s | 13.2 / 18.8 / 15.0 s |
| Fresh test 3 (200 images, baseline and C1, paired bootstrap) | scored | not scored (settings differ) |
| Wall, whole notebook | 129.3 s | 151.2 s |

**Default run.**
- Section 8: grid hash `184785885a816d4becfec2789c796d112be29dbe5f5b4760f2ab7fdb39753793`, equal to the amendment-2
  record `4429e0e` and the 2026-10-23 and 2026-10-24 default runs. Validation AP C0 0.7037 / C1 0.8937 / C2 0.8296
  (AP50 1.0 each), final `total_loss` 1.72345 / 1.5216 / 1.69447. C1 selected ("C1 beat C0 by 0.1900 validation AP");
  "C1 beats the pretrained baseline by 0.7170 validation AP (>= 0.03): the adapted model is the selected system".
  1,891,096 parameters trained; C1 `total_loss` 6.2880 -> 1.5216 over 10 epochs.
- Section 9: `test_look` "reproduction"; fresh test 3 (200 images, 386 boxes) C1 AP 0.8933 / AP50 1.0 against the
  baseline's 0.1317 / 0.1388, `ap_difference` 0.7616, `ci95` [0.7381, 0.782], verdict *improved*; printed
  "reproduction of the pre-registered evaluation of 2026-10-18 (commit 4429e0e): matches the record". Exposed
  development test C1 AP 0.9043 / AP50 1.0 (baseline 0.2083 / 0.2222).
- Section 11 printed `"path": "outputs/yolox-detection-adapter.safetensors"`, `"bytes": 35895336`,
  `"sha256": "c620c3a7a81b2d28096b5ed4a827b13972b50b51c8edf566c9bccff00864115d"`,
  `"sidecar_sha256": "43b9a92a6db01692595f902aeaf4838e021d50a948c565d9e0afcf7b6121214d"`,
  `"format": "dimer-yolox-detection-adapter/2"`, `"tensors": 462`; `{'source': 'artifact:yolox-detection-adapter.safetensors', ..., 'adapted': True}`,
  `{'reloaded_val_ap': 0.8937, 'reloaded_val_ap50': 1.0}` and "fresh reload reproduces the validation scores of the
  selected system (C1) exactly"; the flipped weight byte was refused ("artifact weights digest mismatch") and the
  edited format tag was refused ("artifact format 'not-a-dimer-artifact/9' != 'dimer-yolox-detection-adapter/2'").
- Other checks: COCO scene 4/5 matched (IoU 0.946 / 0.914 / 0.946 / 0.000 / 0.959); RGB order 5 detections, stop
  sign 0.839; blank 0 boxes at both thresholds, noise 0 at the demo thresholds and 1 (`bear` 0.073) at the evaluation
  thresholds. New-data images: every truth label detected, same-label IoU 0.891–0.955. BYOD off; both refusal probes
  fired.

**Activity run (`FREEZE_BACKBONE = False`).**
- Section 8: all 8,938,456 parameters trained. Validation AP C0 0.4197 / AP50 0.9137, C1 0.8104 / 1.0,
  C2 0.0726 / 0.1583; grid hash `dd489ab92ad1…` (equal to the 2026-10-23 and 2026-10-24 activity runs). C1 selected
  ("C1 beat C0 by 0.3907 validation AP"); it beats the baseline by 0.6337. C1 `total_loss` 7.3279 -> 1.7841.
- Section 9 printed `test_look` "not scored (exploratory)" and "fresh test not scored: settings differ from the
  pre-registration: ['freeze_backbone']". Exposed development test C1 AP 0.8286 / AP50 1.0 (descriptive only).
- Section 11: `yolox-detection-adapter.safetensors` (35,895,336 bytes, 462 tensors, SHA-256 `9631c0ae1b6e…`) plus
  `yolox-detection-adapter.json` (SHA-256 `93f96e443917…`); reload 0.8104 / 1.0; both tamper checks refused.

**Differences from the previous hosted runs (2026-10-24, blob `0c4430e8`, commit `85f4838`).** Every printed metric,
grid hash, selection line, fresh-test figure and reload score above equals the 2026-10-24 run of the same role. What
differs: the adapter file digests (default `c620c3a7…` / sidecar `43b9a92a…`, then `770d0218…` / `2d08d512…`;
activity `9631c0ae…` / `93f96e44…`, then `bfb9770b…` / `e0f6a71f…`) at the same size and tensor count, which fits
PR #10's change to the canonical safetensors header (not checked tensor by tensor); the embedded revision and
`module_sha256`; and timing (default fine-tune C1 7.9 s, then 8.2 s; wall 129.3 s and 151.2 s, then 167.3 s and
152.4 s).

**Evidence.** The executed notebooks, `run_summary.json` and `run_copy.ipynb` were copied byte for byte; each
executed notebook was renamed (`_output` dropped, run label added). In each `exec.log`, the two lines that named the
local run directory had it replaced with `<local-run-dir>/`; nothing else changed (byte-wise, no re-serialisation).
The activity run copy differs from the committed blob in exactly one line, `FREEZE_BACKBONE = True` -> `False`.

| File | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `2026-10-25-21fefe0/yolox_detection_finetune_colab_21fefe0_colab-cli-t4_default.ipynb` | `848271c9fe7a6fac3c14fbaa2c78d1a79bcaa072d6ac6b50a94994e40bd29ac5` (= `executed_sha256`) | same |
| `2026-10-25-21fefe0/exec.log` | `41c161d77cb19a2f00f30394cd22fdc5b4d16ea01470b1904b53638cfc0c0ad0` | `c657e083ec390e17d98d42d9214da560cfe81176877647f3814e33f2edda4e7e` |
| `2026-10-25-21fefe0/run_summary.json` | `0d85a75d1baedd5ec38dd4c5d85d319db0a1386d1e442034707c41074a153985` | same |
| `2026-10-25-21fefe0-activity/yolox_detection_finetune_colab_21fefe0_colab-cli-t4_activity.ipynb` | `b4302abddda21d9eae8974efb23c4dd0b6b276368a6569d20952022a113d11cc` (= `executed_sha256`) | same |
| `2026-10-25-21fefe0-activity/exec.log` | `159a3ab0ae5f6539ef144d980766ff78145efb5161ee854ed5a1c05495785e81` | `aa173af4356ecb38d9a6426019f57b78c2fc1d7ae25d182bd75be4ca6dda962e` |
| `2026-10-25-21fefe0-activity/run_summary.json` | `f907b05a4409c75a42832b2db97fce6bf4dc2b6e310c0f6267730c83e4d1ebcb` | same |
| `2026-10-25-21fefe0-activity/run_copy.ipynb` | `617fe7b0feb2f5a44704f1b1ff37895ef68fee82ce83a1056a5d6c551914e1a4` (= `run_copy_sha256`, git blob `a02d210f`) | same |

## Current status

Notebook blob `b564dcae6b02` at commit `21fefe0` (main after PR #10 and #11): the default path and the Section 14
activity are covered by the 2026-10-25 hosted T4 runs above. Status is **Candidate**, pending only the REL12 BYOD
journey and the REL2 browser Run all. Under REL14, every hosted run of an earlier blob listed under "Recorded
executions", including the 2026-10-24 runs of blob `0c4430e821f5`, is history for this blob.
