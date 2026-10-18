# Pre-registration: YOLOX-S tutorial validation grid and fresh test (2026-10-18)

This document is committed alone, before any code that implements it. It fixes how the tutorial
`tutorials/yolox_detection_finetune_colab.ipynb` chooses its fine-tuning configuration and how that
choice is evaluated. It answers review findings R116-M1, R116-M2 and R116-m6 (row 116 of the
2026-10-15 tutorial triage) and follows the shared YOLOX design for rows 116 and 117 and the
fleet-wide fresh-test score-once rule.

## Why this exists

The defaults `epochs=6`, `learning_rate=1e-3`, frozen backbone were chosen in September 2026 by a
grid scored on held-out AP50 (`docs/release-verification.md`, "The fine-tuning grid behind the
defaults"). Four of today's ten held-out images were in that grid's held-out set, and three were
grid training images. That 10-image split is therefore no longer an unbiased test. From now on it is
called the **exposed development test**: it is reported for continuity and decides nothing.

## Data (all drawn in code by the carried `samples.sign_dataset`)

| Pool | Definition | Role |
|---|---|---|
| Training | `sign_dataset(40, seed=0)`, `split_records(holdout=0.25, seed=0)` training part (30 images) | fine-tuning |
| Exposed development test | the same split's held-out part (10 images, 21 boxes) | descriptive only |
| Validation | `sign_dataset(40, seed=1161)` | grid selection and the keep-baseline rule |
| Fresh test | `sign_dataset(200, seed=1160)` | the primary endpoint, scored only under these settings |

- Every validation and fresh-test image is pinned by SHA-256 over its decoded pixels, computed under
  the pinned Pillow 11.3.0 in the notebook's isolated environment. The code commit carries the
  per-pool manifest digests. At run time a mismatch is a friendly refusal and the run becomes
  exploratory; it is not a look.
- The pools must be disjoint from each other and from seed 0 (training, exposed development test)
  and seed 99 (Section 10's new-data images), by image digest and by box layout. Near-duplicate
  distances (nearest-neighbour mean absolute pixel difference) are reported.
- Disclosure: the pools are disjoint as images, not as sources. Every pool is the same renderer
  drawing the same three shapes, so the fresh test measures **in-distribution generalisation on drawn
  signs**, nothing about photographs.
- Pretraining overlap (DAT9): every image is rendered at run time, so none can be in COCO train2017,
  the training set of `yolox_s.pth`.
- No claim compares YOLOX-S (this repository) with YOLOX-X (row 117).

## Fixed settings

Every one of these enters the run-time `preregistered` check. Changing any of them makes a run
exploratory.

- Data: the four pools above (data source = the drawn sample, `N_IMAGES=40`, `DATASET_SEED=0`,
  `HOLDOUT=0.25`, split seed 0, validation seed 1161 × 40, fresh seed 1160 × 200, both pool digests).
- Training: batch size 2; seed 0 (the re-headed classification layers and the shuffle); SGD with
  momentum 0.9 and weight decay 5e-4 on the trainable parameters; the L1 term off; backbone frozen
  with its BatchNorm in eval mode; the final-epoch weights.
- Evaluation: thresholds 0.01 / 0.65; at most 100 detections per image; IoU thresholds
  0.50:0.05:0.95; 101-point interpolation; tied scores scored as one operating point (all detections
  with the same score are accepted together, so the result does not depend on image order).
- The grid below, its selection rule and the keep-baseline rule.
- The fresh test is not masked (`DIMER_MASK_FRESH_TEST` unset).

## Grid

| Config | Epochs | Learning rate |
|---|---|---|
| C0 (the current default) | 6 | 1e-3 |
| C1 | 10 | 1e-3 |
| C2 | 8 | 2e-3 |

The backbone stays frozen in all three. Everything else is fixed as above. Each config starts from
the verified checkpoint with the same seeded head.

**Selection rule (validation only).**
1. Score each config by validation AP@[.50:.95].
2. A challenger replaces C0 only if its validation AP is at least **0.03** above C0's AND its
   validation AP50 is at most **0.02** below C0's.
3. If both challengers qualify, the larger AP gain wins; exact ties (within 1e-9) go to the smaller
   config, in the order C0, C2, C1 (by epochs).
4. **Keep-baseline rule.** The selected adapted config replaces the pretrained (re-headed, untrained
   head) baseline only if its validation AP is at least **0.03** above the baseline's. Otherwise the
   selected system is the baseline, and the run reports a negative result.

These margins are practical tolerances, not significance tests. 0.03 is about half the 0.064 AP
spread between the two hosted T4 records of C0 on the exposed development test.

**Where the grid runs.**
- With a GPU the grid is ON by default. It costs about 24 s of fine-tuning on a T4 (planning
  estimate from recorded timings).
- On CPU the grid is OFF by default. The run then trains the recorded selection (C0 until a hosted
  record exists). Such a run is EXPLORATORY and never scores the fresh test.
- The decision is taken on the hosted T4 run only. CPU and T4 runs of C0 differ by about 0.1 AP
  on the exposed development test, so CPU numbers are development looks.

## Reading rule

- **Primary endpoint:** fresh-test AP@[.50:.95], selected system minus the pretrained baseline,
  both scored on the same 200 images.
- **Interval:** a paired 95 % percentile bootstrap that resamples IMAGES (each image carries its 1–3
  boxes and all its detections), n_boot = 10 000, seed 20261018. AP is recomputed on each resample
  for both systems with the same weights.
- **Verdict:**
  - "improved" if the CI lower bound is > 0 AND the gain is ≥ 0.03;
  - "worse" if the CI upper bound is < 0;
  - otherwise "no detectable difference".
  - If the keep-baseline rule selected the baseline, the verdict is "same model (adaptation not
    selected)".
- Secondary, descriptive only: fresh-test AP50 (it saturates), per-class AP50, and the exposed
  development test.
- **Planning estimate (not a measurement).** If the per-image standard deviation of the AP change
  were about 0.5, n = 200 gives a standard error of about 0.035. The expected CI half-width is then
  about ±0.07 AP, and the minimum detectable change at 80 % power (two-sided 5 %) is about 0.10 AP.
  The recorded C0 effect on the exposed development test is +0.57 to +0.63 AP. The measured
  half-width replaces this estimate in `docs/release-verification.md` after the hosted run.

## Score-once rules

- The fresh test is scored only when ALL of these hold, re-checked immediately before it is touched:
  1. `preregistered`, computed over ALL the fixed settings above;
  2. the grid is complete (C0, C1, C2 all ran) and every row ran with those settings;
  3. the model in hand equals the selected system (state fingerprint);
  4. the generated fresh pool matches its committed digest.
- Any other run is EXPLORATORY: validation and the exposed development test only, written to
  `*_exploratory_<n>` files. A run with `DIMER_MASK_FRESH_TEST=1` (a maintainer development check)
  never generates or scores the fresh test and writes `*_masked_<n>` files.
- A look counts only after scoring completes and its report is written. Reports and the look ledger
  are written atomically (temporary file, then rename). A failed or partial grid, a failed generation
  or digest check, and an interrupted write are NOT looks. An empty or corrupt ledger or report never
  crashes a run or relabels it.
- A repeat pre-registered scoring in the same runtime writes `*_reproduction_<n>` files and is
  labelled "reproduction". In-notebook a pre-registered scoring is labelled "pre-registered
  evaluation (this runtime)"; `docs/release-verification.md` records which hosted run is the
  evaluation of record. The first hosted T4 run with these settings is that record; later identical
  hosted runs are reproductions. `RECORDED_EVALUATION` stays `None` until that run exists.
- CPU checks by the fixer run with the fresh test masked. Nothing seen on validation, the exposed
  development test or a CPU run changes any setting in this document.
