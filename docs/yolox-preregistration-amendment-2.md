# Amendment 2 to the YOLOX-S pre-registration (2026-10-18, round 3): deterministic GPU training and fresh test 3

This document is committed alone, before the code that implements it. It amends
`docs/yolox-preregistration.md` and amendment 1, which both stay unchanged.

## Why

The hosted reproduction 116b (commit `d6794c9`) ran the same code and settings as the evaluation of record
(`ad795d2`), but it selected C0 where the record selected C1:
- C0's validation AP moved from 0.6966 to 0.8624;
- the loss curves are identical through epoch 2 and differ from epoch 3 on.

The cause is CUDA non-determinism. Under it, the 0.03 selection margin sits inside run-to-run noise on a T4. The
fresh-test gain was reproduced within its interval (+0.7211 vs +0.7231). Kurt ruled on 2026-10-18 that GPU
training is to be made deterministic, a third fresh pool added, and a new evaluation of record run. This
decision was taken on the selection instability; the fresh-test numbers did not motivate it.

## History stays valid

The evaluation of record `ad795d2` (fresh test 2, seed 1160) and its reproduction 116b remain valid as
history. Fresh test 2 is renamed the **exposed test (scored by the round-2 record)**. From now on it is
descriptive only and decides nothing.

## B1. Determinism (new pre-registered settings)

Before CUDA initialises, the notebook sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Before any training it sets:
- `torch.backends.cudnn.deterministic = True`;
- `torch.backends.cudnn.benchmark = False`;
- `torch.use_deterministic_algorithms(True)`, **without** `warn_only`;
- seeded `torch`, `numpy` and `random` (the fine-tune's seed). Batches are built in-process: no DataLoader
  and 0 workers.

Any operation without a deterministic CUDA kernel then raises an error. It must be named and either replaced,
or the work STOPS and is reported; `warn_only` is never used. The flags and the device are recorded in every
report, and the flags enter the `preregistered` check.

## B2. Fresh test 3

- Pool: `sign_dataset(200, seed=1162)`, never generated before this amendment.
- It is pinned by SHA-256 under Pillow 11.3.0 in the code commit, and checked disjoint (by image digest and by
  box layout) from:
  - training and the exposed development test (seed 0);
  - validation (1161);
  - the exposed test of round 2 (1160);
  - Section 10's images (seed 99).
- The score-once labelling restarts for fresh test 3: its first pre-registered scoring is "pre-registered
  evaluation (this runtime)", and later ones are reproductions.

## Unchanged

The validation pool (seed 1161), the grid (C0, C1, C2), the margins (0.03 AP, 0.02 AP50, keep-baseline 0.03),
the tie rules, the reading rule (primary endpoint, paired image bootstrap, n_boot 10 000, seed 20261018, the
verdicts) and amendment 1 (only a T4 counts, the gate recomputes, the spent-slot rule).

## B3. The new evaluation of record

1. **Masked determinism check first.** Two hosted T4 runs with `DIMER_MASK_FRESH_TEST=1`, so fresh test 3 is
   never generated. Both must print identical grid numbers:
   - validation AP to every printed digit;
   - the selection;
   - the loss curves;
   - the grid hash that Section 8 prints.

   If they differ, STOP: there is no run of record.
2. Then the first hosted T4 run under this amendment that scores fresh test 3 is the evaluation of record.
   `RECORDED_EVALUATION` for amendment 2 is None until it exists.
3. A grid-off CPU run keeps training the round-2 selection C1 (`RECORDED_SELECTION`) until the amendment-2
   record exists. CPU runs stay exploratory.

## Note after the evaluation of record (dated; the text above is unchanged)

**2026-10-18.** The amendment-2 evaluation of record exists: the hosted T4 run of commit `4429e0e`
(notebook blob `a9dd3990`; evidence `docs/execution-evidence/2026-10-18-4429e0e/`; see
`docs/release-verification.md`). The masked determinism check passed first, and the record printed the same
grid hash (`184785885a81…`) as the two masked runs.
- B3.2: `RECORDED_EVALUATION` in `protocol.py` now carries this record. Every later pre-registered scoring
  of fresh test 3, including the first one in a new runtime, is labelled `reproduction`, is written to
  `_reproduction_<n>` files and prints whether it matches the record.
- B3.3: the record selected C1, as the round-2 record did, so `RECORDED_SELECTION` stays C1 and now cites
  this record. CPU runs stay exploratory.
- CI width: the pre-registration planned a half-width of about ±0.07 AP. On fresh test 3 the observed 95 %
  interval was [0.7381, 0.782] around a difference of +0.7616, a half-width of 0.022.
