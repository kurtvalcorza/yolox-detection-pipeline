# Amendment 1 and erratum to the YOLOX-S pre-registration (2026-10-18, round 2)

This document is committed alone, before the code that implements it. It amends
`docs/yolox-preregistration.md`, which stays unchanged. No fresh-test score exists when this is written:
the fresh test has never been generated and scored by any run, CPU or hosted. The changes come from the
peer review of PR #8 (findings F1–F10). None of them was prompted by a test number.

## A1. The device is a checked pre-registered setting (F1)

The shared design already says the decision is taken on the hosted T4 run only, and that CPU runs are
exploratory. Until now the code did not check this: a CPU run with `RUN_GRID = "on"` would have scored the
fresh test and been labelled the pre-registered evaluation.

- New fixed setting: `device_class = "cuda-t4"`. The notebook computes it as follows:
  - `cpu` when CUDA is not available;
  - `cuda-t4` when the CUDA device name contains `T4`;
  - `cuda-other` for any other GPU.
- Any device other than `cuda-t4` makes the run EXPLORATORY, whatever `RUN_GRID` says. Such a run never
  generates or scores the fresh test.
- Every report records the device class and the device name.

## A2. The fail-closed gate recomputes, never trusts leftover state (F10)

- Section 8 clears the grid status, the selection and the selected model before it trains anything. An
  interrupted Section 8 therefore leaves no usable selection, and Section 9 stops with a message instead of
  scoring.
- Section 9 recomputes the grid check from the grid rows of the current Section 8 run and the current
  settings. It requires all three of:
  - all of C0, C1 and C2 are present;
  - the selection's fingerprint equals its grid row's fingerprint;
  - the model in hand has that fingerprint.

## A3. When is the pre-registered slot spent? (F2, F3)

- Reports are written atomically (temporary file, then rename), so a main report file that exists was
  completely written by a completed look at least once.
- From now on, the pre-registered slot counts as spent if any of these is true:
  - a main report file exists, even an empty or unreadable one;
  - any `_reproduction_<n>` report exists;
  - the ledger records a completed look.
- In that case the next scoring is labelled "reproduction". Only an absent main report, with no other
  evidence, re-opens the slot. An interrupted write still leaves no main report, so it is still not a look.
- Ledger values that are not finite numbers (NaN, Infinity) are ignored, like any other corrupt entry.

## A4. Grid files per run (F4)

Grid rows are written to `outputs/grid/run_<k>/`, with one folder per execution of Section 8. An
exploratory re-run no longer overwrites the rows of an earlier run.

## A5. Sections 10–11 use the selected system (F7)

The new-data demonstration, the artifact and the reload check use the selected system. If the
keep-baseline rule kept the pretrained baseline, the notebook says so, and the exported artifact is that
baseline: the rejected adapted config is not shipped as the result.

## Erratum (F6)

"Where the grid runs" says that CPU and T4 runs of C0 differ by about 0.1 AP on the exposed development
test. The repository's own records do not show that. Recorded C0 AP@[.50:.95] on the exposed development
test:
- reference CPU 0.8110 (2026-09-15);
- Kaggle T4 0.8403 (2026-09-15);
- Colab T4 0.7766 (2026-10-10).

So |CPU − T4| ≤ 0.034, and the two T4 runs differ by 0.064 (different notebook revisions). The rule is
unchanged: CPU is not the decision device, because GPU runs are the evaluation device and are themselves
variable.

## Note on the selection margin (F5)

- In the masked CPU development look, AP50 was 1.0000 for C0, C1 and C2. The AP50 guard therefore decides
  nothing here, and only AP@[.50:.95] decides.
- 0.03 is a practical tolerance at the observed T4 noise floor, not a noise-calibrated threshold.
- CUDA determinism is not enforced. The hosted selection is one draw, and a later hosted reproduction may
  select a different config. That does not bias the primary endpoint, because the fresh test plays no
  part in selection.
- The margin is not changed.

## Unchanged

Everything else in `docs/yolox-preregistration.md` stands: pools and digests, the grid, the margins, the
keep-baseline rule, the reading rule (paired image bootstrap, n_boot 10 000, seed 20261018, the verdicts)
and the score-once rules.

## Note after the evaluation of record (2026-10-18; dated, the text above is unchanged)

The erratum above (|CPU − T4| ≤ 0.034) holds only for C0 on the exposed development test, across the
earlier records. It does not hold on validation. The hosted T4 evaluation of record (commit `ad795d2`)
differs from the CPU masked check by up to 0.116 AP@[.50:.95]:

| Config | T4 record | CPU masked check | T4 − CPU |
|---|---|---|---|
| C0 | 0.6966 | 0.7760 | −0.079 |
| C1 | 0.8554 | 0.8871 | −0.032 |
| C2 | 0.7122 | 0.8285 | −0.116 |

The CPU check ran older code (`825771a`) than the T4 run (`b242f44`), so these gaps mix device and code
effects. Both runs selected C1. The rule is unchanged: only a T4 run decides.
