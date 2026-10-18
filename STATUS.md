# Release status

Current status: **Candidate**

The repository is complete and internally consistent: the package and its unit tests, the vendored
upstream provenance check, the standalone `E2E` notebook and its parity check, the model card, and the
static release-asset validation all pass, and the notebook has been executed once top-to-bottom in a
fresh local CPU kernel.

A hosted run is also recorded: on 2026-09-15 a Kaggle T4 kernel executed an earlier blob of the
notebook (22/22 code cells), but only after a manual restart following the old in-kernel install cell.
That run is workflow history, not one-pass clean-runtime evidence, so gate 6 is open. The 2026-10-05
review fixes regenerated the notebook with an isolated `uv` environment that needs no restart; the regenerated
blob `fe56cb6950f3` (commit `2c3d8bc`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-10 (Colab CLI 0.7.4 sequential execution, 24/24 code cells, 103.3 s; baseline AP 0.2083 / AP50 0.2222 → adapted AP 0.7766 / AP50 1.0; fresh reload identical). Status stays Candidate until a reviewer/integrator accepts that record; the REL12 BYOD journey is not recorded.

The pre-registered **evaluation of record** ran on 2026-10-18: blob `5fe2ef5d4628` (commit `ad795d2`) on a fresh Colab Tesla T4 (Colab CLI 0.7.4 sequential execution, 25/25 code cells, 0 errors, 134.6 s). The validation grid selected C1 (10 epochs at 1e-3; validation AP C0 0.6966 / C1 0.8554 / C2 0.7122). On the 200-image fresh test, C1 scored AP@[.50:.95] 0.8573 against the re-headed baseline's 0.1363: +0.7211, 95 % CI [0.6983, 0.7435], verdict *improved*. The baseline's classification head is untrained, so a large gain was expected. The fresh test is drawn by the same generator, so it measures in-distribution generalisation, not detection on real images. See "Hosted T4 evaluation of record (2026-10-18)" in [`docs/release-verification.md`](docs/release-verification.md). Under amendment 2 (deterministic GPU training, fresh test 3), the **amendment-2 evaluation of record** ran on 2026-10-18: blob `a9dd3990a1d8` (commit `4429e0e`) on a fresh Colab Tesla T4 (Colab CLI 0.7.4, 25/25 code cells, 0 errors, 143.6 s), after two masked hosted runs printed the same grid hash. The grid selected C1 again (validation AP C0 0.7037 / C1 0.8937 / C2 0.8296). On fresh test 3 (200 images, seed 1162), C1 scored AP@[.50:.95] 0.8933 against the baseline's 0.1317: +0.7616, 95 % CI [0.7381, 0.782], verdict *improved*. A reproduction on 2026-10-18 (blob `6b295979c64f`, commit `f0d021e`, fresh Colab Tesla T4, 25/25 code cells, 0 errors, 151.8 s) labelled itself *reproduction*, printed the same grid hash and matched the record exactly. The round-2 record above stays as history. See "Masked determinism check" and "Amendment-2 evaluation of record" in the same file. Status stays **Candidate**.

That local execution is **pre-flight evidence, not clean-runtime evidence**. A green CI run is static
and unit validation only; neither it nor a local kernel run establishes that the notebook completes in
a supported hosted runtime from a cold start. Promotion to **Release-grade** requires a clean-room
execution on a supported runtime recorded under "Recorded executions" in
[`docs/release-verification.md`](docs/release-verification.md), which also lists the gates that remain
open.
