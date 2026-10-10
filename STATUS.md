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

That local execution is **pre-flight evidence, not clean-runtime evidence**. A green CI run is static
and unit validation only; neither it nor a local kernel run establishes that the notebook completes in
a supported hosted runtime from a cold start. Promotion to **Release-grade** requires a clean-room
execution on a supported runtime recorded under "Recorded executions" in
[`docs/release-verification.md`](docs/release-verification.md), which also lists the gates that remain
open.
