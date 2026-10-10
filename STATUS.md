# Release status

Current status: **Candidate**

The repository is complete and internally consistent: the package and its unit tests, the vendored
upstream provenance check, the standalone `E2E` notebook and its parity check, the model card, and the
static release-asset validation all pass, and the notebook has been executed once top-to-bottom in a
fresh local CPU kernel.

A hosted run is also recorded: on 2026-09-15 a Kaggle T4 kernel executed an earlier blob of the
notebook (22/22 code cells), but only after a manual restart following the old in-kernel install cell.
That run is workflow history, not one-pass clean-runtime evidence, so gate 6 is open. The 2026-10-05
review fixes regenerated the notebook with an isolated `uv` environment that needs no restart; no
hosted run of the regenerated notebook is recorded yet.

That local execution is **pre-flight evidence, not clean-runtime evidence**. A green CI run is static
and unit validation only; neither it nor a local kernel run establishes that the notebook completes in
a supported hosted runtime from a cold start. Promotion to **Release-grade** requires a clean-room
execution on a supported runtime recorded under "Recorded executions" in
[`docs/release-verification.md`](docs/release-verification.md), which also lists the gates that remain
open.
