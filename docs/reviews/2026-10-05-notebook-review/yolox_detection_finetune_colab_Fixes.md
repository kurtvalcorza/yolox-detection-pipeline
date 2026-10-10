# YOLOX-S detection fine-tuning notebook — review fixes

**Review:** `yolox_detection_finetune_colab_Review.md` (5 October 2026). Findings: YXD-M1 to YXD-M3 and YXD-m1 to YXD-m3.

**Fixed in:**
- `tools/build_notebook.py`, now the isolated-runtime generator. It is vendored as `build_notebook.py/2.1-swc-yolox`, which carries over this family's `external_source`, `stage_source` and `external_access` keys.
- `tools/notebook_template.py`.
- `pipeline.py`, which gains `read_detection_records` and `load_byod_image`.
- The new hash lock `tutorials/requirements-colab.lock.txt`.
- The validator, `docs/UPSTREAM.md`, `tutorials/README.md`, `docs/release-verification.md` and the tests.

The notebook was regenerated and `--check` passes.

**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged. This includes the gate-6 state token, which now carries an annotation instead.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| YXD-M1 | Fixed in source; run and gate token pending | Isolated `uv` runtime: pinned `uv` 0.12.15 (size + SHA-256), managed CPython 3.12.12, hash lock of 32 packages from `pyproject.toml`. No kernel `pip`, no restart guard. Spec 2.2. Gate 6 in the record now states that the Kaggle run needed a restart and needs re-qualification with `restarted: false`. The **met** token itself is left for the maintainer, because release labels are out of scope for this cycle. | Sections 1–3; validator; record | `test_M1_*`; `grep "Restart the runtime"` returns nothing |
| YXD-M2 | Fixed | **Input.** The BYOD dataset branch reads `annotations.json` plus images from `BYOD_DATASET_DIR`, or from files chosen in the upload dialog, through `read_detection_records`. The reader refuses paths outside the directory, missing files and undecodable images by name. `BYOD_CLASS_NAMES` is a comma-separated field; when empty, the sorted labels are used. At least 8 records are required.<br>**Stages.** The branch then runs validate → split → baseline → fine-tune → evaluate → export. Next comes the fresh reload with an identical-score assertion, then `outputs/byod_yolox_result.json`.<br>**Cleanup.** The empty-list placeholder and its `noqa` are gone. | `pipeline.read_detection_records`; Section 13 | `test_M2_*`: real images through the notebook's own BYOD cell with a stand-in detector; minimum and incompatible-label refusals; path traversal and missing-file refusals |
| YXD-M3 | Fixed | **Orientation:** How to use, I/M/O table, roadmap, glossary.<br>**Predictions:** before Sections 4, 5 (RGB vs BGR, degenerate), 7, 8, 9 and 10, each followed by What to notice and a collapsed answer quoting the recorded numbers. The announced miss ("Expect one honest failure") is now a prediction with its answer after the result.<br>**Activity:** a Section 14 activity that unfreezes the backbone.<br>**Support:** troubleshooting table and conclusion template.<br>**Infrastructure:** cells titled and collapsed. | template | `test_M3_*` (12 tests) |
| YXD-m1 | Fixed | The Prerequisites use single braces; no markdown cell contains `{{`. The upstream `# TODO` in vendored `yolo_head.py` is now documented and justified in `docs/UPSTREAM.md` (written by `tools/vendor_upstream.py`, so `--check` stays green). It is the only placeholder marker in the notebook, as the acceptance check allows. | Prerequisites; `docs/UPSTREAM.md` | `test_m1_no_template_braces_and_only_the_documented_upstream_todo` |
| YXD-m2 | Fixed | Section 8 rebuilds `adapter` from the verified base with the same seeded head before every fine-tune and times it. Section 9 keeps `run_history` and prints the runs side by side. | Sections 8–9 | `test_m2_rerunning_the_finetune_cell_trains_a_fresh_adapter` |
| YXD-m3 | Partly fixed | Fixed: the image branch gains `BYOD_IMAGE_PATH`, a single-upload rule and `load_byod_image`, which names the file it cannot decode. Not changed: `STATUS.md` still describes only the local run, because status files are out of scope for this cycle. The record's new section notes this. | Section 13; record | `test_m3_*` |

Suggestions not taken: S1 (per-class support), S2 (grid table in the notebook; the Section 14 answer quotes it) and S3. S4 applied (spec 2.2).

## User-visible changes

- Section 1 builds an isolated environment for Linux x86_64 only. The Linux `torch` with its CUDA libraries is a larger download than the CPU wheel. Every later cell runs in that environment.
- Section 8 rebuilds the adapter, adding one model load, and prints the fine-tune time. Default results are expected to be unchanged, because the same seed gives the same head. Section 9 prints `run_history`.
- **Section 13:**
  - New fields: `BYOD_IMAGE_PATH` and `BYOD_DATASET_DIR`.
  - `BYOD_CLASS_NAMES` is now a string field; it was a list.
  - The dataset branch reaches adaptation, reload and `byod_yolox_result.json`.
- The pipeline package exports `read_detection_records` and `load_byod_image`.

## Verification (offline, not clean-runtime evidence)

- `build_notebook.py --check`: OK. `validate_release_assets.py`: PASS. `ruff check src tests tools`: clean. The vendored tree is unchanged, so `vendor_upstream.py --check` is unaffected.
- pytest without torch went from 123 passed / 10 failed to **144 passed / 10 failed**. The 10 failures were there before: the vendored-module import tests need torch, which CI installs.
- The BYOD readers ran on real drawn images. The Section 8 and Section 13 cells ran from the generated notebook with a stand-in detector, which proves the plumbing but not the model.
- No YOLOX inference or training ran here, because torch is not installed.

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated blob, recorded with `restarted: false`, then a re-run of the export cell. Afterwards the maintainer decides the gate-6 token.
2. The REL12 BYOD journey: one image by path and one through the dialog; one labelled directory; one incompatible `annotations.json`.
3. Maintainer: bring `STATUS.md` in line with the record.
