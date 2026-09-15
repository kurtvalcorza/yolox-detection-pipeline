# Tutorials

## Registry

| Notebook | Profile | Mode | Spec | Standalone | Workflow | Runtime | Sample | BYOD | Status |
|---|---|---|---|---|---|---|---|---|---|
| [`yolox_detection_finetune_colab.ipynb`](yolox_detection_finetune_colab.ipynb) | `E2E` | `GUIDED` | DIMER Notebook Specification 2.0 (standalone, §4) | yes | COCO detection, then validate → split → baseline → bounded SimOTA fine-tune → evaluate → new-data inference → export → fresh reload | CPU (GPU used automatically when present) | automatic, drawn in code | optional, two branches, both off by default | Candidate |

Candidate status and the gates that remain open are recorded in
[`../docs/release-verification.md`](../docs/release-verification.md); the local execution below is
pre-flight evidence, not clean-runtime evidence.

## What the notebook does

It is **standalone**: it carries the repository's ten-module package verbatim, the pinned model
identity and the file manifest, and the exact runtime pins, so it keeps working after export even if
this repository changes. There is no clone, no `pip install -e .`, no DIMER worker or service, and no
credential. Its only external dependency is `github.com`, to download the pinned `yolox_s.pth` release
asset whose SHA-256 the notebook checks before anything opens it.

Because the profile is `E2E`, the default `Run all` path **actually adapts the model** — NOTEBOOK_SPEC
2.0 RUN7 and FT2 make a real bounded fine-tune mandatory rather than optional, so nothing is hidden
behind a default-off flag:

1. Install the pinned runtime and report versions.
2. Carry the package, module by module in dependency order.
3. Pin, stage and digest-verify the checkpoint.
4. Run COCO detection on a drawn scene and score it per object — including one object the pretrained
   model gets wrong.
5. Probe channel order (BGR vs RGB) and degenerate inputs.
6. Build and validate a 40-image labelled dataset over a three-class vocabulary that is not in COCO.
7. Split it, re-head the detector, and measure a baseline **before** adapting.
8. Run the bounded SimOTA fine-tune.
9. Score the held-out split with COCO-style AP@[.50:.95] and AP50, against the baseline.
10. Detect on images from an unseen seed.
11. Export the artifact, reload it from disk, and assert the reloaded model scores identically.
12. Write every result as JSON with provenance.
13. Optional BYOD: your own image, or your own labelled dataset through the same local stages.

Expect roughly a few minutes on a hosted CPU runtime; on the reference machine the committed notebook
executed in 39.3 s with the checkpoint pre-staged and the pins already present, of which the fine-tune
is about 26 s.

## Reading the numbers

The notebook reports real measurements on **synthetic** data. Both the COCO demonstration scene and the
adaptation dataset are drawn in code with Pillow — flat colours, hard edges, no lighting, no occlusion.
A held-out AP50 of 1.0 after fine-tuning says the plumbing works and the task is easy; it is not
evidence about photographs, and the much lower AP@[.50:.95] is the more honest summary. Upstream's
published 40.5 COCO mAP for YOLOX-S is neither reproduced nor checked.

## Regenerating

The notebook is generated from repository sources and must never be edited by hand:

```bash
python tools/build_notebook.py            # rewrite tutorials/<notebook>
python tools/build_notebook.py --check    # fail if it has drifted from the package (PAR3)
```

`tests/test_notebook_parity.py` enforces the same thing in CI.
