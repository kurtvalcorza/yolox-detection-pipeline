"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.2 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "yolox_detection_pipeline",
    "repo_name": "yolox-detection-pipeline",
    "stem": "yolox_detection",
    "notebook_name": "yolox_detection_finetune_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # NOTEBOOK_SPEC 2.2 §5 (YXD-M1): the pins are installed into an isolated uv environment and every later cell runs in
    # a persistent worker there, so a hosted runtime's preloaded packages never force a restart. The lock is compiled with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "YoloxDetectionPipeline",
    "weights_key": "yolox-s",
    # The package is nine modules: eight vendored upstream YOLOX files plus this repository's own
    # pipeline.py and samples.py. The generator emits one cell per module in dependency order.
    "modules": [
        "coco_classes.py",
        "ops.py",
        "network_blocks.py",
        "darknet.py",
        "yolo_pafpn.py",
        "losses.py",
        "yolo_head.py",
        "yolox.py",
        "samples.py",
        "pipeline.py",
    ],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "torchvision", "numpy"],
    "title": "YOLOX-S — DIMER anchor-free detection and bounded detection fine-tuning (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/yolox-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/yolox-detection-pipeline/blob/main/tutorials/yolox_detection_finetune_colab.ipynb",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-Megvii--BaseDetection%2FYOLOX-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/Megvii-BaseDetection/YOLOX",
        ),
        (
            "Weights",
            "https://img.shields.io/badge/Weights-yolox__s.pth%20%400.1.1rc0-blue?style=flat",
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0",
        ),
        (
            "arXiv",
            "https://img.shields.io/badge/arXiv-2107.08430-b31b1b.svg",
            "https://arxiv.org/abs/2107.08430",
        ),
        (
            "License",
            "https://img.shields.io/badge/License-Apache--2.0-green.svg",
            "https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE",
        ),
    ],
    # The weights are a GitHub release asset: YOLOX publishes no Hugging Face model repository, so the
    # generator's default Hub wording would be wrong here.
    "external_source": "the immutable GitHub release assets of upstream tag `{rev}`",
    "stage_source": "the upstream GitHub release assets **at tag `{rev}`** (an immutable tag, never `main`)",
    "external_access": "- **External access:** `github.com` only, to download the pinned `yolox_s.pth` release asset (~{mb} MB) from upstream tag `{rev}`. YOLOX publishes no Hugging Face model repository, so the checkpoint is a release asset rather than a Hub revision; the SHA-256 in the manifest below is what makes that download trustworthy. No credentials are required, and no code is fetched from any repository — the YOLOX model code you need is carried in Section 2.",
    "capability": "anchor-free object detection over the 80 COCO classes, and a bounded SimOTA fine-tune that re-heads the detector onto your own class vocabulary, evaluates it against a held-out split with COCO-style average precision, and exports a reloadable artifact",
    "intro": (
        "YOLOX-S is a small anchor-free detector: a CSPDarknet backbone and a PAFPN neck feed a **decoupled head** that emits, at each of "
        "8400 anchor points over a 640×640 letterboxed image, one box, one objectness logit and one logit per class. There are no anchor boxes "
        "to tune and no class softmax — objectness and class probability are independent sigmoids whose product is the score, and per-class NMS "
        "does the rest. At training time the same head runs **SimOTA**: it decides for itself which anchor points should be responsible for which "
        "ground-truth object, by solving a small assignment problem over an IoU-and-classification cost, instead of using a fixed rule.\n\n"
        "That last property is why this notebook can fine-tune a detector in a few minutes on a CPU. **The default path really adapts the model:** "
        "it re-heads YOLOX onto a three-class sign vocabulary that does not exist in COCO, measures a pre-adaptation baseline, runs a bounded "
        "SimOTA fine-tune with the backbone frozen, scores the result against a held-out split it never trained on, runs the adapted model on "
        "unseen images, exports the weights as one artifact and reloads that artifact as if from a cold start. Every number you will see comes "
        "from cells in this notebook, in this runtime.\n\n"
        "Two facts about the input pipeline are load-bearing and the notebook will show you both. Channel order is **BGR**, raw 0–255, with no "
        "`/255` and no mean/std normalisation — that is what upstream feeds the network, and handing it RGB instead measurably degrades detection. "
        "And the letterbox pads with grey 114 rather than black. The carried package owns both, so you do not have to."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees and which of its modules are vendored upstream YOLOX rather than "
        "DIMER code; stage and digest-verify an immutable GitHub release asset (not a Hub revision) before anything unpickles it; run COCO "
        "detection on a drawn scene and read the objectness × class score, the two caller-owned thresholds and the per-object `box_iou` correctly, "
        "including one object the pretrained model gets wrong; build and validate a labelled detection dataset in the record shape the fine-tuner "
        "accepts; split it and measure a **baseline before adapting**; run a bounded SimOTA fine-tune onto a new class vocabulary; score it with "
        "COCO-style AP@[.50:.95] and AP50 on the held-out split; detect on images from an unseen seed; and export, reload and re-verify the "
        "resulting artifact."
    ),
    "exclusions": (
        "real traffic-sign detection or any deployment claim — the adaptation dataset is drawn in code, so the fine-tuned model has learned these "
        "renderings and nothing about photographs; published COCO numbers (upstream reports 40.5 AP for YOLOX-S on COCO test-dev, which this "
        "notebook neither reproduces nor checks, and the average-precision helper here is a small faithful implementation without pycocotools' "
        "area ranges or crowd handling); upstream's mosaic/mixup augmentation, its 300-epoch schedule, its EMA and its learning-rate warmup, none "
        "of which a bounded tutorial run uses; the L1 box-refinement term, which upstream only enables for the last 15 epochs; full-network "
        "fine-tuning (the default freezes the backbone, and the repository records that unfreezing it at this learning rate collapses the model); "
        "instance segmentation, tracking, batched or video inference, quantisation, ONNX/TensorRT export, and upstream's latency figures."
    ),
    "guided_opening": [
        (
            "## How to use this notebook\n\n"
            "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab, Kaggle or Jupyter), run cells in order and read "
            "short Python, and who want to see how a pretrained object detector is checked, adapted to new classes, evaluated and exported "
            "honestly. No prior experience with YOLOX is assumed: each term is explained where it is first needed and again in the glossary "
            "below. A CPU runtime is enough. The **Prerequisites** give the details.\n\n"
            "**Running it.** Choose *Runtime → Run all*. The default path needs no edit, no upload, no account, no token and no runtime restart. "
            "Section 1 builds the isolated environment (the pinned `torch` is the largest download, so it is the slowest step); a second Run all "
            "in the same runtime reuses it.\n\n"
            "**Where the code runs.** The first two code cells run in the notebook kernel: they build the environment and start one Python process "
            "inside it. Every later cell is sent to that process, so the pinned `torch`, `torchvision` and `numpy` are used without replacing "
            "anything the hosted runtime had already loaded. Printed output, images and errors come back to the notebook as usual, and variables "
            "persist from cell to cell.\n\n"
            "**Two kinds of cell.** *Infrastructure cells* (Sections 1–3: the isolated install and router, the carried package — mostly vendored "
            "upstream YOLOX code — and the pinned-checkpoint staging) are collapsed and labelled **Infrastructure**; you may run them without "
            "studying their implementation. *Learner cells* (Sections 4–13) are the machine-learning workflow.\n\n"
            "**Form controls.** Learner cells start with fields Colab renders as a form: `threshold`, `nms_threshold` (Section 4); `N_IMAGES`, "
            "`DATASET_SEED`, `EPOCHS` (Section 6); `HOLDOUT`, `SEED` (Section 7); `LEARNING_RATE`, `BATCH_SIZE`, `FREEZE_BACKBONE` (Section 8); "
            "`NEW_DATA_SEED` (Section 10); the BYOD switches and paths (Section 13). Leave them at their defaults for the first run. After a "
            "change, select the changed cell and choose *Runtime → Run after*.\n\n"
            "**Section tags.** Each learner heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation practice]** — how "
            "the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an expectation; after "
            "it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives a worked answer. Write your own "
            "answer first, then open it. The numbers the answers quote come from the repository's recorded runs of the previous version of this "
            "notebook (reference CPU and Kaggle T4, 15 September 2026); no run of this revision is recorded yet."
        ),
        (
            "## The task: Input → Model/System → Output\n\n"
            "| Stage | Input | Model / system | Output |\n"
            "|---|---|---|---|\n"
            "| **Detect (COCO)** | one RGB image, letterboxed to 640×640, fed as BGR 0–255 | YOLOX-S: CSPDarknet backbone, PAFPN neck, decoupled head at 8400 anchor points | boxes (xyxy, input pixels), labels, objectness × class scores after per-class NMS |\n"
            "| **Validate** | labelled records `{{image, boxes, labels}}` over a vocabulary | the carried `validate_dataset` | a dataset manifest with boxes per class |\n"
            "| **Adapt** | 30 training images over three new sign classes | re-headed classification branch; head trained with upstream's SimOTA loss, backbone frozen | an adapted detector |\n"
            "| **Evaluate and export** | 10 held-out and 3 unseen images | COCO-style AP; then one artifact reloaded from disk | AP before/after, unseen-image results, an identical-score reload |\n\n"
            "## Roadmap\n\n"
            "| Section | Tag | What happens | What you read |\n"
            "|---|---|---|---|\n"
            "| 1. Install the pinned runtime | [Engineering] | isolated environment built; later cells routed to it | versions, CUDA |\n"
            "| 2. Package code | [Engineering] | vendored YOLOX plus the repository's modules, carried verbatim | nothing to run by hand |\n"
            "| 3. Pin, stage and verify the model | [Engineering] | release asset downloaded and digest-checked | the verified file |\n"
            "| 4. COCO detection on a drawn scene | [Concept] | the pretrained detector at 0.3 / 0.3 | boxes, per-object IoU |\n"
            "| 5. Channel order and degenerate inputs | [Evaluation practice] | BGR versus RGB; blank and noise images | what changes |\n"
            "| 6. Adaptation dataset | [Evaluation practice] | 40 drawn sign images validated | boxes per class |\n"
            "| 7. Split and baseline | [Evaluation practice] | 30 / 10 split, re-headed model scored | baseline AP |\n"
            "| 8. Fine-tune | [Concept] | a bounded SimOTA fine-tune of the head | losses, `num_fg` |\n"
            "| 9. Evaluate | [Evaluation practice] | held-out AP after adaptation | the change, and why AP50 1.0 is saturated |\n"
            "| 10. New data | [Concept] | three images from an unseen seed | signs found |\n"
            "| 11. Export and reload | [Engineering] | artifact, fresh reload, tamper check | identical scores |\n"
            "| 12. Outputs | [Engineering] | JSON with provenance | the file list |\n"
            "| 13. BYOD (optional) | [Engineering] | your image or labelled directory | the same stages |\n"
            "| 14. Activity (optional) | [Concept] | unfreeze the backbone and compare | your comparison |\n"
            "| Troubleshooting, Conclusion | — | recovery, your notes | when needed |\n\n"
            "**Fast path.** Short on time? Run all, then read Sections 5, 7 and 9 and the conclusion."
        ),
        (
            "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
            "| Term | Meaning in this notebook |\n"
            "|---|---|\n"
            "| **Anchor point** | One of 8400 grid positions over the 640×640 input (strides 8, 16, 32) at which the head predicts one box. |\n"
            "| **Decoupled head** | Separate branches for the box, the objectness and the class scores. |\n"
            "| **Objectness** | The head's estimate that an anchor point holds any object; the score is objectness × class probability. |\n"
            "| **Letterbox** | Resizing to fit 640×640 while keeping the aspect ratio, padding the rest with grey 114. |\n"
            "| **BGR** | Blue-green-red channel order, what upstream's `cv2.imread` produces and the network was trained on. |\n"
            "| **NMS** | Non-maximum suppression: drop a box that overlaps a higher-scoring box of the same class by more than the NMS threshold. |\n"
            "| **SimOTA** | Upstream's training-time assignment: it chooses which anchor points learn each ground-truth object by a cost over IoU and class. |\n"
            "| **`num_fg`** | The number of anchor points SimOTA assigned as positives per image during training. |\n"
            "| **CSPDarknet / PAFPN** | The backbone that extracts features, and the neck that mixes them across three scales. |\n"
            "| **Re-heading** | Rebuilding the classification branch for your classes; it starts random. |\n"
            "| **AP50 / AP@[.50:.95]** | Average precision with a match at IoU ≥ 0.50, and averaged over IoU 0.50–0.95 (which rewards tight boxes). |\n"
            "| **Saturated** | A test so easy that the score is at its ceiling (1.0), so it cannot show further improvement. |\n"
            "| **Artifact** | One file with the adapted weights and the metadata to rebuild the model; reloaded and re-scored in Section 11. |\n"
            "| **Isolated environment** | A separate Python built from hash-locked pins, in which every learner cell runs. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). **CPU is enough and is the documented default** — this profile is float32 on both CPU and GPU, and CUDA is used automatically when present. For scale: on the repository's Windows CPU venv (Intel Core Ultra 9 275HX) loading takes 2.6 s, one `detect` 0.11 s, and the whole six-epoch fine-tune 25.9 s. A hosted CPU runtime is slower; budget a few minutes for the adaptation cell. On a Kaggle T4 the previous version of this notebook ran top to bottom in 27.0 s once installed (15 September 2026), after a first pass of 155.2 s that stopped for a restart in the old install cell; this version builds an isolated environment instead and has not been timed yet. The kernel's own Python version does not matter: Section 1 builds a separate **Linux x86_64** environment with CPython 3.12.12 from the hash-locked pins, and every later cell runs there. The Linux build of `torch==2.14.0` with its CUDA libraries (several GB) and the 72 MB checkpoint are the large downloads.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union measures; roughly what average precision summarises. You do **not** need to know YOLOX internals — SimOTA assignment and the loss are upstream's code, carried and called, not reimplemented here.",
        "- **Data:** everything is drawn in code by the carried `samples` module, so nothing is downloaded and no private data is needed: one 640×640 COCO demonstration scene with reference boxes, and a deterministic 40-image labelled sign dataset for the adaptation. Optional BYOD is gated off by default. Expected BYOD input: for detection, one image decodable by Pillow, sides 16–4096 px; for adaptation, a directory (`BYOD_DATASET_DIR`, or the files chosen in the upload dialog) holding `annotations.json` — a list of `{'file': 'name.png', 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...]}` objects, boxes in that image's own pixels — and the image files it names, at least 8 records. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, downloads and digest-verifies the pinned release "
        "asset, draws the COCO demonstration scene and scores it, builds and validates the labelled adaptation dataset, splits it, measures the "
        "pre-adaptation baseline on the held-out part, **runs the bounded SimOTA fine-tune**, re-scores the held-out split, detects on unseen "
        "images, exports the adapted artifact, reloads it from disk and confirms the reloaded model scores identically, and writes every result "
        "as JSON with provenance. Nothing is skipped behind a default-off flag, and the path needs no repository clone, no DIMER worker or "
        "service, no credential, no upload dialog, no configuration edit and no runtime restart: the pinned dependencies go into an isolated "
        "environment built from a hash lock, leaving the kernel's own packages alone (NOTEBOOK_SPEC 2.2 §5, RUN7, FT2). No run of this "
        "revision is recorded yet."
    ),
    "byod": (
        "Two BYOD branches are provided and both are optional and off by default. `USE_BYOD_IMAGE` runs your own image through the same "
        "validation and detection contract as the sample. `USE_BYOD_DATASET` takes your own labelled records and runs them through the *same* "
        "local stages the sample used — validate, split, baseline, fine-tune, evaluate — rather than only detecting with them, because this is an "
        "adaptation profile (NOTEBOOK_SPEC 2.2 DAT14, §25.10), followed by the same export, fresh reload and identical-score check, and a "
        "result JSON. `BYOD_IMAGE_PATH` and `BYOD_DATASET_DIR` read from a location without an upload dialog; the record format and the "
        "ceilings are stated in the Prerequisites and in Section 13; uploads stay inside this runtime."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. COCO scene
        {
            "md": (
                "## 4. What the pretrained detector does, and where it fails · [Concept]\n\n"
                "Before adapting anything, look at the model you are starting from. The carried `samples` module draws a deterministic 640×640 "
                "street scene containing five objects whose COCO labels and exact boxes it also returns: a **stop sign**, a **traffic light**, an "
                "analogue **clock**, a **sports ball** and a **bench**. Those boxes are references for a per-object `box_iou` check — they are "
                "drawn ground truth on a rendered picture, not a labelled photographic dataset, so nothing here is a mean-average-precision "
                "measurement.\n\n"
                "Why 640×640 specifically: the pipeline letterboxes every input to that size, so an image already at it skips the resize "
                "entirely and the tensor is exactly the drawn pixels with the channels reversed. Nothing about the sample depends on which "
                "resampling filter is used.\n\n"
                "Both thresholds are **caller-owned request parameters**, not pipeline constants. The package names two upstream pairs rather "
                "than inventing a house default: the demo pair `0.3 / 0.3` (upstream `tools/demo.py`, for looking at pictures) and the evaluation "
                "pair `0.01 / 0.65` (upstream `Exp.test_conf` / `Exp.nmsthre`, for average precision, which keeps far more low-scoring boxes "
                "because AP rewards recall). This cell uses the demo pair and passes it explicitly.\n\n"
                "Whatever the model misses is kept rather than tuned away, and the evaluation report records a miss as a zero.\n\n"
                "**Predict before running:** how many of the five drawn objects will be matched at IoU ≥ 0.5? Which one is the model most "
                "likely to miss, and what might it call it instead?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                'threshold = 0.3  # @param {{type:"number"}}\n'
                'nms_threshold = 0.3  # @param {{type:"number"}}\n\n'
                "scene, references = tutorial_scene()\n"
                "buffer = io.BytesIO()\n"
                "scene.save(buffer, format='PNG')\n"
                "print({{'sample_kind': 'synthetic', 'size': list(scene.size), 'sha256': hashlib.sha256(buffer.getvalue()).hexdigest()[:16],\n"
                "       'references': {{label: len(boxes) for label, boxes in references.items()}}}})\n\n"
                "input_manifest = validate_inputs(scene, threshold=threshold, nms_threshold=nms_threshold, names=['tutorial-scene'])\n"
                "print({{'verdict': input_manifest['verdict'], 'findings': input_manifest['findings'], 'inputs': input_manifest['inputs']}})\n\n"
                "coco_result = pipe.detect(scene, threshold=threshold, nms_threshold=nms_threshold)\n"
                "for det in coco_result['detections']:\n"
                "    print(f\"{{det['label']:>14s}} {{det['score']:.3f}}  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n\n"
                "coco_report = evaluation_report(coco_result, references, sample_kind='synthetic')\n"
                "print({{'verdict': coco_report['verdict'], 'n_detections': coco_report['n_detections']}})\n"
                "for metric in coco_report['metrics']:\n"
                "    print(f\"  {{metric['reference']:>18s}}  box_iou {{metric['value']:.3f}}  same-label detections {{metric['n_detected_same_label']}}\")\n"
                "hits = sum(1 for m in coco_report['metrics'] if m['value'] >= 0.5)\n"
                "print(f'{{hits}}/{{len(coco_report[\"metrics\"])}} drawn objects matched at IoU >= 0.5')\n"
                "scene"
            ),
        },
        # ---------------------------------------------------------------- 5. channel order + degenerate
        {
            "md": (
                "**What to notice (Section 4):** the detections with their scores, one `box_iou` per drawn object, and the matched count.\n\n"
                "<details><summary>Check your reasoning</summary>Four of five, in every recorded run: stop sign, traffic light, clock and bench "
                "at IoU 0.91–0.96. The drawn football is not proposed as a `sports ball` at all; the model offers `kite` and a second `clock` on "
                "that box. A flat, drawn ball has little of the texture a photographed one has. The report keeps the miss as a zero instead of "
                "dropping the reference, which is what makes the other four numbers believable.</details>\n\n"
                "## 5. Two checks worth running once: channel order, and what it says about nothing · [Evaluation practice]\n\n"
                "**Channel order.** Upstream reads images with `cv2.imread`, which yields **BGR**, and feeds that array to the network as raw "
                "0–255 floats. It is an easy thing to get silently wrong, because RGB input still produces plausible-looking detections — it just "
                "produces worse ones. The cell below feeds the identical letterboxed tensor with the channels flipped so you can see the "
                "difference rather than take it on trust.\n\n"
                "**Degenerate inputs.** A detector should be asked what it does with a blank page and with pure noise, because a model that "
                "invents confident objects on structure-free input will invent them on your input too.\n\n"
                "**Predict before running:** with RGB instead of BGR, will the model find the same objects with the same scores, the same "
                "objects with lower scores, or fewer objects? And how many boxes will it return for a blank page at the demo thresholds?"
            ),
            "code": (
                "chw, ratio = preprocess(scene)\n"
                "rgb_tensor = torch.from_numpy(chw[::-1].copy()).unsqueeze(0).to(pipe.device)\n"
                "with torch.no_grad():\n"
                "    rgb_raw = pipe.model(rgb_tensor)\n"
                "rgb_kept = postprocess(rgb_raw, len(LABELS), conf_thre=threshold, nms_thre=nms_threshold)[0]\n"
                "rgb_detections = sorted(\n"
                "    ([LABELS[int(row[6])], float(row[4] * row[5])] for row in (rgb_kept.tolist() if rgb_kept is not None else [])),\n"
                "    key=lambda pair: -pair[1],\n"
                ")\n"
                "print('BGR (upstream order):', [(d['label'], round(d['score'], 3)) for d in coco_result['detections']])\n"
                "print('RGB (the mistake)   :', [(label, round(score, 3)) for label, score in rgb_detections])\n\n"
                "degenerate = {{}}\n"
                "for name, image in (('blank', blank_scene()), ('noise', noise_scene(0))):\n"
                "    demo = pipe.detect(image, threshold=threshold, nms_threshold=nms_threshold)['detections']\n"
                "    lenient = pipe.detect(image, threshold=EVAL_DETECTION_THRESHOLD, nms_threshold=EVAL_NMS_THRESHOLD)['detections']\n"
                "    degenerate[name] = {{'at_demo_thresholds': len(demo), 'at_evaluation_thresholds': len(lenient),\n"
                "                        'top': [(d['label'], round(d['score'], 3)) for d in lenient[:3]]}}\n"
                "print(json.dumps(degenerate, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 6. dataset + validate
        {
            "md": (
                "**What to notice (Section 5):** the BGR and RGB lines side by side, and the two degenerate counts.\n\n"
                "<details><summary>Check your reasoning</summary>Fewer and weaker. On the reference machine RGB input returned 5 detections "
                "instead of 6: the stop sign fell from 0.962 to 0.839, the bench disappeared, and a spurious second clock rose from 0.309 to 0.441. "
                "Nothing crashes and the output still looks plausible, which is exactly why the channel order is easy to get silently wrong. The "
                "blank and noise images give no boxes at the demo thresholds; any box there would be a false positive by construction.</details>\n\n"
                "## 6. Sample data for adaptation, and the validation stage · [Evaluation practice]\n\n"
                "The detector above knows 80 COCO classes. Suppose you need three classes it does not have. `sign_dataset` draws a deterministic "
                "40-image labelled set over `SIGN_CLASSES` — `stop-sign`, `yield-sign`, `speed-limit-sign` — with 1–3 signs per image at jittered "
                "positions, sizes and background tints, and returns exact boxes because it knows where it drew them.\n\n"
                "**Keep the two vocabularies apart.** COCO's `stop sign` (with a space) is a class the *pretrained* model predicts; "
                "`stop-sign` (hyphenated) is a class in *your* vocabulary that only exists after adaptation. They are not the same label and the "
                "notebook never treats them as interchangeable. `yield-sign` and `speed-limit-sign` have no COCO counterpart at all.\n\n"
                "`validate_dataset` is the validation stage for this path and applies exactly the ceilings `finetune` applies, so a dataset it "
                "accepts cannot be refused later: record shape, box geometry inside the image, labels drawn from the vocabulary, at most 50 boxes "
                "per image (the head's label tensor width), at most 200 images, at most 20 epochs. It returns a dataset manifest, and it reports "
                "a class with no boxes as a **finding rather than an error** — that dataset is trainable, but that class's head outputs will stay "
                "untrained and you should know before you spend the compute.\n\n"
                "This is drawn data. A model fine-tuned on it learns these renderings; that is what a bounded tutorial adaptation is for, and it "
                "is why none of the numbers later is a claim about photographs of real signs."
            ),
            "code": (
                'N_IMAGES = 40  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'EPOCHS = 6  # @param {{type:"integer"}}\n\n'
                "records = sign_dataset(N_IMAGES, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SIGN_CLASSES, epochs=EPOCHS)\n"
                "print(json.dumps({{k: v for k, v in dataset_manifest.items() if k != 'schema'}}, indent=2))\n"
                "print('\\nrecord shape expected by finetune:', dataset_manifest['schema']['record'])\n"
                "print('COCO classes the pretrained model knows :', [c for c in LABELS if 'sign' in c or c == 'clock'])\n"
                "print('adaptation vocabulary (not COCO)        :', list(SIGN_CLASSES))\n\n"
                "preview = Image.new('RGB', (480, 320))\n"
                "for index, record in enumerate(records[:6]):\n"
                "    preview.paste(record['image'].resize((160, 160)), (160 * (index % 3), 160 * (index // 3)))\n"
                "print('\\nfirst six images, and the labels of the first:', records[0]['labels'])\n"
                "preview"
            ),
        },
        # ---------------------------------------------------------------- 7. split + baseline
        {
            "md": (
                "## 7. Split, re-head, and measure the baseline *before* adapting · [Evaluation practice]\n\n"
                "`split_records` is a seeded permutation into a training part and a held-out part. The split is yours, not the model's: it "
                "happens before any weight is touched and the held-out records are never shown to `finetune`, so the score in Section 9 is a "
                "score on data the adapted model has not seen.\n\n"
                "`from_pretrained(class_names=...)` builds the same YOLOX-S and loads the same verified checkpoint, but rebuilds the "
                "classification branch for your vocabulary. It prints exactly which tensors it could not transfer — the three `head.cls_preds` "
                "weight/bias pairs, one per feature level — and everything else, including the whole backbone and the box and objectness heads, "
                "comes from the COCO checkpoint. The random initialisation is seeded, because those layers are the only untrained weights in the "
                "model and they are precisely what the baseline measures.\n\n"
                "**The baseline is not zero, and that is the interesting part.** The classification head is random, but the box and objectness "
                "heads are COCO-trained and already know how to localise a sign-shaped thing, so the model scores some average precision by "
                "accident. Measuring it is what lets you say later that the fine-tune did something, rather than that a detector produced boxes.\n\n"
                "**Predict before running:** will the baseline AP50 be near 0, near 0.2 or near 0.5? Which class could score by chance?"
            ),
            "code": (
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 0  # @param {{type:"integer"}}\n\n'
                "train_records, held_out = split_records(records, holdout=HOLDOUT, seed=SEED)\n"
                "print({{'train': len(train_records), 'held_out': len(held_out),\n"
                "       'train_boxes': sum(len(r['boxes']) for r in train_records),\n"
                "       'held_out_boxes': sum(len(r['boxes']) for r in held_out)}})\n\n"
                "adapter = YoloxDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "print({{'class_names': list(adapter.class_names), 'device': adapter.device, 'adapted': adapter.adapted}})\n"
                "print('tensors that could not transfer from the COCO checkpoint:')\n"
                "for name in adapter.reinitialised:\n"
                "    print('   ', name)\n\n"
                "baseline = adapter.evaluate(held_out)\n"
                "print(json.dumps({{'ap': round(baseline['ap'], 4), 'ap50': round(baseline['ap50'], 4),\n"
                "                  'per_class_ap50': {{k: round(v, 4) for k, v in baseline['per_class_ap50'].items()}},\n"
                "                  'n_references': baseline['n_references'], 'max_detections': baseline['max_detections']}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 8. finetune
        {
            "md": (
                "**What to notice (Section 7):** the split sizes, the six re-initialised `head.cls_preds` tensors, and the baseline scores.\n\n"
                "<details><summary>Check your reasoning</summary>Every recorded run measured AP 0.2083 and AP50 0.2222, with per-class AP50 "
                "stop 0.0, yield 0.0 and speed-limit 0.667. The box and objectness heads already find sign-shaped things; the random class head "
                "happens to rank the speed-limit boxes well. A non-zero baseline is the floor the fine-tune must beat, not something it "
                "achieved.</details>\n\n"
                "## 8. The bounded SimOTA fine-tune · [Concept]\n\n"
                "This is the cell that makes the notebook an `E2E` tutorial rather than an inference demo, and it runs in the default path.\n\n"
                "The loss and the label assignment are **upstream's**, in the carried `yolo_head` module: put the head in training mode, hand it "
                "images and a `(class, cx, cy, w, h)` target tensor, and it runs SimOTA — building an IoU-and-classification cost between every "
                "ground-truth object and every candidate anchor point, picking a dynamic number of positives per object, and returning the IoU, "
                "objectness and classification terms already weighted. What this repository owns is the bounded loop around it: the target "
                "conversion, batching, the optimiser, the seed and the ceilings.\n\n"
                "Three defaults are worth understanding because they were chosen by measurement, not taste:\n\n"
                "- **The backbone is frozen.** Only the head trains (~1.9 M of 9.0 M parameters). It is several times faster per step, and the "
                "repository measured that unfreezing it at this learning rate *collapses* the model to AP 0.0 — a handful of gradient steps on 30 "
                "small images is enough to destroy COCO-pretrained features. The frozen backbone is also kept in eval mode so its BatchNorm "
                "running statistics are not quietly rewritten by tutorial batches.\n"
                "- **Six epochs at learning rate 1e-3.** A grid over epochs, learning rate and freezing put this at held-out AP50 1.0 against "
                "0.355 at three epochs.\n"
                "- **The L1 box term stays off.** Upstream enables it only for the last 15 of 300 epochs; switching it on for a six-epoch run "
                "would change the loss scale for no benefit. You will see `l1_loss` report 0.0 throughout, and that is correct.\n\n"
                "Watch `total_loss` fall and `num_fg` — the number of anchor points SimOTA assigned as positives per image — stay stable. A "
                "`num_fg` collapsing toward zero would mean the assignment had stopped finding anything to match.\n\n"
                "**Every run of this cell starts from the verified base.** The cell first rebuilds `adapter` with "
                "`from_pretrained(class_names=SIGN_CLASSES, seed=SEED)` — the same COCO weights and, under the same seed, the same "
                "re-initialised classification layers the Section 7 baseline measured — so re-running it after changing a field trains once from "
                "the start, never on top of an already adapted model.\n\n"
                "**Predict before running:** over six epochs, roughly how far will `total_loss` fall, and will `num_fg` stay near 8?"
            ),
            "code": (
                'LEARNING_RATE = 1e-3  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 2  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = True  # @param {{type:"boolean"}}\n\n'
                "import time\n\n"
                "# YXD-m2: start every run of this cell from the verified base and the same seeded head as the baseline.\n"
                "adapter = YoloxDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "finetune_started = time.perf_counter()\n"
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_backbone=FREEZE_BACKBONE,\n"
                "    progress=lambda row: print(\n"
                "        f\"epoch {{row['epoch']}}/{{EPOCHS}}  steps {{row['steps']}}  total {{row['total_loss']:.4f}}  \"\n"
                "        f\"iou {{row['iou_loss']:.4f}}  conf {{row['conf_loss']:.4f}}  cls {{row['cls_loss']:.4f}}  \"\n"
                "        f\"l1 {{row['l1_loss']:.4f}}  num_fg {{row['num_fg']:.2f}}\"\n"
                "    ),\n"
                ")\n"
                "print(json.dumps({{'optimizer': run['optimizer'], 'loss': run['loss'], 'use_l1': run['use_l1'],\n"
                "                  'freeze_backbone': run['freeze_backbone'],\n"
                "                  'trainable_parameters': run['trainable_parameters'],\n"
                "                  'total_parameters': run['total_parameters'],\n"
                "                  'epochs': run['epochs'], 'batch_size': run['batch_size'],\n"
                "                  'learning_rate': run['learning_rate'], 'seed': run['seed']}}, indent=2))\n"
                "first, last = run['history'][0]['total_loss'], run['history'][-1]['total_loss']\n"
                "FINETUNE_SECONDS = round(time.perf_counter() - finetune_started, 1)\n"
                "print(f'total_loss {{first:.4f}} -> {{last:.4f}} over {{run[\"epochs\"]}} epochs in {{FINETUNE_SECONDS}} s')"
            ),
        },
        # ---------------------------------------------------------------- 9. evaluate
        {
            "md": (
                "**What to notice (Section 8):** one line per epoch, `l1` at 0.0, `num_fg` steady, and the trainable-parameter count.\n\n"
                "<details><summary>Check your reasoning</summary>On the reference machine `total_loss` fell from 6.288 to 1.662 over six epochs "
                "and `num_fg` stayed between 7.9 and 8.8: SimOTA kept finding about eight positives per image. Only 1.9 M of 8.9 M parameters "
                "trained. A falling loss shows the head fits the 30 training images; whether that transfers is Section 9's question.</details>\n\n"
                "## 9. Evaluate on the held-out split · [Evaluation practice]\n\n"
                "The same `evaluate` call as the baseline, on the same held-out records, with the same thresholds — so the two numbers are "
                "comparable and the only thing that changed is the weights.\n\n"
                "`ap50` is average precision at IoU 0.50; `ap` is COCO's primary metric, AP@[.50:.95], the mean over ten IoU thresholds from 0.50 "
                "to 0.95 — it is always lower, because it demands progressively tighter boxes. Evaluation uses the *evaluation* thresholds "
                "(`0.01 / 0.65`) rather than the demo pair, since average precision rewards recall, and caps detections at 100 per image as COCO "
                "does.\n\n"
                "What this number is: evidence that a bounded fine-tune on 30 drawn images moved a held-out score. What it is not: these are "
                "not a detection benchmark. The held-out split is ten images from the same generator with the same three shapes, so AP50 reaching 1.0 says the "
                "task is easy and the adaptation worked — not that the model would find a real sign in a real photograph. The "
                "AP@[.50:.95] figure, which is well below 1.0, is the more informative one: the boxes are right but not perfectly tight.\n\n"
                "Every run of Sections 8–9 is appended to `run_history`, so the activity at the end (Section 14) prints both settings side by "
                "side.\n\n"
                "**Predict before running:** after six epochs, will held-out AP50 be near 0.5, near 0.9 or 1.0? And AP@[.50:.95]?"
            ),
            "code": (
                "adapted = adapter.evaluate(held_out)\n"
                "print(json.dumps({{'ap': round(adapted['ap'], 4), 'ap50': round(adapted['ap50'], 4), 'ap75': round(adapted['ap75'], 4),\n"
                "                  'per_class_ap50': {{k: round(v, 4) for k, v in adapted['per_class_ap50'].items()}},\n"
                "                  'threshold': adapted['threshold'], 'nms_threshold': adapted['nms_threshold'],\n"
                "                  'max_detections': adapted['max_detections'],\n"
                "                  'n_images': adapted['n_images'], 'n_references': adapted['n_references']}}, indent=2))\n"
                "print()\n"
                "print(f\"{{'metric':<10s}} {{'baseline':>10s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in ('ap', 'ap50'):\n"
                "    before_value, after_value = baseline[key], adapted[key]\n"
                "    print(f'{{key:<10s}} {{before_value:>10.4f}} {{after_value:>10.4f}} {{after_value - before_value:>+10.4f}}')\n"
                "print('\\nimplementation note:', adapted['implementation'])\n"
                "run_history = globals().get('run_history', [])\n"
                "run_history.append({{'freeze_backbone': FREEZE_BACKBONE, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'ap': round(adapted['ap'], 4), 'ap50': round(adapted['ap50'], 4), 'finetune_seconds': FINETUNE_SECONDS}})\n"
                "for number, row in enumerate(run_history, start=1):\n"
                "    print(f'run {{number}}:', row)"
            ),
        },
        # ---------------------------------------------------------------- 10. new-data inference
        {
            "md": (
                "**What to notice (Section 9):** the adapted scores beside the baseline.\n\n"
                "<details><summary>Check your reasoning</summary>AP50 1.0 in every recorded run; AP@[.50:.95] 0.811 on the reference CPU and "
                "0.840 on a Kaggle T4. The AP50 is saturated: ten images from the training generator are too easy to separate a good adapter from "
                "a perfect one. The two AP values differ by device with identical AP50, which shows how little ten images resolve — the tighter "
                "metric moves with tiny numerical differences.</details>\n\n"
                "## 10. Inference on new data · [Concept]\n\n"
                "Held-out images were drawn from the same seed as the training set. These come from a different seed entirely, so their layouts, "
                "sizes, tints and sign choices were never part of the split at all — the closest a synthetic tutorial gets to new data.\n\n"
                'The detections use the **demo thresholds** again, because this is the "what would I actually deploy" view rather than the '
                "scoring view. Each detection is checked against the drawn truth with `box_iou` on the same label.\n\n"
                "**Predict before running:** how many of the five signs on the three new images will be found with the right label?"
            ),
            "code": (
                'NEW_DATA_SEED = 99  # @param {{type:"integer"}}\n\n'
                "new_records = sign_dataset(3, seed=NEW_DATA_SEED)\n"
                "new_data_rows = []\n"
                "for index, record in enumerate(new_records):\n"
                "    out = adapter.detect(record['image'], threshold=threshold, nms_threshold=0.45)\n"
                "    ious = []\n"
                "    for box, label in zip(record['boxes'], record['labels'], strict=True):\n"
                "        same_label = [d for d in out['detections'] if d['label'] == label]\n"
                "        ious.append(round(max((box_iou(d['box'], box) for d in same_label), default=0.0), 3))\n"
                "    row = {{'image': index, 'truth': record['labels'],\n"
                "           'detections': [(d['label'], round(d['score'], 3)) for d in out['detections']],\n"
                "           'same_label_iou': ious}}\n"
                "    new_data_rows.append(row)\n"
                "    print(json.dumps(row))\n\n"
                "contact = Image.new('RGB', (640, 214))\n"
                "for index, record in enumerate(new_records):\n"
                "    contact.paste(record['image'].resize((213, 213)), (213 * index, 0))\n"
                "contact"
            ),
        },
        # ---------------------------------------------------------------- 11. export + fresh reload
        {
            "md": (
                "**What to notice (Section 10):** one row per image with the truth, the detections and the same-label IoU.\n\n"
                "<details><summary>Check your reasoning</summary>All five, with the right labels and IoU 0.89–0.95, and no spurious boxes, on the "
                "reference machine. That is encouraging for drawn signs from an unseen seed, but they come from the same generator: same shapes, "
                "same colours, same rendering. It is not evidence about photographs.</details>\n\n"
                "## 11. Export the artifact, then reload it as if from a cold start · [Engineering]\n\n"
                "`save_artifact` writes one file: the adapted `state_dict` plus the metadata needed to rebuild the model — the format tag, the "
                "pinned model identity, the upstream code revision, the class vocabulary, the architecture constants and the digest of the base "
                "weights it started from. It is a plain `torch.save` of tensors and simple values, with no archive to extract and nothing that "
                "requires `weights_only=False` to read back, so there is no unpacking step to make safe.\n\n"
                "`load_artifact` is the fresh-reload check: it rebuilds the model from the file alone, without reference to the `adapter` object "
                "still in memory, and refuses an artifact whose format tag or pinned identity does not match this package. The cell then re-scores "
                "the held-out split with the reloaded model and asserts the numbers are identical — a reload that quietly lost the adaptation "
                "would show up here as a different AP, and the assertion would stop the notebook.\n\n"
                'The last part of the cell tampers with a copy of the artifact and confirms it is refused, because "it loads" is only evidence '
                "if something that should not load is also tried."
            ),
            "code": (
                "import os\n"
                "from pathlib import Path\n\n"
                "OUTPUTS = Path('outputs')\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "artifact_path = OUTPUTS / ARTIFACT_FILE\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='standalone tutorial run')\n"
                "print(json.dumps({{k: v for k, v in descriptor.items() if k != 'base_state_digest'}}, indent=2))\n\n"
                "reloaded = YoloxDetectionPipeline.load_artifact(artifact_path)\n"
                "print({{'source': reloaded.source, 'class_names': list(reloaded.class_names), 'adapted': reloaded.adapted}})\n"
                "reloaded_metrics = reloaded.evaluate(held_out)\n"
                "print({{'reloaded_ap': round(reloaded_metrics['ap'], 4), 'reloaded_ap50': round(reloaded_metrics['ap50'], 4)}})\n"
                "assert abs(reloaded_metrics['ap'] - adapted['ap']) < 1e-9, 'the reloaded artifact does not reproduce the adapted score'\n"
                "assert abs(reloaded_metrics['ap50'] - adapted['ap50']) < 1e-9\n"
                "print('fresh reload reproduces the adapted scores exactly')\n\n"
                "tampered = OUTPUTS / 'tampered-artifact.pt'\n"
                "payload = torch.load(artifact_path, map_location='cpu', weights_only=True)\n"
                "payload['format'] = 'not-a-dimer-artifact/9'\n"
                "torch.save(payload, tampered)\n"
                "try:\n"
                "    YoloxDetectionPipeline.load_artifact(tampered)\n"
                "except ValueError as exc:\n"
                "    print('tampered artifact refused ->', exc)\n"
                "else:\n"
                "    raise AssertionError('a tampered artifact was accepted')\n"
                "finally:\n"
                "    tampered.unlink(missing_ok=True)"
            ),
        },
        # ---------------------------------------------------------------- 12. export outputs
        {
            "md": (
                "## 12. Machine-readable outputs and provenance · [Engineering]\n\n"
                "Everything the notebook established, written to `outputs/` as JSON beside the artifact: the pinned identity and manifest digest, "
                "the runtime, the input and dataset manifests, the COCO detections and their per-object IoU, the baseline and adapted metrics with "
                "the exact request that produced them, the new-data rows, and the artifact descriptor. A later reader can tell what was measured, "
                "on what, with which weights, without rerunning anything."
            ),
            "code": (
                "import platform\n\n"
                "export = {{\n"
                "    'notebook': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'license': MODEL_LICENSE,\n"
                "              'upstream_code_revision': UPSTREAM_CODE_REVISION, 'release_asset': RELEASE_ASSET_URL,\n"
                "              'manifest_sha256': [entry['sha256'] for entry in MANIFEST['files']],\n"
                "              'depth': MODEL_DEPTH, 'width': MODEL_WIDTH, 'input_size': list(INPUT_SIZE)}},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__,\n"
                "                'torchvision': torchvision.__version__, 'device': adapter.device,\n"
                "                'cuda': torch.cuda.is_available(), 'precision': 'float32'}},\n"
                "    'coco_demonstration': {{'input_manifest': input_manifest, 'detections': coco_result['detections'],\n"
                "                           'evaluation': coco_report}},\n"
                "    'degenerate_inputs': degenerate,\n"
                "    'channel_order': {{'bgr': [(d['label'], round(d['score'], 4)) for d in coco_result['detections']],\n"
                "                      'rgb': [(label, round(score, 4)) for label, score in rgb_detections]}},\n"
                "    'adaptation': {{'dataset': dataset_manifest, 'split': {{'train': len(train_records), 'held_out': len(held_out)}},\n"
                "                   'run': run, 'baseline': baseline, 'adapted': adapted,\n"
                "                   'new_data': new_data_rows, 'artifact': descriptor}},\n"
                "}}\n"
                "path = OUTPUTS / '{stem}_results.json'\n"
                "with open(path, 'w', encoding='utf-8') as handle:\n"
                "    json.dump(export, handle, indent=2, default=str)\n"
                "print({{'written': str(path), 'bytes': path.stat().st_size, 'artifact': str(artifact_path)}})\n"
                "print(sorted(p.name for p in OUTPUTS.iterdir()))"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: your own image, or your own labelled dataset · [Engineering]\n\n"
                "Both branches are off by default so the cell above completes a full `Run all` without stopping for an upload.\n\n"
                "**`USE_BYOD_IMAGE`** runs one image — from `BYOD_IMAGE_PATH`, or exactly one file from the upload dialog, refused by name if "
                "Pillow cannot decode it — through the same `validate_inputs` → `detect` → `evaluation_report` contract as the "
                "sample, with the pretrained COCO model. There are no reference boxes for your image, so the evaluation report will say "
                "`not-measurable` — which is the correct answer, not a failure.\n\n"
                "**`USE_BYOD_DATASET`** is the one that matters for an adaptation profile. Supply your own labelled records and they go through "
                "the *same* local stages the sample did: validate, split, baseline, fine-tune, evaluate. It does not degrade into inference-only "
                "just because the data is yours (NOTEBOOK_SPEC 2.2 DAT14). Put the images and an `annotations.json` in a directory and set "
                "`BYOD_DATASET_DIR` to it (any Jupyter runtime), or leave it empty and select the images and `annotations.json` together in the "
                "upload dialog. `annotations.json` is a list of `{{'file': 'name.png', 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...]}}` "
                "objects with xyxy boxes in that image's pixels; `read_detection_records` refuses a path outside the directory, a missing file "
                "and an undecodable image by name, and `validate_dataset` then applies the same ceilings as for the sample. `BYOD_CLASS_NAMES` is a "
                "comma-separated vocabulary; leave it empty to use the sorted set of labels in the annotations. At least **8 records** are needed, "
                "so the 25 % held-out split has two or more images; for a usable measurement aim for dozens. The cell then repeats Sections 7–9 on "
                "your data, exports the artifact, reloads it and asserts identical held-out scores as in Section 11, and writes "
                "`outputs/byod_yolox_result.json`. Labelling images is outside this notebook's scope — export boxes in xyxy pixel coordinates "
                "from whatever tool you use.\n\n"
                "The cell rejects one deliberately incompatible input so you can see what a refusal looks like before you trust an acceptance."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'BYOD_IMAGE_PATH = ""  # @param {{type:"string"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_DATASET_DIR = ""  # @param {{type:"string"}}\n'
                'BYOD_CLASS_NAMES = ""  # @param {{type:"string"}}\n\n'
                "# What a refusal looks like, always run: the validator names the first violated ceiling.\n"
                "for description, thunk in (\n"
                "    ('an image that is not a PIL image', lambda: validate_inputs('/path/to/image.png')),\n"
                "    ('a box outside its image', lambda: validate_dataset(\n"
                "        [{{'image': blank_scene(), 'boxes': [[0, 0, 9999, 10]], 'labels': [SIGN_CLASSES[0]]}}], SIGN_CLASSES)),\n"
                "):\n"
                "    try:\n"
                "        thunk()\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print(f'refused {{description}} -> {{type(exc).__name__}}: {{exc}}')\n\n"
                "BYOD_DIR = OUTPUTS / 'byod'\n\n\n"
                "def _upload(target):\n"
                "    from google.colab import files  # type: ignore[import-not-found]\n\n"
                "    uploaded = files.upload()\n"
                "    target.mkdir(parents=True, exist_ok=True)\n"
                "    for upload_name, upload_data in uploaded.items():\n"
                "        (target / Path(upload_name).name).write_bytes(upload_data)\n"
                "    return uploaded\n\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    if BYOD_IMAGE_PATH:\n"
                "        name, data = Path(BYOD_IMAGE_PATH).name, Path(BYOD_IMAGE_PATH).read_bytes()\n"
                "    else:\n"
                "        uploaded = _upload(BYOD_DIR / 'image')\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError(f'Upload exactly one image (received {{len(uploaded)}}); or set BYOD_IMAGE_PATH and run this cell again.')\n"
                "        ((name, data),) = uploaded.items()\n"
                "    byod_image = load_byod_image(name, data)\n"
                "    print(validate_inputs(byod_image, threshold=threshold, nms_threshold=nms_threshold, names=[name])['verdict'])\n"
                "    byod_result = pipe.detect(byod_image, threshold=threshold, nms_threshold=nms_threshold)\n"
                "    for det in byod_result['detections'][:20]:\n"
                "        print(f\"{{det['label']:>14s}} {{det['score']:.3f}}\")\n"
                "    print(evaluation_report(byod_result, None, sample_kind='byod')['verdict'])\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True (and optionally BYOD_IMAGE_PATH) and re-run this cell.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    if BYOD_DATASET_DIR:\n"
                "        dataset_dir = Path(BYOD_DATASET_DIR)\n"
                "    else:\n"
                "        dataset_dir = BYOD_DIR / 'dataset'\n"
                "        _upload(dataset_dir)\n"
                "    byod_records = read_detection_records(dataset_dir)\n"
                "    if len(byod_records) < 8:\n"
                "        raise ValueError(f'the BYOD dataset has {{len(byod_records)}} records; at least 8 are needed so the held-out split has 2 or more images')\n"
                "    byod_names = [n.strip() for n in BYOD_CLASS_NAMES.split(',') if n.strip()] or sorted({{label for r in byod_records for label in r['labels']}})\n"
                "    byod_manifest = validate_dataset(byod_records, byod_names, epochs=EPOCHS)\n"
                "    print(json.dumps({{k: v for k, v in byod_manifest.items() if k != 'schema'}}, indent=2))\n"
                "    byod_train, byod_held = split_records(byod_records, holdout=HOLDOUT, seed=SEED)\n"
                "    byod_pipe = YoloxDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_names, seed=SEED)\n"
                "    byod_baseline = byod_pipe.evaluate(byod_held)\n"
                "    print('baseline:', {{k: round(byod_baseline[k], 4) for k in ('ap', 'ap50')}})\n"
                "    byod_run = byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_backbone=FREEZE_BACKBONE)\n"
                "    byod_adapted = byod_pipe.evaluate(byod_held)\n"
                "    print('adapted :', {{k: round(byod_adapted[k], 4) for k in ('ap', 'ap50')}})\n"
                "    byod_artifact = OUTPUTS / 'byod-adapter.pt'\n"
                "    byod_descriptor = byod_pipe.save_artifact(byod_artifact, notes='BYOD tutorial run')\n"
                "    byod_reloaded_metrics = YoloxDetectionPipeline.load_artifact(byod_artifact).evaluate(byod_held)\n"
                "    assert abs(byod_reloaded_metrics['ap'] - byod_adapted['ap']) < 1e-9, 'the reloaded BYOD artifact does not reproduce the adapted score'\n"
                "    assert abs(byod_reloaded_metrics['ap50'] - byod_adapted['ap50']) < 1e-9\n"
                "    byod_reload_check = {{'reloaded_ap': byod_reloaded_metrics['ap'], 'reloaded_ap50': byod_reloaded_metrics['ap50'], 'identical': True}}\n"
                "    print('fresh reload reproduces the BYOD adapted scores exactly')\n"
                "    byod_export = {{'notebook': NOTEBOOK_SOURCE, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'dataset_dir': str(dataset_dir),\n"
                "                   'class_names': byod_names, 'dataset': {{k: v for k, v in byod_manifest.items() if k != 'schema'}},\n"
                "                   'split': {{'train': len(byod_train), 'held_out': len(byod_held), 'holdout': HOLDOUT, 'seed': SEED}},\n"
                "                   'run': byod_run, 'baseline': byod_baseline, 'adapted': byod_adapted, 'artifact': byod_descriptor, 'reload_check': byod_reload_check}}\n"
                "    with open(OUTPUTS / 'byod_yolox_result.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump(byod_export, handle, indent=2, default=str)\n"
                "    print({{'written': str(OUTPUTS / 'byod_yolox_result.json')}})\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True and BYOD_DATASET_DIR (or upload), then re-run this cell.')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**What this notebook established, in this runtime.** The pinned `yolox_s.pth` release asset was downloaded, digest-verified against a "
        "committed SHA-256 before anything unpickled it, and loaded into YOLOX-S built from vendored upstream code. The pretrained detector "
        "found four of the five drawn COCO objects with per-object IoU around 0.91–0.96 and missed the fifth entirely. A three-class vocabulary "
        "that does not exist in COCO was adapted onto it by a bounded SimOTA fine-tune of the head alone, scored against a held-out split with "
        "COCO-style average precision before and after, run on images from an unseen seed, exported as a single artifact and reloaded from that "
        "artifact to identical scores.\n\n"
        "**What it did not establish.** Nothing here is a benchmark and nothing here is about photographs. Both datasets are drawn in code with "
        "Pillow: flat colours, hard edges, no lighting, no occlusion, no motion blur, no scale variety beyond the jitter the generator applies. "
        "A held-out AP50 of 1.0 on ten images from the same generator says the task is easy and the plumbing works; it is not evidence about real "
        "signs, and the much lower AP@[.50:.95] is the more honest summary — the boxes are right, not tight. Upstream's published 40.5 AP for "
        "YOLOX-S on COCO test-dev is neither reproduced nor checked here, and the average-precision helper carried in Section 2 is a small "
        "faithful implementation without pycocotools' area ranges, crowd handling or official matching, so its numbers are not comparable to "
        "published COCO results.\n\n"
        "**The misses are the useful part.** The drawn football is never proposed as a `sports ball`; the model offers `kite` and a second `clock` "
        "on that box instead, and the evaluation report records a zero rather than quietly dropping the reference. Earlier probing found "
        "something sharper: a plain white disc with two hands is not read as a clock at all, and drawing it that way made the *ball* become the "
        "clock — the numerals in the sample's clock face are there because of that. Treat a detector's confidence on rendered graphics as a "
        "statement about rendered graphics.\n\n"
        "**If you adapt this to your own data.** The three defaults that were chosen by measurement will not automatically transfer. The frozen "
        "backbone is right for a few dozen images and a few epochs; with real data and real compute, unfreeze it and lower the learning rate — the "
        "repository measured a full fine-tune at 1e-3 collapsing to AP 0.0, which is what a too-large step on pretrained features looks like. Six "
        "epochs is a tutorial budget, not a recipe. And the thresholds are yours: the demo pair for looking at results, the evaluation pair for "
        "scoring, neither calibrated for anything.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set and the "
        "pinned checkpoint together reproduce these stages in a fresh runtime, without the repository being cloned or installed and without "
        "any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the adapted model "
        "generalises beyond the drawn data it was fitted to.\n\n"
        "**Reproducibility.** Every random choice is seeded — dataset generation, the split, the re-headed classification layers and the training "
        "shuffle — and the seeds are form parameters at the top of their cells. Precision is float32 on CPU and GPU alike; no autocast, no "
        "quantisation, no compiled kernels. Re-running this notebook unchanged in an equivalent runtime should reproduce the numbers; changing "
        "`DATASET_SEED` or `SEED` will change them, which is a useful thing to do once to see how much of the result is the seed.\n\n"
        "## 14. Activity: change one thing — unfreeze the backbone · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**. It changes nothing unless you do it.\n\n"
        "1. **Predict:** with `FREEZE_BACKBONE = False` the whole network trains at the same learning rate. Will held-out AP rise, stay or "
        "fall? Will the fine-tune take longer?\n"
        "2. **Change:** in Section 8 set `FREEZE_BACKBONE = False`. Change nothing else.\n"
        "3. **Run:** select the Section 8 cell and choose *Runtime → Run after*. Section 8 rebuilds `adapter` from the verified base with the "
        "same seeded head, so this is one clean run with the new setting, not a second run on top of the first. Sections 9–12 overwrite the "
        "artifact and the outputs (download `outputs/` first if you want to keep them).\n"
        "4. **Observe:** Section 9 prints `run_history` with both runs: AP, AP50 and the fine-tune seconds. Compare `num_fg` and the trainable "
        "parameters in Section 8.\n"
        "5. **Explain:** in one sentence, why can training more of a pretrained network make it worse on 30 images?\n\n"
        "<details><summary>Check your reasoning</summary>The repository measured this exact change: with the backbone unfrozen at 1e-3 the "
        "held-out AP and AP50 collapsed to 0.0, and the run took about 4.5× longer (91.3 s against 20.2 s in its grid). A step size that suits a "
        "new head is far too large for COCO-pretrained features; a few dozen gradient steps on 30 drawn images destroy what the backbone knew, "
        "and the head has nothing left to read. With real data you would unfreeze with a much smaller learning rate, and check on a held-out "
        "split, as here.</details>\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle or a Linux x86_64 Jupyter; the lock holds manylinux x86_64 wheels. |\n"
        "| Section 1 fails while downloading, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the Section 1 install cell again; a repeated mismatch means the download is being altered — never edit the digest. |\n"
        "| `holds Python …, not 3.12.12` in Section 1 | an older `dimer_isolated_env/` folder from another notebook version | Delete that folder (or start a fresh runtime) and run the install cell again. |\n"
        "| `The isolated environment's Python process exited` | the worker ran out of memory | Lower `BATCH_SIZE` in Section 8, restart the session and choose *Runtime → Run all*. |\n"
        "| The release-asset download in Section 3 fails, or a `sha256` mismatch | `github.com` is unreachable, or the file changed | Run Section 3 again; a digest mismatch is never loaded — do not edit the manifest. |\n"
        "| Held-out AP 0.0 after Section 8 | `FREEZE_BACKBONE = False` at learning rate 1e-3 (the measured collapse), or a too-large `LEARNING_RATE` | Restore the defaults and run again from Section 8. |\n"
        "| `RuntimeError: Upload exactly one image` in Section 13 | the dialog was cancelled, or several files were chosen | Run the cell again and choose one image, or set `BYOD_IMAGE_PATH`. |\n"
        "| `…: not an image Pillow can decode` or `image … not found` in Section 13 | a corrupt file, or a name in `annotations.json` that is not in the directory | Fix the named file or entry and run Section 13 again. |\n"
        "| `the BYOD dataset has N records; at least 8 are needed` | too few labelled images | Add labelled images; a few dozen give a more useful measurement. |\n"
        "| A `ValueError` naming a record, box or label | `annotations.json` breaks the dataset contract (Section 13) | Fix the named record and run Section 13 again. |\n"
        "| `the reloaded artifact does not reproduce the adapted score` | the artifact did not round-trip | Re-run from Section 8; do not ship the artifact if it repeats. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. On the drawn COCO scene the detector matched ___ of 5 objects; the miss was ___, called ___.\n"
        "2. Feeding RGB instead of BGR changed ___.\n"
        "3. Held-out AP50 went from ___ to ___, and AP@[.50:.95] from ___ to ___; the new-data images gave ___ of 5 correct.\n"
        "4. With `FREEZE_BACKBONE = False` (Section 14), held-out AP ___ and the fine-tune took ___.\n"
        "5. What I would need before trusting this adapted detector on real photographs: ___.\n\n"
        "## References\n\n"
        "- Ge, Z., Liu, S., Wang, F., Li, Z. and Sun, J. (2021). *YOLOX: Exceeding YOLO Series in 2021.* [arXiv:2107.08430](https://arxiv.org/abs/2107.08430) — the anchor-free design, the decoupled head and SimOTA.\n"
        "- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) — the upstream repository, Apache-2.0. The model code carried in Section 2 is vendored from it at a pinned commit; the repository's `docs/UPSTREAM.md` records the revision, the per-file digests and the three edits applied.\n"
        "- [Release `0.1.1rc0`](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) — the immutable tag whose assets carry every published YOLOX checkpoint, including the `yolox_s.pth` this notebook verifies.\n"
        "- Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312) — the 80 classes the pretrained checkpoint predicts, and the average-precision protocol the evaluation helper approximates.\n"
        "- Repository model card: https://github.com/kurtvalcorza/yolox-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/yolox-detection-pipeline`](https://github.com/kurtvalcorza/yolox-detection-pipeline) — this notebook's source repository: the package carried in Section 2, its tests, `MODEL_CARD.md`, and `docs/release-verification.md` with the measured fine-tuning grid behind the defaults used here."
    ),
}
