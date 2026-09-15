"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.0 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "yolox_detection_pipeline",
    "repo_name": "yolox-detection-pipeline",
    "stem": "yolox_detection",
    "notebook_name": "yolox_detection_finetune_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
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
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). **CPU is enough and is the documented default** — this profile is float32 on both CPU and GPU, and CUDA is used automatically when present. For scale: on the repository's Windows CPU venv (Intel Core Ultra 9 275HX) loading takes 2.6 s, one `detect` 0.11 s, and the whole six-epoch fine-tune 25.9 s. A hosted CPU runtime is slower; budget a few minutes for the adaptation cell. The pinned `torch==2.14.0` install and the 72 MB checkpoint are the large downloads.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union measures; roughly what average precision summarises. You do **not** need to know YOLOX internals — SimOTA assignment and the loss are upstream's code, carried and called, not reimplemented here.",
        "- **Data:** everything is drawn in code by the carried `samples` module, so nothing is downloaded and no private data is needed: one 640×640 COCO demonstration scene with reference boxes, and a deterministic 40-image labelled sign dataset for the adaptation. Optional BYOD is gated off by default. Expected BYOD input: for detection, one image decodable by Pillow, sides 16–4096 px; for adaptation, a list of `{{'image': PIL.Image, 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...]}}` records with boxes in that image's own pixels. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, downloads and digest-verifies the pinned release "
        "asset, draws the COCO demonstration scene and scores it, builds and validates the labelled adaptation dataset, splits it, measures the "
        "pre-adaptation baseline on the held-out part, **runs the bounded SimOTA fine-tune**, re-scores the held-out split, detects on unseen "
        "images, exports the adapted artifact, reloads it from disk and confirms the reloaded model scores identically, and writes every result "
        "as JSON with provenance. Nothing is skipped behind a default-off flag, and the path needs no repository clone, no DIMER worker or "
        "service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two BYOD branches are provided and both are optional and off by default. `USE_BYOD_IMAGE` runs your own image through the same "
        "validation and detection contract as the sample. `USE_BYOD_DATASET` takes your own labelled records and runs them through the *same* "
        "local stages the sample used — validate, split, baseline, fine-tune, evaluate — rather than only detecting with them, because this is an "
        "adaptation profile (NOTEBOOK_SPEC 2.0 DAT14, §25.10). The expected record shape and the ceilings are stated in the Prerequisites and "
        "printed by the cell; uploads stay inside this runtime."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. COCO scene
        {
            "md": (
                "## 4. What the pretrained detector does, and where it fails\n\n"
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
                "**Expect one honest failure.** The drawn football is not detected as a `sports ball` at all — the model offers `kite` and a "
                "second `clock` on that box instead. That miss is kept rather than tuned away, and the evaluation report records it as a zero."
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
                "## 5. Two checks worth running once: channel order, and what it says about nothing\n\n"
                "**Channel order.** Upstream reads images with `cv2.imread`, which yields **BGR**, and feeds that array to the network as raw "
                "0–255 floats. It is an easy thing to get silently wrong, because RGB input still produces plausible-looking detections — it just "
                "produces worse ones. The cell below feeds the identical letterboxed tensor with the channels flipped so you can see the "
                "difference rather than take it on trust.\n\n"
                "**Degenerate inputs.** A detector should be asked what it does with a blank page and with pure noise, because a model that "
                "invents confident objects on structure-free input will invent them on your input too. Run it and read the counts; this one is "
                "well-behaved at the demo threshold, and the notebook would say so either way."
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
                "## 6. Sample data for adaptation, and the validation stage\n\n"
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
                "## 7. Split, re-head, and measure the baseline *before* adapting\n\n"
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
                "accident. Measuring it is what lets you say later that the fine-tune did something, rather than that a detector produced boxes."
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
                "## 8. The bounded SimOTA fine-tune\n\n"
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
                "`num_fg` collapsing toward zero would mean the assignment had stopped finding anything to match."
            ),
            "code": (
                'LEARNING_RATE = 1e-3  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 2  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = True  # @param {{type:"boolean"}}\n\n'
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
                "print(f'total_loss {{first:.4f}} -> {{last:.4f}} over {{run[\"epochs\"]}} epochs')"
            ),
        },
        # ---------------------------------------------------------------- 9. evaluate
        {
            "md": (
                "## 9. Evaluate on the held-out split\n\n"
                "The same `evaluate` call as the baseline, on the same held-out records, with the same thresholds — so the two numbers are "
                "comparable and the only thing that changed is the weights.\n\n"
                "`ap50` is average precision at IoU 0.50; `ap` is COCO's primary metric, AP@[.50:.95], the mean over ten IoU thresholds from 0.50 "
                "to 0.95 — it is always lower, because it demands progressively tighter boxes. Evaluation uses the *evaluation* thresholds "
                "(`0.01 / 0.65`) rather than the demo pair, since average precision rewards recall, and caps detections at 100 per image as COCO "
                "does.\n\n"
                "What this number is: evidence that a bounded fine-tune on 30 drawn images moved a held-out score. What it is not: these are "
                "not a detection benchmark. The held-out split is ten images from the same generator with the same three shapes, so AP50 reaching 1.0 says the "
                "task is easy and the adaptation worked — not that the model would find a real sign in a real photograph. The "
                "AP@[.50:.95] figure, which is well below 1.0, is the more informative one: the boxes are right but not perfectly tight."
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
                "print('\\nimplementation note:', adapted['implementation'])"
            ),
        },
        # ---------------------------------------------------------------- 10. new-data inference
        {
            "md": (
                "## 10. Inference on new data\n\n"
                "Held-out images were drawn from the same seed as the training set. These come from a different seed entirely, so their layouts, "
                "sizes, tints and sign choices were never part of the split at all — the closest a synthetic tutorial gets to new data.\n\n"
                'The detections use the **demo thresholds** again, because this is the "what would I actually deploy" view rather than the '
                "scoring view. Each detection is checked against the drawn truth with `box_iou` on the same label."
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
                "## 11. Export the artifact, then reload it as if from a cold start\n\n"
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
                "## 12. Machine-readable outputs and provenance\n\n"
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
                "## 13. Optional: your own image, or your own labelled dataset\n\n"
                "Both branches are off by default so the cell above completes a full `Run all` without stopping for an upload.\n\n"
                "**`USE_BYOD_IMAGE`** runs one uploaded image through the same `validate_inputs` → `detect` → `evaluation_report` contract as the "
                "sample, with the pretrained COCO model. There are no reference boxes for your image, so the evaluation report will say "
                "`not-measurable` — which is the correct answer, not a failure.\n\n"
                "**`USE_BYOD_DATASET`** is the one that matters for an adaptation profile. Supply your own labelled records and they go through "
                "the *same* local stages the sample did: validate, split, baseline, fine-tune, evaluate. It does not degrade into inference-only "
                "just because the data is yours (NOTEBOOK_SPEC 2.0 DAT14). You supply `BYOD_CLASS_NAMES` and a list of records shaped exactly like "
                "`sign_dataset`'s output; the cell prints the dataset manifest and then repeats Sections 7–9 on your data. Labelling images is "
                "outside this notebook's scope — export boxes in xyxy pixel coordinates from whatever tool you use.\n\n"
                "The cell rejects one deliberately incompatible input so you can see what a refusal looks like before you trust an acceptance."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                "BYOD_CLASS_NAMES = ['my-class-a', 'my-class-b']  # @param\n\n"
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
                "if USE_BYOD_IMAGE:\n"
                "    from google.colab import files  # type: ignore[import-not-found]\n\n"
                "    uploaded = files.upload()\n"
                "    name, data = next(iter(uploaded.items()))\n"
                "    byod_image = Image.open(io.BytesIO(data))\n"
                "    print(validate_inputs(byod_image, threshold=threshold, nms_threshold=nms_threshold, names=[name])['verdict'])\n"
                "    byod_result = pipe.detect(byod_image, threshold=threshold, nms_threshold=nms_threshold)\n"
                "    for det in byod_result['detections'][:20]:\n"
                "        print(f\"{{det['label']:>14s}} {{det['score']:.3f}}\")\n"
                "    print(evaluation_report(byod_result, None, sample_kind='byod')['verdict'])\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True and re-run this cell to use your own image.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    # Replace byod_records with your own labelled records:\n"
                "    #   [{{'image': PIL.Image, 'boxes': [[x0, y0, x1, y1], ...], 'labels': ['my-class-a', ...]}}, ...]\n"
                "    byod_records = []  # noqa: F841 -- supply your own\n"
                "    byod_manifest = validate_dataset(byod_records, BYOD_CLASS_NAMES, epochs=EPOCHS)\n"
                "    print(json.dumps({{k: v for k, v in byod_manifest.items() if k != 'schema'}}, indent=2))\n"
                "    byod_train, byod_held = split_records(byod_records, holdout=HOLDOUT, seed=SEED)\n"
                "    byod_pipe = YoloxDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=BYOD_CLASS_NAMES, seed=SEED)\n"
                "    print('baseline:', {{k: round(v, 4) for k, v in byod_pipe.evaluate(byod_held).items() if k in ('ap', 'ap50')}})\n"
                "    byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED)\n"
                "    print('adapted :', {{k: round(v, 4) for k, v in byod_pipe.evaluate(byod_held).items() if k in ('ap', 'ap50')}})\n"
                "    byod_pipe.save_artifact(OUTPUTS / 'byod-adapter.pt', notes='BYOD tutorial run')\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True, supply byod_records and BYOD_CLASS_NAMES, and re-run this cell.')"
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
        "## References\n\n"
        "- Ge, Z., Liu, S., Wang, F., Li, Z. and Sun, J. (2021). *YOLOX: Exceeding YOLO Series in 2021.* [arXiv:2107.08430](https://arxiv.org/abs/2107.08430) — the anchor-free design, the decoupled head and SimOTA.\n"
        "- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) — the upstream repository, Apache-2.0. The model code carried in Section 2 is vendored from it at a pinned commit; the repository's `docs/UPSTREAM.md` records the revision, the per-file digests and the three edits applied.\n"
        "- [Release `0.1.1rc0`](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) — the immutable tag whose assets carry every published YOLOX checkpoint, including the `yolox_s.pth` this notebook verifies.\n"
        "- Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312) — the 80 classes the pretrained checkpoint predicts, and the average-precision protocol the evaluation helper approximates.\n"
        "- Repository model card: https://github.com/kurtvalcorza/yolox-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/yolox-detection-pipeline`](https://github.com/kurtvalcorza/yolox-detection-pipeline) — this notebook's source repository: the package carried in Section 2, its tests, `MODEL_CARD.md`, and `docs/release-verification.md` with the measured fine-tuning grid behind the defaults used here."
    ),
}
