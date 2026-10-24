# Weights

## What is pinned

| | |
|---|---|
| Upstream project | `Megvii-BaseDetection/YOLOX` |
| Revision | `0.1.1rc0` — an immutable **GitHub release tag** |
| Asset | `yolox_s.pth` |
| Bytes | 72,089,125 |
| SHA-256 | `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30` |
| Licence | Apache-2.0 |
| Published | 2021-08-18 (release `published_at` 2021-08-18T12:41:41Z) |
| Served file | `yolox_s.safetensors`, 36,014,492 bytes, SHA-256 `8fa0c1135952a70148c3e3114ac1ddae2d399600053841b89738dc3694e3aaf0` — written locally from the asset by `convert_checkpoint`, git-ignored, not distributed |
| Manifest | `weights/yolox-s/dimer-base-manifest.json` |
| Download | `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth` |

## Why a release tag and not a Hub revision

YOLOX publishes no Hugging Face model repository. Every published YOLOX checkpoint lives in the assets
of the `0.1.1rc0` release; the later `0.2.0` and `0.3.0` tags ship code only, and upstream's own
`yolox/models/build.py` at `0.3.0` points its download URLs back at `0.1.1rc0`. That is why this
package pairs `0.3.0` model code with a `0.1.1rc0` checkpoint: it is upstream's declared pairing, and
loading it `strict=True` matches all 462 state-dict keys with none missing or unexpected.

A release tag is immutable, but it is not a content digest, so the manifest carries the SHA-256 and
that is what makes the download trustworthy — not the URL.

## The checkpoint is a pickle

There is no SafeTensors release. `yolox_s.pth` is a `torch.save` pickle whose top-level keys are
`amp`, `model`, `optimizer` and `start_epoch`; the `model` value is the 462-entry state dict.

Three steps, in this order:

1. `verify_snapshot` re-hashes the file and raises on the first size or digest mismatch. It runs
   **before** the package imports `torch`, so a tampered snapshot never reaches a deserializer.
2. `torch.load(..., map_location="cpu", weights_only=True)` then refuses to construct arbitrary
   globals. Upstream's loader passes neither; the upstream downloader
   (`torch.hub.load_state_dict_from_url`, which fetches with no digest check at all) is deliberately
   not vendored.
3. `convert_checkpoint` reads the file that one time, keeps its `model` state dict, checks that it loads
   `strict=True` into the vendored architecture, and writes `yolox_s.safetensors` with `safetensors.torch.save_file`
   (36,014,492 bytes, SHA-256 `8fa0c1135952a70148c3e3114ac1ddae2d399600053841b89738dc3694e3aaf0`, pinned in code as `CONVERTED_SHA256`; converting twice gives
   identical bytes). `from_pretrained` serves only that file: it re-hashes it against the pin before
   importing `torch`, reads it with `safetensors.torch.load_file` and applies it with
   `load_state_dict(strict=True)`. Once the conversion exists the pickle is never read again; on CPU the
   converted tensors equal the source's exactly and the detections on the tutorial scene are identical.

Exported adapters are **not** pickles. `save_artifact` writes the adapted state dict as
`yolox-detection-adapter.safetensors` (tensors only, via `safetensors.torch.save_file`) and its metadata as a
plain-JSON sidecar `yolox-detection-adapter.json` that also records the SHA-256 of the weights file.
`load_artifact` reads the sidecar, refuses a foreign format tag, a mismatched pinned identity or model
key, or a weights file whose SHA-256 differs from the sidecar's, all before the weights are opened,
then reads them with `safetensors.torch.load_file` and applies them with `load_state_dict(strict=True)`.
Version-1 `.pt` adapters are no longer written or read.

## Staging

The manifest is committed; the checkpoint is not (`.gitignore` excludes `weights/**/*.pth`). On a
fresh clone:

```python
from yolox_detection_pipeline import stage_missing_files, verify_snapshot

stage_missing_files(allow_download=True)   # downloads only manifest-listed files, from the pinned tag
verify_snapshot()                          # re-hashes them; raises on the first mismatch
```

`stage_missing_files` refuses outright if the manifest's `modelId`/`revision` differ from the package
constants, so a swapped manifest cannot redirect the download.

`weights/** -text` is set in `.gitattributes`: on a Windows clone with `core.autocrlf=true`, line-ending
conversion would rewrite the bytes of any file under `weights/` and break the digest check.

## Preprocessing, and the one place this package differs from upstream

Upstream's `ValTransform(legacy=False)` letterboxes onto a 640×640 canvas filled with grey `114`,
preserving the aspect ratio and pasting the image at the top-left, then hands the network **BGR** in
raw 0–255 float32 — no `/255`, no mean/std normalisation. This package reproduces that exactly, with
one substitution: the resize uses `PIL.Image.BILINEAR` instead of `cv2.INTER_LINEAR`, so the runtime
does not need OpenCV.

The two kernels are not bit-identical in general. They are irrelevant on every sample in this
repository, because the resize is **skipped entirely** when the input is already 640×640 — a Pillow
bilinear resize to the source size is verified to be the identity, and a unit test asserts that a
640×640 input becomes the source pixels with the channels reversed and nothing else. Only a
differently sized image goes through the differing kernel, where scores may shift slightly from what
upstream's OpenCV path would produce. Nothing in this repository's recorded numbers depends on it.

Channel order is worth stating separately because it is silent when wrong: feeding the same
letterboxed tensor as RGB rather than BGR changed the tutorial scene from six detections to five, lost
the bench, took the stop sign from 0.962 to 0.839 and promoted a spurious second clock from 0.309 to
0.441.

## Vendored model code

The weights are useless without the architecture, and the architecture here is vendored rather than
installed. Provenance for those eight modules — upstream commit
`419778480ab6ec0590e5d3831b3afb3b46ab2aa3`, a SHA-256 per upstream file, and a diff of each of the
three declared edits — is in [`UPSTREAM.md`](UPSTREAM.md), and `tools/vendor_upstream.py --check`
re-fetches upstream and fails on any divergence.
