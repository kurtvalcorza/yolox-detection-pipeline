# `weights/yolox-s/`

The snapshot directory for the pinned YOLOX-S checkpoint.

| File | Committed | Notes |
|---|---|---|
| `dimer-base-manifest.json` | yes | identity, per-file byte size and SHA-256, and the release-asset URL |
| `yolox_s.pth` | **no** | 72,089,125 bytes; git-ignored, staged on demand |

`yolox_s.pth` is an asset of the upstream GitHub release `0.1.1rc0` — YOLOX publishes no Hugging Face
model repository. To stage it:

```python
from yolox_detection_pipeline import stage_missing_files, verify_snapshot

stage_missing_files(allow_download=True)
verify_snapshot()
```

`stage_missing_files` downloads only files the manifest lists, and only from the pinned tag; it
refuses outright if the manifest's identity differs from the package constants. `verify_snapshot`
then re-hashes what arrived. The checkpoint is a **pickle**, so that digest check runs before
`torch.load(..., weights_only=True)` and before the package imports `torch` at all.

`.gitattributes` sets `weights/** -text`: on a Windows clone with `core.autocrlf=true`, line-ending
conversion would rewrite these bytes and break the digest check.

See [`../../docs/WEIGHTS.md`](../../docs/WEIGHTS.md).
