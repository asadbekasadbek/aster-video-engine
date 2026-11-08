# Quality trial downloads — approval required, 2026-10-03

Official repository Lightricks/LTX-Video, pinned revision
8984fa25007f376c1a299016d0957a37a2f797bb. Sizes and SHA256 obtained from official
HF model API with blobs=true. Existing CUDA and2B/T5 files are reused.

| File | Bytes | Purpose | SHA256 |
| --- | ---: | --- | --- |
| ltxv-13b-0.9.8-dev.safetensors | 28579183340 | Full/dev BF16 checkpoint for highest-quality LTX0.9.8 trial | 56b39a874ef9d0c7ad4099954f0ee4fa7e600ce83a24b3736f9e12223e7c71be |
| ltxv-spatial-upscaler-0.9.8.safetensors | 505024432 | Learned latent upscaler for recommended multi-scale pipeline, also usable with existing2B | 5b076031c6f860db9037a54f3bb819f10bfb5532ea26a6d30062292428a0c208 |

Total29084207772bytes =29.084GB (decimal). Direct URLs:
https://huggingface.co/Lightricks/LTX-Video/resolve/8984fa25007f376c1a299016d0957a37a2f797bb/ltxv-13b-0.9.8-dev.safetensors
https://huggingface.co/Lightricks/LTX-Video/resolve/8984fa25007f376c1a299016d0957a37a2f797bb/ltxv-spatial-upscaler-0.9.8.safetensors

Same LTXV Open Weights0.X license dated2025-04-15 as previously accepted:
https://huggingface.co/Lightricks/LTX-Video/blob/8984fa25007f376c1a299016d0957a37a2f797bb/LTX-Video-Open-Weights-License-0.X.txt
No new system software or Python runtime proposed. No download yet.

13B cannot be loaded whole onto12GB: even the official FP8 checkpoint is
15694279916bytes including VAE. Current Rust loader rejects FP8, and library's
13B-dev preset replaces dynamic guidance with fixed peak values; this is NOT
recommended-pipeline parity. Before real13B inference, implement and test bounded
BF16 block residency plus exact guidance/STG/scheduler and activation headroom.
Downloading is preparation, not proof of compatibility or quality. Prefer BF16
streaming for the quality reference; quantization is not the first quality choice.
The learned upscaler path also needs a Rust implementation/validation.