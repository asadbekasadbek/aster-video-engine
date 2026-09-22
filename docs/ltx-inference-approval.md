# LTX inference approval manifest — 2026-09-22

Status: proposed, NOT downloaded, NOT installed, NOT license-accepted. Windows/CUDA model inference has not been tested. User approval should cover the following staged experiment; model fit/quality are not guaranteed.

## Selected model and backend
LTX-Video 0.9.8 2B distilled, original non-FP8 unified SafeTensors. Text encoder: T5-v1.1-XXL encoder Q5_K_M GGUF on CPU; retain prompt embeddings, release T5 before GPU denoising. First workload 384x256, 25 frames, batch 1, seed 42, version-specific 7-step schedule. Taxi evaluation later uses 145 frames/24 fps (6.04 s), only if measured memory allows. English model prompts are the first validated target; arbitrary Russian prompt understanding/translation is not implemented.

Rust code sources, pinned for the proposed build:
- https://github.com/FerrisMind/candle-video/tree/ebe43c414d2519535357113cad22d15fba0ce900 (Apache-2.0).
- https://github.com/FerrisMind/candle/tree/d8e9292b78ad82cb76a6ad8b42b4e2f42ae6ff2b (MIT OR Apache-2.0), selected branch dependency must be pinned in local integration/lockfile.
- Source manifests inspected: candle-video default features are EMPTY, unlike its README. Initial build uses cuda only, without Flash Attention, cuDNN, NCCL, MKL, Vulkan or wgpu. Existing registry Candle CPU probe stays isolated.
- Requested Rust dependency download/build: pinned sources above plus their resolved transitive Cargo dependencies. Direct declarations in video manifest: safetensors 0.8.0, serde 1.0.228, serde_json 1.0.150, anyhow 1.0.102, tokenizers 0.23.1, thiserror 2.0.18, hf-hub 0.4.3, clap 4.6.1, image 0.25.10, tracing 0.1.44, tracing-subscriber 0.3.23, glob 0.3.3, statrs 0.18.0, gif 0.14.2, rayon 1.12.0, rand 0.10.1, regex 1.12.3. These are upstream semver declarations, NOT an already-resolved installation inventory. CUDA fork declares cudarc 0.19.8 with cuda-version-from-build-system and candle-kernels/cudaforge 0.1.2. After approval resolve and record exact versions/checksums/licenses in an isolated Cargo lockfile before integrating. No upstream Python scripts or example auto-download paths will be executed.

## Exact weights/tokenizer download proposal
Target: D:/Projects/aster-video-engine/models/ltx-0.9.8-2b/ . Only the three files below, plus small model cards/license notices. API sizes/identities checked at the pinned revisions; not yet local integrity verification.

| File | Source revision | Bytes | Integrity identity |
| --- | --- | ---: | --- |
| ltxv-2b-0.9.8-distilled.safetensors | Lightricks/LTX-Video @ 8984fa25007f376c1a299016d0957a37a2f797bb | 6340744492 | SHA-256 76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f |
| text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf | oxide-lab/LTX-Video-0.9.8-2B-distilled @ 97a56ae43148edc6a4d26efc9e35811458ad9a24 | 3386856640 | SHA-256 b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae |
| text_encoder_gguf/tokenizer.json | same oxide-lab revision | 1389353 | Git blob SHA-1 41911369be79611d82d152725113e66206bd7b79 (not a raw SHA-256) |

Total: 9728990485 bytes (~9.73 decimal GB, ~9.06 GiB), excluding tiny notices. No FP8 weights, upscalers, datasets or extra model versions.

Pinned download URLs:
- https://huggingface.co/Lightricks/LTX-Video/resolve/8984fa25007f376c1a299016d0957a37a2f797bb/ltxv-2b-0.9.8-distilled.safetensors
- https://huggingface.co/oxide-lab/LTX-Video-0.9.8-2B-distilled/resolve/97a56ae43148edc6a4d26efc9e35811458ad9a24/text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf
- https://huggingface.co/oxide-lab/LTX-Video-0.9.8-2B-distilled/resolve/97a56ae43148edc6a4d26efc9e35811458ad9a24/text_encoder_gguf/tokenizer.json

License decision: original model is [LTXV Open Weights 0.X, 2025-04-15](https://huggingface.co/Lightricks/LTX-Video/blob/8984fa25007f376c1a299016d0957a37a2f797bb/LTX-Video-Open-Weights-License-0.X.txt), a custom license with use restrictions and special commercial conditions for entities with annual revenue >=USD 10M. User must review/accept it. [Original T5](https://huggingface.co/google/t5-v1_1-xxl) declares Apache-2.0; oxide-lab bundles a community quantization and marks the combined repository license as other. Exact conversion provenance/parity is not independently verified; do not redistribute the bundle as uniformly Apache-2.0. These are weights, not licensed training data.

## System dependency proposal
NVIDIA CUDA Toolkit 12.6 Update 3 for Windows x86_64, compiler + runtime + cuBLAS + NVRTC and their development headers/libraries. Official source: https://developer.nvidia.com/cuda-12-6-3-download-archive . Windows installation guide: https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-microsoft-windows/index.html . Existing VS2022/MSVC is available. Actual compile/run is the compatibility gate, not the version number.

Local installer: https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_561.17_windows.exe . HEAD verified HTTP 200 and 3227678920 bytes (3.23 GB), not downloaded. Installer includes an old driver; install Toolkit components ONLY, do not install/change the NVIDIA driver, Nsight or optional integrations. Validate NVIDIA signature before execution. NVIDIA installer license/UAC may require direct user action. Proposed normal installation modifies Program Files/system configuration and therefore needs explicit approval. Do not run unattended full/default installation. No cuDNN requested for the initial small CUDA trial.

Combined known downloads ~12.96 GB plus Cargo source/build cache and notices. Reserve additional SSD space for extraction, build output, weights staging and MP4. Cargo download size is unresolved; it cannot honestly be given as an exact total before dependency resolution.

## Implementation gates after permission
1. Isolated pinned Rust/CUDA compile and CPU/GPU numerical smoke checks before weights download. Do not replace the working CPU backend or silently fall back from requested CUDA.
2. Download only manifest artifacts, verify hashes/size, inspect model metadata and license notices.
3. Adapt weight loading: upstream example loads the whole unified tensor map onto the selected device; this does NOT satisfy Aster staging. Read validated tensor ranges on CPU and load only the denoiser; release it before loading the VAE. Quantized T5 runs on CPU and is dropped after encoding. Reserve activation/transfer workspaces in addition to weight bytes. Core residency ledger alone does not enforce driver allocation.
4. Audit native ops: T5 GGUF linear/embedding/norm; DiT linear, RoPE, full attention, F32 softmax and version scheduler; VAE temporal conv implemented through 2D convolution slices. First disable optional attention kernels and validate small tensors. Upstream temporal/spatial tiling must be checked for seams and correctness before enabling quality claims.
5. Connect decoded RGB8 frames to existing mp4::encode. Measure whole job and per-stage timing, process peak working set, sampled device VRAM baseline/peak with sampling interval and scope. Device-wide nvidia-smi is not process-only VRAM. Record GPU utilization/energy as unavailable unless actually measured.
6. First actual arbitrary English prompt -> neural video -> MP4, then taxi scenario visual/event evaluation. The restricted Russian planner currently emits symbolic events only; planned coordinates are not automatically consumed as neural conditioning. No image montage may satisfy this gate.
7. Full spatiotemporal attention and current pipeline expose no verified local/regional scene-edit API. Initial neural generation is FULL; cache only verified identical conditioning products. Add spatial constraints/selective compute only with paired baseline and quality evidence.