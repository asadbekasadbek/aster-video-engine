# SD1.5 candidate artifact manifest — NOT APPROVED OR DOWNLOADED

Prepared 2026-09-20 from Hugging Face model metadata API, requested with `blobs=true`. This is an inventory for future approval, not a download script. Weights are not datasets. No model/license acceptance is implied.

Model repository: `stable-diffusion-v1-5/stable-diffusion-v1-5`

Revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`

| File | Bytes | LFS SHA-256 |
| --- | ---: | --- |
| text_encoder/model.fp16.safetensors | 246144864 | 77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e |
| unet/diffusion_pytorch_model.fp16.safetensors | 1719125304 | c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21 |
| vae/diffusion_pytorch_model.fp16.safetensors | 167335342 | 4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c |

Weights total: 2132605510 bytes (about 1.99 GiB). Size is disk payload, not runtime peak. CPU f32 loading increases weight memory; GPU activations and workspaces need separate measurement.

Tokenizer used by upstream Candle SD1.5 example: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, file `tokenizer.json`, 2224041 bytes, Git blob ID `564c0ebd5ce29c4ee4864004aee693deadd3128c` (NOT SHA-256). No content digest checked because file not downloaded. Complete download total including tokenizer: 2134829551 bytes. Reverify tokenizer provenance/license before approval.

Sources:
- [SD model metadata API](https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5?blobs=true)
- [Tokenizer metadata API](https://huggingface.co/api/models/openai/clip-vit-base-patch32?blobs=true)
- [Candle 0.11.0 SD example](https://github.com/huggingface/candle/blob/0.11.0/candle-examples/examples/stable-diffusion/main.rs)
- [Model card / CreativeML OpenRAIL-M license link](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

Remaining integration gates:
1. Validate pinned Candle CPU operations (separate smoke command); CUDA requires an approved toolkit/dependency path and kernel test, not just nvidia-smi.
2. Audit Candle 0.11.0 model APIs against the actual registry package; web main/tag snapshots and published crate contents can differ. Add candle-nn/candle-transformers only with dependency approval.
3. Implement explicit local paths, no implicit hf-hub auto-downloads, bounded model configuration and output handling. Upstream example's historical runwayml default must not be used implicitly.
4. Obtain approval for exact files and applicable license, download only pinned revisions, verify bytes/digests, inspect headers and component compatibility.
5. Batch 1 / 256x256 / deterministic seed smoke image with measured RAM/VRAM/time. Do not call a one-step noise image quality evidence. Full inference must complete with a documented scheduler/steps and reviewed image.

An SD image is the reduced first inference milestone. It does not satisfy actual video inference, long-duration consistency or 1080p native generation.
