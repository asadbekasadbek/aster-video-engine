# One-second quality experiments — 2026-10-03

**Acceptance remains FAILED.** Five real neural clips were generated; successful
encoding is not photorealism. Q3 is the provisional comparison candidate, not an
approved quality baseline. World/action/efficiency work is deferred.

## Reproduction and provenance

Prompt: `examples/quality-baseline.txt`, exactly the owner's English prompt.
All outputs are under `artifacts/`; each name below has `.mp4`, `.log`, `.png`
and `.probe.json`. Q1–Q4 additionally have `.params.txt` recorded by the Rust
worker, including exact model hashes, prompt, seed, scheduler and VAE parameters.
Q0 used the existing `generate` command and the same settings as Q1 except the
historical decoder noise order. No prompt enhancement, reference image, external
frames, or post-generation detail synthesis was used.

```powershell
.\target\release\aster.exe quality --prompt-file examples/quality-baseline.txt --output artifacts/my-quality.mp4 --worker target/ltx/release/aster-ltx.exe --models models/ltx-0.9.8-2b --ffmpeg C:\OSPanel\modules\php\php_8.2\ffmpeg.exe --profile native576-t5-256 --seed 42
```

The CLI accepts `legacy`, `corrected`, `native576`, `native576-t5-256` profiles.
The output must not already exist. Fixed 25 frames at 24 fps = 1.041667 seconds.
Profiles use conservative free-device-memory admission, not a universal guarantee
about all CUDA allocations. None of these five trials encountered OOM.

## Measured comparison

| Trial / artifact stem | Resolution | T5 length | Seed | Noise order | Total ms | Generation ms | Peak working set bytes | Sampled device VRAM MiB |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| quality-q0-current | 384x256 | 128 | 42 | historical | 22004 | not separately logged | 4259692544 | 5312 |
| quality-q1-corrected | 384x256 | 128 | 42 | normalized | 22354 | 17457 | 4259807232 | 5289 |
| quality-q2-native576 | 576x384 | 128 | 42 | normalized | 25433 | 20587 | 4260339712 | 5290 |
| quality-q3-t5-256 | 576x384 | 256 | 42 | normalized | 27901 | 23131 | 4262666240 | 5358 |
| quality-q4-seed43 | 576x384 | 128 | 43 | normalized | 25507 | 20545 | 4259880960 | 5341 |

Total is worker time from before model hash verification through export, excluding
process startup/CUDA initialization. Generation excludes hash verification.
Windows peak working set measures this worker, not combined system RAM or the
separate FFmpeg child. VRAM is device-wide sampling (200ms wait plus query time),
not allocator-instrumented true peak; short spikes may be missed. Device baseline
MiB: Q0 903, Q1 855, Q2 856, Q3 856, Q4 1058. Sample counts: 83,86,97,106,99;
zero sampling errors. No simultaneous builds/inference. No speedup claim from
single runs; disk cache and other desktop GPU usage are uncontrolled.

## Visual review

Reviewed decoded frames 0,6,12,18,24 for each trial, plus all 25 Q3 frames in
`quality-q3-all-frames.png`. This is manual frame-based review, not an independent
human rating or a calibrated perceptual score. FFmpeg fully decoded all five
clips without errors; ffprobe confirms H264, 25 frames, 24/1 fps, 1.041667 seconds.

| Trial | Findings | Acceptance |
| --- | --- | --- |
| Q0 | Coherent yellow car/street, but blurred artificial face/hands and unnatural body detail. | FAIL |
| Q1 | Similar composition and defects; noise-order correction alone does not establish meaningful realism improvement. | FAIL |
| Q2 | More legible car geometry and street detail. Human face/hands remain weak; glossy synthetic appearance. | FAIL |
| Q3 | Similar to Q2; slightly more legible facial profile in sampled frames. Human detail remains insufficient. Provisional candidate, no quantified improvement over Q2. | FAIL |
| Q4 | Fairly coherent yellow car, but the required standing man is absent. A clean car-only scene does not satisfy the prompt. | FAIL |

Q3 has broadly consistent car silhouette and background across this short sample;
the man subtly lowers his head/body. There is no basis to certify correct fingers,
natural skin, physically correct lighting or absence of small geometry drift.
Increasing padding length did not solve the quality problem. Resolution changes
also change latent noise tensor shape/composition, even with the same integer seed;
Q1/Q2 are not an identical-latent image-detail comparison. No endless seed search.

## Inference audit

Current checkpoint: `ltxv-2b-0.9.8-distilled.safetensors`, SHA256
`76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f`.
DiT and bundled VAE BF16, no weight quantization. Scheduler latents F32.
CPU T5-v1.1-XXL Q5_K_M SHA256
`b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae`.
Pinned candle-video revision `ebe43c414d2519535357113cad22d15fba0ce900`.

- Seven explicit FlowMatch Euler sigmas: 1,.9937,.9875,.9812,.975,.9094,.725,
  then terminal 0. CFG1, STG0, no negative conditioning or stochastic sampling.
  These match the official distilled **first pass**. Arbitrary extra steps do
  not reproduce dev inference.
- Official 0.9.8 config uses learned spatial upscaling and a second denoising
  pass; our implementation currently has only one pass. This is a concrete
  parity gap, not measured proof that it alone causes the observed defects.
- Official VAE decode noise is applied before inverse latent normalization.
  New quality profiles implement that order and a numerical test; historical
  generate/sequence/control retain their old order to preserve the baseline.
  VAE decode timestep .05/noise .025; spatial tile512, temporal16/stride8.
- Official text length defaults to256; original Rust128 rejects overlong prompts
  rather than truncating. This prompt fits128. Q3 tests256. Quantized T5 versus
  official full-precision embeddings is not numerically validated.
- PCG32 stream1442695040888963407 is not Torch's RNG. Same seed is reproducible
  input within this backend, not a claim of Python/Rust sample parity.
- Full model numerical parity and VAE tiling parity remain unverified. Low
  resolution, distilled capacity and missing second pass are plausible quality
  contributors, not experimentally isolated root causes.

Primary sources inspected 2026-10-03 (main URLs are mutable):
[2B config](https://raw.githubusercontent.com/Lightricks/LTX-Video/main/configs/ltxv-2b-0.9.8-distilled.yaml),
[official pipeline](https://raw.githubusercontent.com/Lightricks/LTX-Video/main/ltx_video/pipelines/pipeline_ltx_video.py),
[13B dev config](https://raw.githubusercontent.com/Lightricks/LTX-Video/main/configs/ltxv-13b-0.9.8-dev.yaml),
[publisher recommendations](https://github.com/Lightricks/LTX-Video).

## Next quality step and blocker

New weight approval is pending: exact URLs, pinned revision, sizes, hashes and
license in `docs/quality-model-downloads.md`. 13B dev BF16 plus learned upscaler
total 29,084,207,772 bytes. Nothing downloaded. Existing CUDA/2B approval does not
authorize these new files.

13B dev is a candidate, not a working Rust/Windows/12GB configuration. Required
work: bounded per-block BF16 CPU-to-GPU residency, exact dynamic CFG/STG/rescale
and CFG-star schedules, and the learned multi-scale path. Current loader cannot
load 13B whole; it rejects FP8. Quantizing just to make it fit is not the selected
quality baseline. Test upscaler/second pass against the existing 2B first, then
attempt a bounded 13B run only after implementation/resource validation. No
Python runtime, new system software, or quality claim from model size alone.

Validation this cycle: root fmt/clippy/tests/release and isolated CUDA
fmt/clippy/tests/release passed (36+10 tests), plus explicit native MP4 roundtrip
(1): 47 tests. Existing native LNK4098 warning remains documented. Final visual
acceptance is still open regardless of these checks.

## 2026-11-08: text-only, unsplit temporal decode

Same local RTX4070, verified weights, 25 frames/24fps, seed42 PCG32, single-pass BF16 2B distilled, seven Euler steps, CFG1/STG0, CPU Q5 T5 padded to256. No input image, RGB upscaling or detail filter. All three new MP4s decoded end to end and were probed; see artifacts with the listed stems plus .params.txt (full parameters), .log (telemetry), .probe.json and .png. All clips are 1.041667 seconds. Measurements below are worker total/generation excluding model hash verification; working set is worker-only, VRAM is the largest sampled value across the device.

| Stem | Prompt / native size | Total / generation (ms) | Peak working set (bytes) | Sampled VRAM (MiB) | Visual result |
| --- | --- | ---: | ---: | ---: | --- |
| quality-q5-text576 | owner / 576x384 | 74438 / 29006 | 4262768640 | 5469 | Compared with original Q3, unsplit decode alone does not resolve face/hand detail. |
| quality-q6-text960 | owner / 960x640 | 39159 / 34190 | 4263546880 | 8610 | Clear yellow sedan, but standing man missing. Fails the requested scene. |
| quality-q7-medium960 | medium-shot description / 960x640 | 38961 / 34005 | 4262879232 | 8610 | Best candidate: recognizable facial profile and shirt, stable taxi, consistent daylight. Skin is smooth, hands cropped, motion slight. |

Q5 and Q6 change only resolution; changing the noise tensor dimensions can also change composition. Q6 and Q7 change only the prompt. For Q7 the text describes the man in the foreground, clothing, camera and parked taxi instead of relying mainly on quality adjectives. This single comparison is not a universal prompt rule. Inspected Q7 at native resolution in frame12 and reviewed all25 frames in order; all25 decoded frame hashes are distinct. The hands/blink requested by the alternate prompt cannot be verified, full-body anatomy and the entire car cannot be judged from this framing. No conspicuous face or car morphing seen. Q7 is submitted for owner review, not certified photorealism or evidence that arbitrary prompts produce this quality.

Reproduce with the quality command above, replacing the prompt file with examples/quality-daylight-medium-shot.txt, using --profile text960-fullvae --seed 42 and a fresh output name. Original comparison prompt remains in examples/quality-baseline.txt. 13B dev and learned upscaling remain untested locally; no new weight download. No optimization before visual approval.

2026-11-09 update: opt-in reference13b-704-fullvae ran the downloaded 13B Dev on Windows/RTX 4070 in one pass without the learned upscaler. See [local profile](../integrations/ltx/README.md) and PROGRESS.md for the fixed-seed control. This supersedes the earlier runtime status for this profile only.
