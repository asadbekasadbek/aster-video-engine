# Temporal reference experiment — 2026-09-28

Status: real neural representation reuse is verified. Full taxi action/identity
acceptance is **not met**. No sparse transformer speedup is claimed.

## Mechanism

`aster sequence` replays six compound events through World State. Change Detection
and a reference stamp (previous tick, object IDs/appearance handles, globals)
decide whether the last generated frame may condition the next segment. The VAE
encodes that frame; its normalized latent replaces the first latent frame of the
next segment. Those 96 tokens receive timestep zero and remain fixed through all
seven scheduler updates. Full attention and MLP computation still executes on
every token. These operations occur in native Rust/CUDA, without Python.

The original `generate` model path remains separate. The adapted transformer
uses the pinned library's attention/norm/projection code; only block/model
modulation was adapted to per-token timesteps. Tests compare scalar and repeated
token timesteps against the original model with shared nonzero test weights.
The Apache-2.0 source notice/license is retained in `src/reference`.

This is temporal conditioning, not object segmentation, object-ID grounding,
spatial inpainting, or selective transformer evaluation. Symbolic identity
equality cannot establish visual identity equality.

## First paired experiment (v1 prompts, before initial-prompt correction)

Both paths: LTX 0.9.8 2B distilled, BF16 CUDA, CPU T5 Q5_K_M, 384x256,
24 fps, seven first-pass sigmas, initial segment + six event segments, 49 frames
per segment. Seeds 42 through 48. The first frame of each later segment is
omitted from export equally in both paths: 337 output frames / 14.041667 seconds.
No second-pass upscaler or additional model is used.

Artifacts and full logs are under `artifacts/`:
`taxi-reference-49.{mp4,log}` and `taxi-independent-49.{mp4,log}`.
Both MP4s fully decode with FFmpeg. Contact sheets sample the start/middle/end
of every segment: `taxi-reference-contact.png`, `taxi-independent-contact.png`.

| Measurement | Independent events | Previous-frame reference |
| --- | ---: | ---: |
| Output frames | 337 | 337 |
| Transformer forwards | 49 | 49 |
| Transformer input tokens over all forwards | 32928 | 32928 |
| VAE reference encodes | 0 | 6 |
| Fixed tokens per conditioned forward | 0 | 96 |
| Maximum fixed-token drift | N/A | 0 |
| Sum of measured segment time | 139.268 s | 142.864 s |
| Total including model integrity I/O | 144.278 s | 222.222 s |
| Process peak working set | 4709027840 bytes | 4712599552 bytes |
| Sampled device-wide VRAM peak | 5209 MiB | 5147 MiB |
| GPU baseline | 706 MiB | 679 MiB |
| Mean RGB MAE at six segment joins | 0.29784850 | 0.05345055 |

The reference run was first and had cold integrity I/O. Total time is therefore
cache-confounded; segment times are shown separately. No build ran during either
inference. These are single observations, not a statistical speed benchmark.
GPU memory includes other applications and is sampled (short peaks may be
missed); process RAM excludes FFmpeg. Energy is unavailable.

MAE is measured on RGB8 frames **before H.264 encoding**, comparing the last
exported frame of a segment with the first exported frame of the next, divided
by 255. The observed 82.05% reduction measures visual discontinuity only;
motion, identity and correct actions require separate inspection. Segment time
increased 2.58%; measured transformer computation saved is **zero**. References
add six VAE encoding operations, so this is not a compute optimization.

## Visual acceptance

Manual contact-sheet inspection:

- Initial standing person: FAIL, person is absent in both shared initial clips.
- Hailing: not demonstrated clearly in the reference run.
- Approach/stop: vehicle movement exists, exact intended stop is not established.
- Rear door opening: FAIL/not clearly shown.
- Entry and seated occupancy: FAIL; reference person crouches outside the car.
- Departure with that person inside: FAIL; person remains outside.
- Independent events change camera, car body, person and street abruptly.
- References retain more scene context but still allow drift and do not fix actions.

The current v2 initial prompt is more explicit about a visible full-body person;
references are clamped to the VAE input range. Its results must not be represented
as the above v1 comparison. `--events 0` is an initial-only probe; `--seed` makes
candidate selection reproducible. Failed candidates remain in artifacts.

## Reuse limits and alternatives

- Exact text embeddings may be cached only for identical model/tokenizer/text.
  Changing an action changes bidirectional T5 context: copying a shared prefix's
  embeddings is not an exact optimization.
- Matching VAE-encoded references can be retained and reused; here each previous
  frame is different, so it is encoded once for its next event.
- Whole-scene reference tokens influence the next segment, but do not assign
  a token subset to a particular World State object.
- Global self-attention couples tokens. Freezing a latent or output mask does
  not justify reusing intermediate activations or claiming skipped operations.
- Official LTX supports image/multiple keyframe conditioning; a control-model
  route needs a separate compatibility and weight/license review. This project
  has not verified 13B pose-control weights with the approved 2B model.

Primary sources: [official pipeline](https://github.com/Lightricks/LTX-Video/blob/main/ltx_video/pipelines/pipeline_ltx_video.py),
[official transformer](https://github.com/Lightricks/LTX-Video/blob/main/ltx_video/models/transformers/transformer3d.py),
[pinned Rust implementation](https://github.com/FerrisMind/candle-video/blob/ebe43c414d2519535357113cad22d15fba0ce900/src/models/ltx_video/ltx_transformer.rs).
Sources inspected 2026-09-28; official main URLs are mutable. See RESEARCH.md.

## Revision 2: explicit initial person, seed 43

`taxi-reference-v2-seed43.mp4` and matching log: 337 frames, H.264,
384x256/24 fps, 14.041667 seconds, independent full decode passes.
Total 149.100 s; generation excluding integrity I/O 144.120 s;
process peak RAM 4712796160 bytes; sampled device-wide VRAM 5170 MiB
(baseline 734, 566 samples, no sampling errors). Six VAE anchors, 49 forwards,
zero frozen-token drift and zero skipped transformer forwards.

Start/middle/end contact-sheet inspection shows the initial person and yellow
car, raised arms during hail, car movement, then the man walking and bending
outside. Rear-door opening and entry are not demonstrated; the car changes body
shape and the person remains outside before departure. Full action and identity
acceptance FAIL. This trial improves initial object presence, not reliable control.
Seed 42 with the same explicit initial prompt omitted the man; both failed and
successful initial probes are retained. This is selected-seed evidence, not a
success-rate estimate. Lower join MAE cannot establish action correctness.

Pose-control alternative inspected: official LTX-Video-ICLoRA-pose-13b-0.9.7
model card specifies LTXV_13B_097_DEV as its base, not this 2B distilled model.
No compatibility with the current native adapter or 12 GB VRAM is established.
No additional weights were downloaded. Another prompt-only retry is not a
validated fix for body/door contact or identity grounding.
## Equal-output-length comparison with the preserved whole-clip path

Both outputs: 113 frames, 384x256/24 fps (4.708333 s), full FFmpeg decode passes.
Whole-clip source is examples/taxi-full-baseline.txt (seed42). Sequence uses
examples/taxi-control.txt, v2 action prompts, seed43..49,17frames/segment.
These are two strategies for the same story, not matched conditioning/noise;
use the v1 paired ablation above to isolate reference conditioning.

| Measurement | Existing generate | Sequence with references |
| --- | ---: | ---: |
| Total time including integrity | 31.031 s | 128.884 s |
| Process peak RAM bytes | 4259897344 | 4741750784 |
| Sampled device-wide VRAM MiB | 5966 | 5222 |
| GPU baseline MiB | 732 | 733 |
| Transformer forwards | 7 | 49 |
| Tokens processed over forwards | 10080 | 14112 |
| Additional VAE reference encodes | 0 | 6 |

Artifacts: taxi-whole-113.mp4/log and taxi-reference-113.mp4/log.
Manual sampled-frame review: whole clip shows an open door and the person
outside, without the complete hail/approach/entry/departure chain. Short-event
sequence remains mostly a standing man beside a moving taxi; entry fails.
Neither meets action acceptance. Sequence is 4.15x slower in this observation;
shorter model segments use less sampled peak GPU memory, but do more forwards,
reload components and repeat T5 encoding. No compute saving or semantic quality
improvement over the existing whole-clip path is established.