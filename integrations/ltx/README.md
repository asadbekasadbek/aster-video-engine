# Native LTX experiment

Isolated Rust workspace. Its pinned Candle fork is distinct from the main
workspace's registry Candle CPU baseline. Always use `--locked`.
Windows CUDA numerical operations and full text-to-MP4 execution have passed
on the development RTX 4070. Complex hail/enter taxi action adherence failed;
working inference is distinct from reliable scene control.

## Build

On the validated Windows host (CUDA Toolkit 12.6.3, MSVC 14.42.34433):

```powershell
$env:CARGO_HOME = 'D:\Projects\aster-video-engine\.cargo-cache'
$env:CARGO_TARGET_DIR = 'D:\Projects\aster-video-engine\target\ltx'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:NVCC_CCBIN = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64'
$env:PATH = "$env:CUDA_PATH\bin;$env:NVCC_CCBIN;$env:PATH"
cargo build --manifest-path integrations/ltx/Cargo.toml --features cuda --release --locked --offline
.\target\ltx\release\aster-ltx.exe probe cuda
```

The linker currently reports LNK4098 from mixed CRT selection in native
dependencies. It has not been suppressed; numerical probes and unit tests run,
but that does not establish safety of every native operation.

## First model experiment

Approved files are downloaded and verified on the development host. From the repository root:

```powershell
.\target\release\aster.exe generate --prompt "A yellow taxi drives slowly along a city street in daylight." --output artifacts/ltx-first.mp4 --worker target/ltx/release/aster-ltx.exe --models models/ltx-0.9.8-2b --ffmpeg C:\OSPanel\modules\php\php_8.2\ffmpeg.exe
```

The parent CLI accepts `--prompt-file` instead of `--prompt` as well. Output
must not already exist. This experiment uses 384x256, 25 frames at 24 fps,
seed 42 and seven published distilled sigmas. `--frames N` accepts 8n+1 between
9 and 121. Optional `--planner taxi-v1` accepts the restricted Russian scenario
in examples/taxi.txt, replays its state and sends English event text to T5.
This is whole-clip text conditioning, not per-frame neural control.
It runs the first denoising pass
only, without the official multi-scale refinement/upscaler. Prompt limit is
128 tokens; longer input fails explicitly. English prompts are the initial
quality-evaluation target. No Python runtime is involved.

SHA-256 is checked before loading weights. T5 Q5_K_M executes on CPU and is
released before the BF16 transformer loads onto CUDA. Transformer weights are
released before VAE weights load. This is component staging; it does not yet
implement transformer-block streaming. The residency manager gates real
RAM-to-VRAM weight copies but does not track all tensor allocations/lifetimes.
The 9 GiB bound covers selected weight payload only;
activations, conversions, CUDA context and workspaces require extra memory.
Non-finite decoded values fail instead of being silently converted to pixels.

Timing includes integrity checks and model loading after CUDA initialization.
RAM is process-lifetime peak working set, excluding the FFmpeg child. VRAM is
sampled device-wide with nvidia-smi, including other applications; short peaks
can be missed. No exact per-process VRAM or energy measurement is claimed.
World State replay can supply whole-clip text conditioning through taxi-v1.
It does not reduce neural computation or bind object identity. Mode is FULL.

## Observed runs

| Run | Frames | Total time | Process peak RAM bytes | Sampled device-wide VRAM MiB |
| --- | ---: | ---: | ---: | ---: |
| First yellow taxi | 25 | 22.151 s | 4260130816 | 5406 |
| Hail/enter text | 121 | 31.404 s | 4261302272 | 6473 |
| Residency replay | 25 | 21.653 s | 4261154816 | 5400 |
| World-plan text | 121 | 30.926 s | 4260700160 | 6384 |

All outputs decode successfully. First/residency decoded frame MD5s match
exactly. Build work overlapped the residency run: no performance improvement
claim is made. Both long scene tests failed action adherence on visual
inspection. These measurements are a first-pass-only baseline, not a claim of
official pipeline parity, training quality, or general model correctness.
