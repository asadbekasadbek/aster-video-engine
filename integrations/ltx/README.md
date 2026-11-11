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

## Temporal reference experiment

`sequence` applies World State events and lets change detection invalidate a
previous-frame reference before real VAE encoding and per-token denoising.
It uses the same approved weights; no Python or extra model is needed.

```powershell
.\target\release\aster.exe sequence --prompt-file examples/taxi-control.txt --output artifacts/my-taxi-sequence.mp4 --worker target/ltx/release/aster-ltx.exe --models models/ltx-0.9.8-2b --ffmpeg C:\OSPanel\modules\php\php_8.2\ffmpeg.exe --frames 49 --reuse on --seed 43
```

`--frames` is per segment. Initial + six events gives 337 frames at 49;
`--events 0` probes only the initial scene. `--reuse off` runs independent
segments for comparison. Seed defaults to 42 and increments per segment.
The parser accepts only the documented taxi grammar, not arbitrary stories.
This is temporal conditioning, not spatially selective computation. All tokens
still pass through the transformer. The v1 paired experiment reduced join MAE
82.05%, with no transformer work saved and segment time increased 2.58%.
The v2 clip raises the person's arms, but rear-door opening and entry still fail;
car appearance drifts. See ../../docs/temporal-reference-benchmark.md.
## Reviewed taxi transitions (experimental control command)

```powershell
.\target\release\aster.exe control --worker target/ltx/release/aster-ltx.exe --models models/ltx-0.9.8-2b --run-dir artifacts/my-reviewed-taxi --ffmpeg C:\OSPanel\modules\php\php_8.2\ffmpeg.exe --stage 0
```

This predefined taxi profile has 14 microstages (0..13), 49 generated frames per
stage, 384x256/24fps, seeds43..56, seven distilled steps. It is not a general
language planner. Generate/sequence remain available with their previous APIs.
Each invocation generates one real MP4 and stops awaiting external visual review.
Nine actual generated frames (stored as bounded finite F32 pixels) are VAE-encoded
as two hard latent frames for the next accepted stage. The first nine frames in
later MP4s are overlapping context, not nine new timeline frames.

Before requesting stage N+1, inspect stage N's video against its expected state,
stage0's canonical appearance reference and the last accepted segment. Write
stage-NN.review with exactly these fields; replace the hash with the worker's
manifest_sha256 output, and record honest decisions (pass/fail/unknown):

```text
manifest_sha256=THE_PRINTED_SHA256
action=pass
person=pass
vehicle=pass
background=pass
transition=pass
reviewer=NAME_OR_EVALUATOR_VERSION
evidence=Concrete observed actions, identity comparisons and inspected frame indices
```

The evidence must describe actual visual inspection.
No existing artifact is overwritten. Failed runs remain for diagnosis; use a new
run directory for a new experiment. A missing/failed/unknown/stale review blocks
inference before CUDA initialization. Review is an externally supplied assertion,
not a cryptographically authenticated reviewer or an automatic vision model.
Hashes bind video, reference pixels, prompt, settings and expected state to the
review. Do not mark pass merely because a file was written. Current implementation
retains canonical evidence for identity comparison; it does NOT inject an
object-specific learned identity embedding or guarantee physical interactions.
Export accepted stages only, dropping the nine-frame overlap from later segments:

```powershell
.\target\release\aster.exe control-export --worker target/ltx/release/aster-ltx.exe --run-dir artifacts/my-reviewed-taxi --ffmpeg C:\OSPanel\modules\php\php_8.2\ffmpeg.exe --output artifacts/my-reviewed-prefix.mp4 --stages 3
```

The Rust command rechecks all manifests/reviews, reserves a new output and invokes
native FFmpeg to concatenate/re-encode. It reports partial_scenario until all14
stages are accepted. It cannot include a failed stage. Manual approval of all
stages is still external evidence, not automated semantic certification.
The symbolic positions are planning defaults, not calibrated physical geometry.
# One-second quality experiments

The opt-in `quality` command preserves historical generation paths. Profiles:
`legacy`, `corrected` (default), `native576`, `native576-t5-256`; fixed25frames
at24fps; optional `--seed`. It saves `.params.txt`; capture stdout for sampled
GPU telemetry. See [audit/results](../../docs/quality-baseline.md). The earlier audit predates the 13B profile below. No general 13B runtime
compatibility or photorealism acceptance is claimed.

User-photo experiment: `quality --image PATH --profile reference704-fullvae`
produces25neuralframes at704x384 with a VAE-encoded first-image reference and
unsplit temporal decoding. It saves reference hash/crop/settings in the sidecar.
See [photo comparison](../../docs/photo-quality-baseline.md); equal-photo-detail
acceptance remains open, and this does not validate text-only quality.

Text-only comparisons: use --profile text576-fullvae or text960-fullvae with the same quality command and no --image. Both use the 256-token text encoder setting and decode all 25 frames together. Resolution is native, not a final RGB upscale. A params.txt sidecar records input and inference settings; capture stdout to retain sampled GPU telemetry.

For an exact three-second image-to-video run choose --profile reference704-3s-fullvae and --image PATH. Generate one 73-frame neural sequence at 704x384 and export its first 72 frames at 24fps. No loop, frame duplication or playback-rate adjustment. Static spatial decoder tiles 256/stride192 keep the temporal decode unsplit.

For a native 960x544, three-second image-conditioned clip use --profile reference960-3s-fullvae (same 73 generated/72 exported frames). Reference self-attention evaluates 512 queries at a time against the complete key/value sequence; this does not skip attention or imply a speedup. VAE uses 256px spatial tiles and unsplit time.

## Потоковая загрузка весов

Сборка CUDA использует потоковую загрузку по умолчанию. Прежние команды quality, generate и запуска сцен получают этот режим через worker. Для существующей установки пересоберите интеграцию с настроенным CUDA 12.6 и укажите новый aster-ltx.exe как --worker. Пример из корня проекта:

```powershell
cargo build --manifest-path integrations/ltx/Cargo.toml --target-dir target/ltx --locked --release --features cuda
```

Для контрольного прогона установите `$env:ASTER_WEIGHT_MODE = 'resident'` перед обычной командой. Вернуться можно через `$env:ASTER_WEIGHT_MODE = 'stream'`. В quality режим записывается в params.txt.

T5 читает по одному блоку из GGUF на CPU. DiT держит общие проекции и текущий блок на GPU, после завершения вычислений блока освобождает его. VAE загружает только энкодер для исходного кадра или декодер для вывода. На диске остаётся исходный checkpoint. ОС может кэшировать прочитанные страницы в RAM. Пик RAM в отчёте — рабочий набор процесса. Размер активаций зависит от разрешения и длины видео.

Проверено на локальной RTX 4070, LTX 0.9.8 2B distilled. Для 960×544, 72 выходных кадров, seed 43, одинакового фото и настроек текущие resident и stream дали побайтно одинаковые MP4. Пик рабочего набора 4,26 → 1,23 ГБ, наблюдаемая VRAM всего устройства 6662 → 3261 МиБ, время генерации 74,30 → 89,29 с. Отдельная текстовая проверка 576×384 также воспроизвела исходный MP4. Публикуемые числа получены на отдельных прогонах; это не лимиты памяти для других входов.

В потоковом режиме T5 читает только строки словаря для токенов запроса. Веса с совпадающим типом передаются из прочитанного буфера на устройство напрямую. Экспорт преобразует в RGB каждый выходной кадр отдельно. Серия контрольных прогонов и SHA-256 записаны в PROGRESS.md. Для следующих изменений памяти сравнивайте побайтовое содержимое кадров при одинаковых весах, параметрах, входе, seed и версии кодировщика.

## Предзагрузка блоков при генерации

Из корня проекта скопируйте `.env.example` в `.env` и запускайте обычную команду генерации с пересобранным CUDA-worker. Файл читается из текущего каталога процесса. Указанные в PowerShell переменные окружения имеют приоритет над значениями файла.

```dotenv
ASTER_PREFETCH=1
ASTER_PREFETCH_RAM_PERCENT=10
ASTER_PREFETCH_VRAM_PERCENT=25
```

Работает в режиме `ASTER_WEIGHT_MODE=stream` (по умолчанию). При отсутствии настроек или при `ASTER_PREFETCH=0` используется последовательная загрузка. Файл .env понимает эти три ключа, значения задаются целыми числами без кавычек. Оба процента допускают 0–100.

RAM-процент задаёт бюджет очереди из **доступной физической памяти** перед загрузкой трансформера. Фоновый поток читает будущие блоки с диска и подготавливает их на CPU во время вычислений текущего блока. VRAM-процент задаёт отдельный бюджет из **свободной памяти CUDA** для хранения уже загруженных блоков между шагами удаления шума. Передача блока из RAM на GPU выполняется по запросу вычисляющего потока. При нулевом RAM-проценте фоновое чтение выключено; при нулевом VRAM-проценте выключен кэш. Эти бюджеты относятся только к дополнительным весам и буферам. Обязательные веса, активации и VAE требуют отдельной памяти.

Планировщик оставляет минимум 1 ГиБ доступной RAM и 6 ГиБ свободной VRAM при расчёте дополнительных бюджетов. Размер слота для 2B составляет 256 МиБ, для 13B — 544 МиБ; четыре RAM-слота резервируются для подготовки и передачи блоков. Малый бюджет может дать ноль слотов. В журнале `prefetch=on` показывает проценты, бюджеты в байтах и фактическую ёмкость в блоках. После этапа DiT кэш освобождается. Показатели использования GPU зависят также от текста, VAE и операций ввода-вывода.

## Портрет из фото на LTX-Video 13B Dev

Поместите официальный BF16-файл в models/ltxv-13b-0.9.8-dev/ltxv-13b-0.9.8-dev.safetensors. Параметр --models по-прежнему указывает на папку с T5 и токенизатором. Если файл 13B лежит в другом месте, задайте его полный путь переменной процесса ASTER_13B_CHECKPOINT. Рабочий процесс проверяет размер и SHA-256 весов.

```powershell
.\target\release\aster.exe quality --prompt-file examples/quality-evening-portrait.txt --output artifacts/my-13b.mp4 --worker target/ltx/release/aster-ltx.exe --models models/ltx-0.9.8-2b --profile reference13b-704-fullvae --seed 44 --image artifacts/quality-reference/portrait-crop.png
```

Для выбора по умолчанию укажите `ASTER_QUALITY_PROFILE=reference13b-704-fullvae` в .env в текущем каталоге. Явный --profile имеет приоритет. Для сравнения с 2B выберите reference704-fullvae и другой путь результата. Оба профиля генерируют 25 кадров 704×384 со скоростью воспроизведения 24 кадра/с.

Экспериментальный профиль 13B использует один проход, 30 шагов LinearQuadratic, CFG=1, STG=0, полный временной декод VAE и загрузку весов по блокам. Официальный многоэтапный процесс Lightricks дополнительно применяет динамическое управление и обученный апскейлер.

### Двухсекундный портрет с обученным апскейлером

Для того же запуска замените имя профиля на `reference13b-multiscale-2s` и укажите новый выходной файл. Рядом с весами 13B требуется `ltxv-spatial-upscaler-0.9.8.safetensors`. Профиль читает исходное фото, выполняет 27 шагов на первом масштабе, увеличивает латенты обученной сетью и выполняет 13 шагов уточнения. На выходе 48 кадров, 704×384, 24 кадра/с. Выводы об улучшении детализации требуют просмотра результата.

Параметры CFG-star, STG по значениям внимания, блоки и расписания взяты из официальной конфигурации 0.9.8 Dev. В Rust используются T5 Q5_K_M, генератор шума PCG32 и исходный текст без автоматического расширения. Совпадение с PyTorch по пикселям не проверялось.
