# Локальная проверка учителей

Запускайте из корня проекта в PowerShell. Веса и DAVIS хранятся локально. Для каждой новой выгрузки указывайте свободный путь. Depth, SAM и CoTracker проверены на CPU. Для Qwen проверен режим BF16 на RTX 4070. Используйте окружение сборки CUDA проекта. Команда предварительно проверяет запас видеопамяти (10 752 МиБ). Экспериментальный CPU-режим без флага --cuda использовал до 26,9 ГБ RAM и был остановлен до получения подписи. Обычный запуск LTX и команда обучения RGB-модуля доступны отдельно.

```powershell
$d = "artifacts/teachers/davis2017/DAVIS"
$seq = "$d/JPEGImages/480p/blackswan"
$ann = "$d/Annotations/480p/blackswan"
$m = "models/teachers"
cargo run --locked --release --manifest-path integrations/ltx/Cargo.toml -- depth-frame "$m/depth-v2-small/depth_anything_v2_vits.pth" "$seq/00001.jpg" depth.safetensors
cargo run --locked --release --manifest-path integrations/sam2/Cargo.toml -- "$m/sam2/sam2.1_hiera_tiny.pt" $seq "$ann/00000.png" 50 masks-blackswan
cargo run --locked --release --manifest-path integrations/ltx/Cargo.toml -- davis-iou $d blackswan masks-blackswan
cargo run --locked --release --manifest-path integrations/ltx/Cargo.toml -- cotracker-features "$m/cotracker3/scaled_offline.pth" "$seq/00001.jpg" descriptors.safetensors
cargo run --locked --release --manifest-path integrations/ltx/Cargo.toml -- cotracker-features "$m/cotracker3/scaled_offline.pth" "$seq/00002.jpg" descriptors2.safetensors
cargo run --locked --release --manifest-path integrations/ltx/Cargo.toml -- cotracker-pair "$m/cotracker3/scaled_offline.pth" descriptors.safetensors descriptors2.safetensors 268 73
cargo run --locked --release --manifest-path integrations/ltx/Cargo.toml --features teacher-cuda -- qwen-caption "$m/qwen3-vl" "$seq/00001.jpg" --cuda
```

- Depth Anything V2 Small сохраняет относительную глубину центрального квадратного кропа. Размер исходника, координаты кропа и SHA-256 записаны в метаданные. Для сопоставления с полноразмерной маской сначала учтите кроп.
- SAM 2.1 Tiny переносит ID объектов из первой маски на последующие кадры. Оценщик сравнивает ID с DAVIS val и пропускает фон при выборе объектов, разметку void, первый и последний кадры. Выводится только региональная метрика J.
- CoTracker3 загружает обученный пространственный энкодер fnet. Выход содержит 128 признаков на ячейку сетки 96×128. Для двух карт одного видео cotracker-pair выполняет шесть уточнений с корреляциями четырёх масштабов, временным и пространственным вниманием, выдаёт координаты, видимость и уверенность одной точки. Координаты относятся к сетке 512×384. Проверены соседние кадры, одинаковый кадр и известный сдвиг. Длинные треки и окклюзии требуют отдельной оценки. Веса и исследовательское применение ограничены CC BY-NC 4.0.
- Qwen получает один RGB-кадр и короткий запрос, затем декодирует до 40 токенов жадным выбором. Подпись требует проверки человеком перед включением в тренировочные данные.
- Последовательность blackswan из val зарезервирована для оценки. При подготовке обучающих примеров храните все кадры одного исходного ролика в одном сплите и проверяйте права на исходное видео.

LTX генерирует видео. Для существующего RGB-ученика Real-ESRGAN создаёт целевые RGB-кадры. Новые типы разметки доступны через отдельные команды. Интеграция этих целей в функцию потерь ученика требует отдельного эксперимента.
