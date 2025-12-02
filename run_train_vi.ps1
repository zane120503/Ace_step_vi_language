param(
    [switch]$DebugRun,
    [switch]$SkipDatasetConvert
)

# Script train LoRA tiếng Việt cho ACE-Step
# Chạy trong PowerShell: .\run_train_vi.ps1 [-DebugRun] [-SkipDatasetConvert]

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ACE-Step LoRA Training (Tiếng Việt)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra môi trường conda
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Đang khởi tạo conda..." -ForegroundColor Yellow
    & "C:\Users\admin\anaconda3\shell\condabin\conda-hook.ps1"
}

# Kích hoạt môi trường
Write-Host "[1/3] Kích hoạt môi trường ace_step..." -ForegroundColor Green
conda activate ace_step

if ($LASTEXITCODE -ne 0) {
    Write-Host "Lỗi: Không thể kích hoạt môi trường ace_step" -ForegroundColor Red
    exit 1
}

# Ép torchaudio ưu tiên backend soundfile để tránh phụ thuộc torchcodec
$env:TORCHAUDIO_USE_SOUNDFILE = "1"

if (-not $SkipDatasetConvert) {
    # Bước 1: Convert dataset
    Write-Host ""
    Write-Host "[2/3] Chuyển đổi dữ liệu sang HuggingFace dataset..." -ForegroundColor Green
    Write-Host "Chạy: python convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset" -ForegroundColor Yellow

    python convert2hf_dataset.py --data_dir "./data" --repeat_count 2000 --output_name "vi_lora_dataset"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Lỗi: Convert dataset thất bại!" -ForegroundColor Red
        exit 1
    }

    Write-Host "✓ Convert dataset thành công!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[2/3] Bỏ qua bước convert dataset (theo yêu cầu)" -ForegroundColor Yellow
}

if (-not (Test-Path "./vi_lora_dataset/dataset_info.json")) {
    Write-Host "⚠️ Không tìm thấy ./vi_lora_dataset/dataset_info.json - vui lòng kiểm tra lại dữ liệu!" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✓ Đã kiểm tra vi_lora_dataset: dataset_info.json tồn tại." -ForegroundColor Green
}

# Bước 2: Train LoRA
Write-Host ""
Write-Host "[3/3] Bắt đầu training LoRA..." -ForegroundColor Green

$learningRate = "1e-4"
$accumulateGrad = "4"
$precisionMode = "16"
$maxSteps = "20000"
$logEveryTrainSteps = "50"  # Giảm từ 100 xuống 50 để lưu checkpoint thường xuyên hơn
$gradStatsInterval = "0"
$activationStatsInterval = "0"
$batchSize = "1"
$debugMessage = "Chế độ thường"
$detectAnomalyArgs = @()

if ($DebugRun) {
    Write-Host "⚙️  DebugRun bật: cấu hình batch nhỏ, detect_anomaly, log gradient/activation chi tiết" -ForegroundColor Yellow
    $learningRate = "5e-5"
    $accumulateGrad = "1"
    $precisionMode = "32"
    $maxSteps = "200"
    $logEveryTrainSteps = "10"
    $gradStatsInterval = "1"
    $activationStatsInterval = "1"
    $debugMessage = "Chế độ DEBUG"
    $detectAnomalyArgs = @("--detect_anomaly")
} else {
    Write-Host "Chạy ở chế độ chuẩn (không detect_anomaly)" -ForegroundColor Cyan
}

Write-Host "Thông số huấn luyện ($debugMessage):" -ForegroundColor Yellow
Write-Host "  - learning_rate: $learningRate"
Write-Host "  - precision: $precisionMode"
Write-Host "  - batch_size: $batchSize"
Write-Host "  - accumulate_grad_batches: $accumulateGrad"
Write-Host "  - max_steps: $maxSteps"
Write-Host "  - every_n_train_steps: $logEveryTrainSteps"
Write-Host "  - grad_stats_interval: $gradStatsInterval"
Write-Host "  - activation_stats_interval: $activationStatsInterval"
Write-Host ""

# Tạo thư mục output nếu chưa có
New-Item -ItemType Directory -Force -Path "./exps/checkpoints/vi_lora" | Out-Null
New-Item -ItemType Directory -Force -Path "./exps/logs/vi_lora" | Out-Null

# Tự động tìm LoRA checkpoint mới nhất để resume (nếu có)
# Lưu ý: Code không còn tạo .ckpt file nữa, chỉ tạo LoRA adapter checkpoint (.safetensors)
$loraCheckpointDir = $null
$checkpointsDir = "./exps/logs/vi_lora/lightning_logs"
if (Test-Path $checkpointsDir) {
    # Tìm LoRA checkpoint mới nhất (thư mục chứa pytorch_lora_weights.safetensors)
    $loraCheckpoints = Get-ChildItem -Path $checkpointsDir -Recurse -Filter "pytorch_lora_weights.safetensors" | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 1
    
    if ($loraCheckpoints) {
        $loraCheckpointDir = $loraCheckpoints.DirectoryName
        Write-Host "✓ Tìm thấy LoRA checkpoint mới nhất để resume" -ForegroundColor Green
        Write-Host "  Checkpoint: $loraCheckpointDir" -ForegroundColor Yellow
        Write-Host "  File: $($loraCheckpoints.Name)" -ForegroundColor Yellow
    } else {
        Write-Host "ℹ Chưa có LoRA checkpoint, sẽ train từ đầu" -ForegroundColor Cyan
    }
} else {
    Write-Host "ℹ Chưa có LoRA checkpoint, sẽ train từ đầu" -ForegroundColor Cyan
}

# Lệnh train
$trainArgs = @(
    "--num_nodes", "1",
    "--devices", "1",
    "--dataset_path", "./vi_lora_dataset",
    "--exp_name", "vi_lora_small",
    "--lora_config_path", "config/vi_lora_config.json",
    "--learning_rate", $learningRate,
    "--accumulate_grad_batches", $accumulateGrad,
    "--precision", $precisionMode,
    "--batch_size", $batchSize,
    "--num_workers", "0",
    "--max_steps", $maxSteps,
    "--every_n_train_steps", $logEveryTrainSteps,
    "--shift", "3.0",
    "--checkpoint_dir", "./exps/checkpoints/vi_lora",
    "--logger_dir", "./exps/logs/vi_lora",
    "--epochs", "-1",
    "--every_plot_step", "2000",
    "--gradient_clip_val", "0.5",
    "--gradient_clip_algorithm", "norm",
    "--grad_stats_interval", $gradStatsInterval,
    "--activation_stats_interval", $activationStatsInterval
)

# Thêm --lora_checkpoint_dir nếu có LoRA checkpoint
# Lưu ý: Không dùng --ckpt_path nữa vì code không tạo .ckpt file
if ($loraCheckpointDir) {
    $trainArgs += "--lora_checkpoint_dir"
    $trainArgs += $loraCheckpointDir
}

# Thêm detect_anomaly khi bật debug
if ($detectAnomalyArgs.Count -gt 0) {
    $trainArgs += $detectAnomalyArgs
}

# Chạy lệnh train
& python trainer.py $trainArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Lỗi: Training thất bại!" -ForegroundColor Red
    Write-Host "Kiểm tra log tại: ./exps/logs/vi_lora" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Training hoàn tất!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Checkpoint được lưu tại: ./exps/checkpoints/vi_lora" -ForegroundColor Yellow
Write-Host "Logs tại: ./exps/logs/vi_lora" -ForegroundColor Yellow
Write-Host ""


