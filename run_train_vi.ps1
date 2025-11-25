# Script train LoRA tiếng Việt cho ACE-Step
# Chạy trong PowerShell: .\run_train_vi.ps1

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

# Bước 2: Train LoRA
Write-Host ""
Write-Host "[3/3] Bắt đầu training LoRA..." -ForegroundColor Green
Write-Host "Tham số tối ưu cho RTX 3050 (6GB VRAM):" -ForegroundColor Yellow
Write-Host "  - precision: 16 (FP16)" -ForegroundColor Yellow
Write-Host "  - accumulate_grad_batches: 4" -ForegroundColor Yellow
Write-Host "  - max_steps: 20000" -ForegroundColor Yellow
Write-Host "  - checkpoint mỗi: 100 steps" -ForegroundColor Yellow
Write-Host ""

# Tạo thư mục output nếu chưa có
New-Item -ItemType Directory -Force -Path "./exps/checkpoints/vi_lora" | Out-Null
New-Item -ItemType Directory -Force -Path "./exps/logs/vi_lora" | Out-Null

# Tự động tìm checkpoint mới nhất để resume (nếu có)
$checkpointPath = $null
$checkpointsDir = "./exps/logs/vi_lora/lightning_logs"
if (Test-Path $checkpointsDir) {
    $latestCheckpoint = Get-ChildItem -Path $checkpointsDir -Recurse -Filter "*.ckpt" | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 1
    
    if ($latestCheckpoint) {
        $checkpointPath = $latestCheckpoint.FullName
        Write-Host "✓ Tìm thấy checkpoint mới nhất để resume" -ForegroundColor Green
        Write-Host "  Checkpoint: $checkpointPath" -ForegroundColor Yellow
    } else {
        Write-Host "ℹ Chưa có checkpoint, sẽ train từ đầu" -ForegroundColor Cyan
    }
} else {
    Write-Host "ℹ Chưa có checkpoint, sẽ train từ đầu" -ForegroundColor Cyan
}

# Lệnh train
$trainArgs = @(
    "--num_nodes", "1",
    "--devices", "1",
    "--dataset_path", "./vi_lora_dataset",
    "--exp_name", "vi_lora_small",
    "--lora_config_path", "config/vi_lora_config.json",
    "--learning_rate", "1e-4",
    "--accumulate_grad_batches", "4",
    "--precision", "16",
    "--num_workers", "0",
    "--max_steps", "20000",
    "--every_n_train_steps", "100",
    "--shift", "3.0",
    "--checkpoint_dir", "./exps/checkpoints/vi_lora",
    "--logger_dir", "./exps/logs/vi_lora",
    "--epochs", "-1",
    "--every_plot_step", "2000",
    "--gradient_clip_val", "0.5",
    "--gradient_clip_algorithm", "norm"
)

# Thêm --ckpt_path nếu có checkpoint
if ($checkpointPath) {
    $trainArgs += "--ckpt_path"
    $trainArgs += $checkpointPath
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


