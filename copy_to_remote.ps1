# Script copy code len may remote
# Su dung: .\copy_to_remote.ps1 -RemoteHost "user@remote-ip" -RemotePath "~/ACE-Step"

param(
    [Parameter(Mandatory=$true)]
    [string]$RemoteHost,
    
    [Parameter(Mandatory=$false)]
    [string]$RemotePath = "~/ACE-Step"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Copy Code len May Remote" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Remote Host: $RemoteHost" -ForegroundColor Yellow
Write-Host "Remote Path: $RemotePath" -ForegroundColor Yellow
Write-Host ""

# Kiem tra ssh co san khong
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] SSH khong tim thay! Can cai dat OpenSSH hoac dung scp." -ForegroundColor Red
    exit 1
}

# Kiem tra scp
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] SCP khong tim thay! Can cai dat OpenSSH." -ForegroundColor Red
    exit 1
}

Write-Host "[1/4] Kiem tra ket noi den may remote..." -ForegroundColor Green
$testConnection = ssh -o ConnectTimeout=5 -o BatchMode=yes "$RemoteHost" "echo 'OK'" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Khong the ket noi bang SSH key. Ban co the can nhap password." -ForegroundColor Yellow
}

Write-Host "[2/4] Tao thu muc tren may remote..." -ForegroundColor Green
ssh "$RemoteHost" "mkdir -p $RemotePath"

Write-Host "[3/4] Copy code len may remote..." -ForegroundColor Green
Write-Host "Dang copy... (co the mat nhieu thoi gian)" -ForegroundColor Yellow

# Copy cac file/folder can thiet
$itemsToCopy = @(
    "acestep",
    "config",
    "trainer.py",
    "requirements.txt",
    "setup.py",
    "run_train_vi.ps1",
    "check_training_progress.py"
)

foreach ($item in $itemsToCopy) {
    if (Test-Path $item) {
        Write-Host "  Copying: $item" -ForegroundColor Cyan
        scp -r "$item" "${RemoteHost}:${RemotePath}/"
    } else {
        Write-Host "  [WARNING] $item khong ton tai, bo qua" -ForegroundColor Yellow
    }
}

Write-Host "[4/4] Copy dataset (neu co)..." -ForegroundColor Green
if (Test-Path "vi_lora_dataset") {
    Write-Host "  [WARNING] Dataset rat lon, ban co muon copy khong? (Y/N)" -ForegroundColor Yellow
    $confirm = Read-Host
    if ($confirm -eq "Y" -or $confirm -eq "y") {
        Write-Host "  Copying vi_lora_dataset... (rat lon, co the mat nhieu thoi gian)" -ForegroundColor Cyan
        scp -r "vi_lora_dataset" "${RemoteHost}:${RemotePath}/"
    }
} else {
    Write-Host "  Dataset khong ton tai, bo qua" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Hoan tat!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "De SSH vao may remote va setup:" -ForegroundColor Yellow
Write-Host "  ssh $RemoteHost" -ForegroundColor Cyan
Write-Host "  cd $RemotePath" -ForegroundColor Cyan
Write-Host "  chmod +x setup_remote_training.sh" -ForegroundColor Cyan
Write-Host "  ./setup_remote_training.sh" -ForegroundColor Cyan
Write-Host ""

