# Script copy CHI file .mp3 len may remote
# Su dung: .\copy_mp3_only.ps1 -RemoteHost "root@192.168.11.94" -RemotePath "/root/ACE-Step"

param(
    [Parameter(Mandatory=$true)]
    [string]$RemoteHost,
    
    [Parameter(Mandatory=$false)]
    [string]$RemotePath = "/root/ACE-Step"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Copy CHI File .mp3 len May Remote" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Remote Host: $RemoteHost" -ForegroundColor Yellow
Write-Host "Remote Path: $RemotePath" -ForegroundColor Yellow
Write-Host ""

# Kiem tra scp
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] SCP khong tim thay! Can cai dat OpenSSH." -ForegroundColor Red
    exit 1
}

# Kiem tra data folder
if (-not (Test-Path ".\data")) {
    Write-Host "[ERROR] Folder data khong tim thay!" -ForegroundColor Red
    exit 1
}

# Dem so file mp3
$mp3Files = Get-ChildItem -Path ".\data" -Filter "*.mp3" -ErrorAction SilentlyContinue
$mp3Count = $mp3Files.Count

Write-Host "[1/3] Kiem tra file .mp3..." -ForegroundColor Green
if ($mp3Count -eq 0) {
    Write-Host "[ERROR] Khong tim thay file MP3 nao trong folder data!" -ForegroundColor Red
    exit 1
}

Write-Host "  Tim thay $mp3Count file MP3" -ForegroundColor Yellow

# Tinh tong dung luong
$totalSize = ($mp3Files | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "  Tong dung luong: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Yellow
Write-Host ""

# Xac nhan
Write-Host "[2/3] Copy file .mp3 len may remote..." -ForegroundColor Green
Write-Host "  Se copy $mp3Count file MP3 (co the mat nhieu thoi gian)" -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "Ban co muon tiep tuc? (Y/N)"

if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Da huy!" -ForegroundColor Yellow
    exit 0
}

# Kiem tra ket noi
Write-Host ""
Write-Host "  Dang kiem tra ket noi den may remote..." -ForegroundColor Cyan
$testConnection = ssh -o ConnectTimeout=5 "$RemoteHost" "echo 'OK'" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Khong the ket noi bang SSH key. Ban se can nhap password khi copy." -ForegroundColor Yellow
}

# Tao thu muc tren may remote truoc
Write-Host "  Dang tao thu muc tren may remote..." -ForegroundColor Cyan
ssh "$RemoteHost" "mkdir -p $RemotePath/data" 2>&1 | Out-Null

# Copy CHI file .mp3
Write-Host ""
Write-Host "  Dang copy file .mp3... (co the mat nhieu thoi gian, vui long cho...)" -ForegroundColor Cyan
Write-Host "  (Dang copy $mp3Count file MP3...)" -ForegroundColor Yellow
Write-Host ""

# Copy tat ca file .mp3
scp ".\data\*.mp3" "${RemoteHost}:${RemotePath}/data/"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[3/3] Copy hoan tat!" -ForegroundColor Green
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Hoan tat!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Cac buoc tiep theo tren may remote:" -ForegroundColor Yellow
    Write-Host "1. SSH vao may remote: ssh $RemoteHost" -ForegroundColor Cyan
    Write-Host "2. Di chuyen den thu muc: cd $RemotePath" -ForegroundColor Cyan
    Write-Host "3. Kiem tra file .mp3: ls -lh data/*.mp3 | head -5" -ForegroundColor Cyan
    Write-Host "4. Dem so file .mp3: ls -1 data/*.mp3 | wc -l" -ForegroundColor Cyan
    Write-Host "5. Tao dataset: python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "[ERROR] Copy that bai!" -ForegroundColor Red
    Write-Host "  Kiem tra ket noi SSH hoac dung cach khac" -ForegroundColor Yellow
    Write-Host ""
}

