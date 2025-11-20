# Script copy data folder len may remote
# Su dung: .\copy_data_to_remote.ps1 -RemoteHost "root@192.168.11.94" -RemotePath "/root/ACE-Step"

param(
    [Parameter(Mandatory=$true)]
    [string]$RemoteHost,
    
    [Parameter(Mandatory=$false)]
    [string]$RemotePath = "/root/ACE-Step"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Copy Data len May Remote" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Remote Host: $RemoteHost" -ForegroundColor Yellow
Write-Host "Remote Path: $RemotePath" -ForegroundColor Yellow
Write-Host ""

# Kiem tra scp
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] SCP khong tim thay! Can cai dat OpenSSH hoac dung WSL/Git Bash." -ForegroundColor Red
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

Write-Host "[1/3] Kiem tra data folder..." -ForegroundColor Green
Write-Host "  Tim thay $mp3Count file MP3 trong folder data" -ForegroundColor Yellow

if ($mp3Count -eq 0) {
    Write-Host "[WARNING] Khong tim thay file MP3 nao!" -ForegroundColor Yellow
    Write-Host "  Chi co file text (.txt) se duoc copy" -ForegroundColor Yellow
}

# Dem so file text
$txtFiles = Get-ChildItem -Path ".\data" -Filter "*_prompt.txt" -ErrorAction SilentlyContinue
$txtCount = $txtFiles.Count
Write-Host "  Tim thay $txtCount file prompt.txt" -ForegroundColor Yellow
Write-Host ""

# Xac nhan
Write-Host "[2/3] Copy data folder len may remote..." -ForegroundColor Green
Write-Host "  Data folder co $mp3Count file MP3 (rat lon, co the mat nhieu thoi gian)" -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "Ban co muon tiep tuc? (Y/N)"

if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Da huy!" -ForegroundColor Yellow
    exit 0
}

# Copy data folder
Write-Host ""
Write-Host "  Dang copy data folder... (co the mat nhieu thoi gian, vui long cho...)" -ForegroundColor Cyan
Write-Host ""

# Tao thu muc tren may remote truoc
ssh "$RemoteHost" "mkdir -p $RemotePath/data"

# Copy data folder
scp -r ".\data\*" "${RemoteHost}:${RemotePath}/data/"

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
    Write-Host "3. Tao dataset: python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "[ERROR] Copy that bai!" -ForegroundColor Red
    Write-Host "  Kiem tra ket noi SSH hoac dung cach khac" -ForegroundColor Yellow
    Write-Host ""
}

