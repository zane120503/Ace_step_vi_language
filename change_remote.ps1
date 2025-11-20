# Script thay doi Git remote va push len repo moi
# Su dung: .\change_remote.ps1 -NewRepoUrl "https://github.com/username/repo-name.git"

param(
    [Parameter(Mandatory=$true)]
    [string]$NewRepoUrl
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Thay doi Git Remote" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kiem tra xem co trong git repo khong
if (-not (Test-Path ".git")) {
    Write-Host "Loi: Khong phai git repository!" -ForegroundColor Red
    exit 1
}

# Hien thi remote hien tai
Write-Host "[1/4] Remote hien tai:" -ForegroundColor Yellow
git remote -v
Write-Host ""

# Thay doi remote URL
Write-Host "[2/4] Dang thay doi remote URL thanh:" -ForegroundColor Green
Write-Host "  $NewRepoUrl" -ForegroundColor Cyan
git remote set-url origin $NewRepoUrl

# Xac nhan da thay doi
Write-Host ""
Write-Host "[3/4] Remote moi:" -ForegroundColor Green
git remote -v
Write-Host ""

# Kiem tra branch hien tai
$currentBranch = git branch --show-current
Write-Host "[4/4] Branch hien tai: $currentBranch" -ForegroundColor Yellow
Write-Host ""

# Thong bao hoan tat
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✓ Da thay doi remote URL!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "De push code len repo moi, chay lenh sau:" -ForegroundColor Yellow
$pushCmd = "  git push -u origin " + $currentBranch
Write-Host $pushCmd -ForegroundColor Cyan
Write-Host ""
