# Script activate conda environment
# Su dung: .\activate_env.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Activate Conda Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kiem tra conda co san khong
if (Test-Path "C:\Users\admin\anaconda3\Scripts\conda.exe") {
    Write-Host "[1/2] Khoi tao conda..." -ForegroundColor Yellow
    
    # Khoi tao conda trong PowerShell
    & "C:\Users\admin\anaconda3\Scripts\conda.exe" init powershell
    
    Write-Host "[2/2] Activate environment ace_step..." -ForegroundColor Green
    
    # Activate environment
    & "C:\Users\admin\anaconda3\Scripts\conda.exe" activate ace_step
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  ✓ Da activate environment!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "De su dung, chay lenh sau:" -ForegroundColor Yellow
    Write-Host "  conda activate ace_step" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Loi: Khong tim thay conda tai C:\Users\admin\anaconda3\Scripts\conda.exe" -ForegroundColor Red
    Write-Host "Vui long kiem tra duong dan anaconda3 cua ban." -ForegroundColor Yellow
}

