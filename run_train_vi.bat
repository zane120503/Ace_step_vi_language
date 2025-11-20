@echo off
REM Script train LoRA tiếng Việt cho ACE-Step (Windows Batch)
REM Chạy bằng cách double-click hoặc: run_train_vi.bat

echo ========================================
echo   ACE-Step LoRA Training (Tiếng Việt)
echo ========================================
echo.

REM Kích hoạt conda và môi trường
call C:\Users\admin\anaconda3\Scripts\activate.bat ace_step
if errorlevel 1 (
    echo Loi: Khong the kich hoat moi truong ace_step
    pause
    exit /b 1
)

echo [1/3] Kich hoat moi truong ace_step thanh cong!
echo.

REM Bước 1: Convert dataset
echo [2/3] Chuyen doi du lieu sang HuggingFace dataset...
echo Chay: python convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset
echo.

python convert2hf_dataset.py --data_dir "./data" --repeat_count 2000 --output_name "vi_lora_dataset"
if errorlevel 1 (
    echo Loi: Convert dataset that bai!
    pause
    exit /b 1
)

echo Convert dataset thanh cong!
echo.

REM Tạo thư mục output
if not exist "exps\checkpoints\vi_lora" mkdir "exps\checkpoints\vi_lora"
if not exist "exps\logs\vi_lora" mkdir "exps\logs\vi_lora"

REM Bước 2: Train LoRA
echo [3/3] Bat dau training LoRA...
echo Tham so toi uu cho RTX 3050 (6GB VRAM):
echo   - precision: 16 (FP16)
echo   - accumulate_grad_batches: 8
echo   - max_steps: 20000
echo.

python trainer.py ^
    --num_nodes 1 ^
    --devices 1 ^
    --dataset_path "./vi_lora_dataset" ^
    --exp_name "vi_lora_small" ^
    --lora_config_path "config/vi_lora_config.json" ^
    --learning_rate 1e-4 ^
    --accumulate_grad_batches 8 ^
    --precision 16 ^
    --num_workers 2 ^
    --max_steps 20000 ^
    --every_n_train_steps 500 ^
    --shift 3.0 ^
    --checkpoint_dir "./exps/checkpoints/vi_lora" ^
    --logger_dir "./exps/logs/vi_lora" ^
    --epochs -1 ^
    --every_plot_step 2000 ^
    --val_check_interval None ^
    --gradient_clip_val 0.5 ^
    --gradient_clip_algorithm "norm"

if errorlevel 1 (
    echo.
    echo Loi: Training that bai!
    echo Kiem tra log tai: ./exps/logs/vi_lora
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Training hoan tat!
echo ========================================
echo Checkpoint duoc luu tai: ./exps/checkpoints/vi_lora
echo Logs tai: ./exps/logs/vi_lora
echo.
pause

