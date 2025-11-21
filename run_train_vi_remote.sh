#!/bin/bash
# Script train LoRA tiếng Việt cho ACE-Step trên máy remote (Linux)
# Chạy: chmod +x run_train_vi_remote.sh && bash run_train_vi_remote.sh

set -e  # Exit on error

echo "========================================"
echo "  ACE-Step LoRA Training (Tiếng Việt)"
echo "========================================"
echo ""

# Kiểm tra virtual environment
if [ ! -d "/root/ace_step_env" ]; then
    echo "[ERROR] Virtual environment /root/ace_step_env khong ton tai!"
    echo "        Can tao virtual environment truoc:"
    echo "        python3 -m venv /root/ace_step_env"
    exit 1
fi

# Kích hoạt virtual environment
echo "[1/4] Kich hoat virtual environment..."
source /root/ace_step_env/bin/activate

if [ $? -ne 0 ]; then
    echo "[ERROR] Khong the kich hoat virtual environment!"
    exit 1
fi

echo "  [OK] Virtual environment da kich hoat"
echo ""

# Ép torchaudio ưu tiên backend soundfile
export TORCHAUDIO_USE_SOUNDFILE=1

# Kiểm tra dataset đã có chưa
echo "[2/4] Kiem tra dataset..."
if [ ! -d "vi_lora_dataset" ]; then
    echo "  Dataset vi_lora_dataset chua ton tai, dang tao..."
    
    # Kiểm tra data folder có file .mp3 chưa
    mp3_count=$(ls -1 data/*.mp3 2>/dev/null | wc -l)
    if [ "$mp3_count" -eq 0 ]; then
        echo "[ERROR] Khong tim thay file .mp3 trong folder data!"
        echo "        Can copy file .mp3 tu may Windows len may remote truoc:"
        echo "        scp D:\\ACE-Step\\data\\*.mp3 root@192.168.11.94:/root/ACE-Step/data/"
        exit 1
    fi
    
    echo "  Tim thay $mp3_count file .mp3"
    echo "  Dang tao dataset..."
    
    python3 convert2hf_dataset.py --data_dir "./data" --repeat_count 2000 --output_name "vi_lora_dataset"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Tao dataset that bai!"
        exit 1
    fi
    
    echo "  [OK] Dataset da tao thanh cong!"
else
    echo "  [OK] Dataset vi_lora_dataset da ton tai"
fi
echo ""

# Kiểm tra dataset có nội dung
if [ ! -f "vi_lora_dataset/data-00000-of-00001.arrow" ]; then
    echo "[ERROR] Dataset khong co noi dung!"
    echo "        Xoa va tao lai dataset:"
    echo "        rm -rf vi_lora_dataset"
    echo "        python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset"
    exit 1
fi

# Kiểm tra config
echo "[3/4] Kiem tra config..."
if [ ! -f "config/vi_lora_config.json" ]; then
    echo "[ERROR] File config/vi_lora_config.json khong tim thay!"
    exit 1
fi

echo "  [OK] Config file tim thay"
echo ""

# Kiểm tra GPU
echo "[4/4] Kiem tra GPU..."
if ! python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "[ERROR] GPU khong available!"
    exit 1
fi

gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
echo "  [OK] GPU: $gpu_name"
echo ""

# Tạo thư mục output nếu chưa có
mkdir -p ./exps/checkpoints/vi_lora
mkdir -p ./exps/logs/vi_lora

# Bắt đầu training
echo "========================================"
echo "  Bat dau Training LoRA..."
echo "========================================"
echo ""
echo "Tham so toi uu cho RTX 5060 Ti (16GB VRAM):"
echo "  - precision: 16 (FP16)"
echo "  - accumulate_grad_batches: 8"
echo "  - max_steps: 20000"
echo "  - num_workers: 4"
echo ""

# Chạy training
python3 trainer.py \
    --num_nodes 1 \
    --devices 1 \
    --dataset_path "./vi_lora_dataset" \
    --exp_name "vi_lora_small" \
    --lora_config_path "config/vi_lora_config.json" \
    --learning_rate 1e-4 \
    --accumulate_grad_batches 8 \
    --precision 16 \
    --num_workers 4 \
    --max_steps 20000 \
    --every_n_train_steps 500 \
    --shift 3.0 \
    --checkpoint_dir "./exps/checkpoints/vi_lora" \
    --logger_dir "./exps/logs/vi_lora" \
    --epochs -1 \
    --every_plot_step 2000 \
    --gradient_clip_val 0.5 \
    --gradient_clip_algorithm "norm"

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Training that bai!"
    echo "Kiem tra log tai: ./exps/logs/vi_lora"
    exit 1
fi

echo ""
echo "========================================"
echo "  Training hoan tat!"
echo "========================================"
echo "Checkpoint duoc luu tai: ./exps/checkpoints/vi_lora"
echo "Logs tai: ./exps/logs/vi_lora"
echo ""

