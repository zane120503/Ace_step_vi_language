#!/bin/bash
# Script chay training tren may remote
# Su dung: chmod +x run_train_remote.sh && bash run_train_remote.sh

echo "========================================"
echo "  Chay Training tren May Remote"
echo "========================================"
echo ""

# Kiem tra dataset
echo "[1/3] Kiem tra dataset..."
if [ ! -d "vi_lora_dataset" ]; then
    echo "[ERROR] Dataset vi_lora_dataset khong tim thay!"
    echo "        Can tao dataset truoc:"
    echo "        python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset"
    exit 1
fi
echo "  [OK] Dataset tim thay"
echo ""

# Kiem tra PyTorch
echo "[2/3] Kiem tra PyTorch..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "[ERROR] PyTorch chua cai dat!"
    exit 1
fi
if ! python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "[ERROR] PyTorch khong nhan dien duoc CUDA!"
    exit 1
fi
echo "  [OK] PyTorch da cai dat va nhan dien CUDA"
python3 -c "import torch; print(f'       GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Tao thu muc neu chua co
mkdir -p ./exps/checkpoints/vi_lora
mkdir -p ./exps/logs/vi_lora

# Chay training
echo "[3/3] Bat dau training..."
echo "  Tham so toi uu cho RTX 5060 Ti (16GB VRAM):"
echo "  - precision: 16 (FP16)"
echo "  - accumulate_grad_batches: 8"
echo "  - max_steps: 20000"
echo "  - num_workers: 4"
echo ""

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

echo ""
echo "========================================"
echo "  Training Hoan tat!"
echo "========================================"
echo ""

