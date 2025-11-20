#!/bin/bash
# Script setup moi truong training tren may remote
# Su dung: chmod +x setup_remote.sh && ./setup_remote.sh

set -e  # Exit on error

echo "========================================"
echo "  Setup Moi truong Training"
echo "========================================"
echo ""

# 1. Kiem tra Python
echo "[1/7] Kiem tra Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 khong tim thay!"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "  [OK] $PYTHON_VERSION"
echo ""

# 2. Kiem tra pip
echo "[2/7] Kiem tra pip..."
if ! command -v pip3 &> /dev/null; then
    echo "[ERROR] pip3 khong tim thay!"
    exit 1
fi
echo "  [OK] $(pip3 --version)"
echo ""

# 3. Kiem tra GPU
echo "[3/7] Kiem tra GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi khong tim thay!"
    exit 1
fi
echo "  [OK] GPU tim thay:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "  CUDA Version: $CUDA_VERSION"
echo ""

# 4. Cai dat PyTorch
echo "[4/7] Cai dat PyTorch voi CUDA..."
if python3 -c "import torch" 2>/dev/null; then
    echo "  [INFO] PyTorch da cai dat"
    python3 -c "import torch; print(f'       Version: {torch.__version__}'); print(f'       CUDA: {torch.cuda.is_available()}')"
    if ! python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        echo "  [WARNING] PyTorch chua co CUDA support, cai dat lai..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
    fi
else
    echo "  Dang cai dat PyTorch voi CUDA 12.1..."
    echo "  (Co the mat nhieu thoi gian, vui long cho...)"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    if [ $? -eq 0 ]; then
        echo "  [OK] PyTorch da cai dat thanh cong"
    else
        echo "  [ERROR] Cai dat PyTorch that bai!"
        echo "  Co the Python 3.13 chua duoc PyTorch ho tro day du"
        echo "  Thu cai dat PyTorch nightly:"
        echo "  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121"
        exit 1
    fi
fi
echo ""

# 5. Kiem tra PyTorch
echo "[5/7] Kiem tra PyTorch..."
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" || {
    echo "  [ERROR] PyTorch khong hoat dong!"
    exit 1
}
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python3 -c "import torch; print(f'  CUDA version: {torch.version.cuda}'); print(f'  GPU count: {torch.cuda.device_count()}'); print(f'  GPU name: {torch.cuda.get_device_name(0)}')"
else
    echo "  [WARNING] PyTorch khong nhan dien duoc CUDA!"
fi
echo ""

# 6. Clone code tu GitHub (neu chua co)
echo "[6/7] Kiem tra code..."
if [ ! -d "ACE-Step" ]; then
    echo "  Clone code tu GitHub..."
    cd /root
    git clone https://github.com/zane120503/Ace_step_vi_language.git ACE-Step
    cd ACE-Step
else
    echo "  [OK] Folder ACE-Step da ton tai"
    cd ACE-Step
    echo "  Cap nhat code..."
    git pull || echo "  [WARNING] Khong the cap nhat (co the khong phai git repo)"
fi
echo ""

# 7. Cai dat dependencies
echo "[7/7] Cai dat dependencies..."
if [ -f "requirements.txt" ]; then
    echo "  Dang cai dat tu requirements.txt..."
    echo "  (Co the mat nhieu thoi gian, vui long cho...)"
    pip3 install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "  [OK] Dependencies da cai dat thanh cong"
    else
        echo "  [WARNING] Co mot so dependencies cai dat that bai, thu cai lai..."
        pip3 install -r requirements.txt --upgrade
    fi
else
    echo "  [WARNING] requirements.txt khong tim thay!"
fi
echo ""

echo "========================================"
echo "  Setup Hoan tat!"
echo "========================================"
echo ""
echo "Cac buoc tiep theo:"
echo "1. Tao dataset (neu chua co):"
echo "   cd /root/ACE-Step"
echo "   python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset"
echo ""
echo "2. Chay training:"
echo "   cd /root/ACE-Step"
echo "   python3 trainer.py --dataset_path ./vi_lora_dataset --exp_name vi_lora_small ..."
echo ""
echo "3. De giu training chay sau khi dong SSH, dung tmux hoac screen:"
echo "   tmux new -s training"
echo "   # Trong tmux, chay training"
echo "   # Tach khoi tmux: Ctrl+B, sau do D"
echo "   # Xem lai: tmux attach -t training"
echo ""

