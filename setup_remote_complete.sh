#!/bin/bash
# Script setup hoan chinh moi truong training tren may remote
# Su dung: chmod +x setup_remote_complete.sh && bash setup_remote_complete.sh

set -e  # Exit on error

echo "========================================"
echo "  Setup Hoan Chinh Moi Truong Training"
echo "========================================"
echo ""

# 1. Kiem tra Python
echo "[1/6] Kiem tra Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 khong tim thay!"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "  [OK] $PYTHON_VERSION"
echo ""

# 2. Kiem tra GPU
echo "[2/6] Kiem tra GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi khong tim thay!"
    exit 1
fi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "  [OK] GPU tim thay"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  CUDA Version: $CUDA_VERSION"
echo ""

# 3. Cai dat PyTorch
echo "[3/6] Cai dat PyTorch voi CUDA..."
if python3 -c "import torch" 2>/dev/null; then
    echo "  [INFO] PyTorch da cai dat"
    python3 -c "import torch; print(f'       Version: {torch.__version__}'); print(f'       CUDA: {torch.cuda.is_available()}')"
    if ! python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        echo "  [WARNING] PyTorch chua co CUDA support, cai dat lai..."
        pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --upgrade
    fi
else
    echo "  Dang cai dat PyTorch nightly voi CUDA 12.1..."
    echo "  (Python 3.13 can PyTorch nightly, co the mat nhieu thoi gian...)"
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
fi
echo ""

# 4. Kiem tra PyTorch
echo "[4/6] Kiem tra PyTorch..."
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

# 5. Clone code tu GitHub
echo "[5/6] Clone code tu GitHub..."
cd /root
if [ ! -d "ACE-Step" ]; then
    echo "  Dang clone code..."
    git clone https://github.com/zane120503/Ace_step_vi_language.git ACE-Step
    cd ACE-Step
else
    echo "  [INFO] Folder ACE-Step da ton tai, cap nhat..."
    cd ACE-Step
    git pull || echo "  [WARNING] Khong the cap nhat (co the khong phai git repo)"
fi
echo ""

# 6. Cai dat dependencies
echo "[6/6] Cai dat dependencies..."
if [ -f "requirements.txt" ]; then
    echo "  Dang cai dat tu requirements.txt..."
    echo "  (Co the mat nhieu thoi gian, vui long cho...)"
    pip3 install -r requirements.txt
    echo "  [OK] Dependencies da cai dat"
else
    echo "  [WARNING] requirements.txt khong tim thay!"
fi
echo ""

echo "========================================"
echo "  Setup Hoan tat!"
echo "========================================"
echo ""
echo "Cac buoc tiep theo:"
echo "1. Copy data folder len may remote (tu may Windows):"
echo "   scp -r D:\\ACE-Step\\data root@192.168.11.94:/root/ACE-Step/"
echo ""
echo "2. Tao dataset tren may remote:"
echo "   cd /root/ACE-Step"
echo "   python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset"
echo ""
echo "3. Chay training:"
echo "   tmux new -s training"
echo "   cd /root/ACE-Step"
echo "   python3 trainer.py --dataset_path ./vi_lora_dataset --exp_name vi_lora_small ..."
echo ""

