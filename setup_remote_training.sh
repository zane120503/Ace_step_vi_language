#!/bin/bash
# Script setup training tren may remote
# Su dung: chay script nay tren may remote de setup moi truong

echo "========================================"
echo "  Setup Moi truong Training tren May Remote"
echo "========================================"
echo ""

# Kiem tra Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 khong tim thay! Can cai dat Python3 truoc."
    exit 1
fi

echo "[1/6] Python: $(python3 --version)"
echo ""

# Kiem tra pip
if ! command -v pip3 &> /dev/null; then
    echo "[ERROR] pip3 khong tim thay! Can cai dat pip truoc."
    exit 1
fi

echo "[2/6] pip: $(pip3 --version)"
echo ""

# Kiem tra GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "[WARNING] nvidia-smi khong tim thay! Co the GPU chua duoc cai dat dung."
    exit 1
fi

echo "[3/6] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Kiem tra CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "[4/6] CUDA Version: $CUDA_VERSION"
echo ""

# Cai dat PyTorch voi CUDA
echo "[5/6] Cai dat PyTorch voi CUDA..."
if python3 -c "import torch" 2>/dev/null; then
    echo "  PyTorch da cai dat, kiem tra version..."
    python3 -c "import torch; print(f'  Current: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
    
    if ! python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        echo "  PyTorch chua co CUDA support, cai dat lai..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo "  Cai dat PyTorch voi CUDA 12.1..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi
echo ""

# Cai dat dependencies
echo "[6/6] Cai dat dependencies..."
pip3 install -r requirements.txt

echo ""
echo "========================================"
echo "  Setup Hoan tat!"
echo "========================================"
echo ""
echo "De kiem tra lai, chay:"
echo "  python3 -c \"import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())\""
echo ""
echo "De chay training:"
echo "  python3 trainer.py --dataset_path ./vi_lora_dataset --exp_name vi_lora_small ..."
echo ""

