#!/bin/bash
# Script kiem tra moi truong tren may remote
# Su dung: chay script nay tren may remote de kiem tra

echo "========================================"
echo "  Kiem tra Moi truong May Remote"
echo "========================================"
echo ""

# Kiem tra OS
echo "[1/10] He dieu hanh:"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "  OS: $NAME"
    echo "  Version: $VERSION"
elif [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    echo "  OS: $DISTRIB_DESCRIPTION"
fi
echo ""

# Kiem tra Python
echo "[2/10] Python:"
if command -v python3 &> /dev/null; then
    echo "  ✓ python3: $(which python3)"
    python3 --version
elif command -v python &> /dev/null; then
    echo "  ✓ python: $(which python)"
    python --version
else
    echo "  ✗ Python khong tim thay"
fi
echo ""

# Kiem tra pip
echo "[3/10] pip:"
if command -v pip3 &> /dev/null; then
    echo "  ✓ pip3: $(which pip3)"
    pip3 --version
elif command -v pip &> /dev/null; then
    echo "  ✓ pip: $(which pip)"
    pip --version
else
    echo "  ✗ pip khong tim thay"
fi
echo ""

# Kiem tra Conda/Anaconda
echo "[4/10] Conda/Anaconda:"
if command -v conda &> /dev/null; then
    echo "  ✓ conda: $(which conda)"
    conda --version
    echo "  Environments:"
    conda env list
else
    echo "  ✗ Conda khong tim thay"
    echo "  → Can cai dat conda hoac miniconda"
fi
echo ""

# Kiem tra GPU
echo "[5/10] GPU (NVIDIA):"
if command -v nvidia-smi &> /dev/null; then
    echo "  ✓ nvidia-smi tim thay"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
    echo "  GPU details:"
    nvidia-smi
else
    echo "  ✗ nvidia-smi khong tim thay"
    echo "  → Co the chua cai NVIDIA driver"
fi
echo ""

# Kiem tra CUDA
echo "[6/10] CUDA:"
if command -v nvcc &> /dev/null; then
    echo "  ✓ nvcc: $(which nvcc)"
    nvcc --version
else
    echo "  ✗ nvcc khong tim thay"
    echo "  → Co the chua cai CUDA toolkit"
fi

# Kiem tra CUDA version tu nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    if [ ! -z "$CUDA_VERSION" ]; then
        echo "  CUDA Version (tu driver): $CUDA_VERSION"
    fi
fi
echo ""

# Kiem tra PyTorch
echo "[7/10] PyTorch:"
if python3 -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch da cai dat"
    python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
    python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
    if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        python3 -c "import torch; print(f'  CUDA version: {torch.version.cuda}')"
        python3 -c "import torch; print(f'  GPU count: {torch.cuda.device_count()}')"
        python3 -c "import torch; print(f'  GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    fi
else
    echo "  ✗ PyTorch chua cai dat"
fi
echo ""

# Kiem tra disk space
echo "[8/10] Disk space:"
df -h / | tail -1 | awk '{print "  Total: " $2 ", Available: " $4 ", Used: " $5}'
echo ""

# Kiem tra RAM
echo "[9/10] RAM:"
if command -v free &> /dev/null; then
    free -h | grep Mem | awk '{print "  Total: " $2 ", Available: " $7}'
fi
echo ""

# Kiem tra dependencies co ban
echo "[10/10] Dependencies co ban:"
python3 -c "import sys; print(f'  Python: {sys.version}')" 2>/dev/null || echo "  Python: Khong tim thay"

echo ""
echo "========================================"
echo "  Ket qua Kiem tra"
echo "========================================"
echo ""
echo "De cai dat dependencies, chay:"
echo "  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "Hoac cai conda truoc:"
echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
echo "  bash Miniconda3-latest-Linux-x86_64.sh"
echo ""

