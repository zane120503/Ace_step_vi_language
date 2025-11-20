#!/bin/bash
# Script kiem tra moi truong tren may remote
# Chay: bash check_remote.sh hoac chmod +x check_remote.sh && ./check_remote.sh

echo "========================================"
echo "  Kiem tra Moi truong May Remote"
echo "========================================"
echo ""

# 1. OS
echo "[1/8] He dieu hanh:"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "  OS: $NAME $VERSION"
    echo "  Kernel: $(uname -r)"
else
    echo "  OS: $(uname -s) $(uname -r)"
fi
echo ""

# 2. Python
echo "[2/8] Python:"
if command -v python3 &> /dev/null; then
    PYTHON3=$(which python3)
    PYTHON3_VERSION=$(python3 --version 2>&1)
    echo "  [OK] python3: $PYTHON3"
    echo "       Version: $PYTHON3_VERSION"
elif command -v python &> /dev/null; then
    PYTHON=$(which python)
    PYTHON_VERSION=$(python --version 2>&1)
    echo "  [OK] python: $PYTHON"
    echo "       Version: $PYTHON_VERSION"
else
    echo "  [ERROR] Python khong tim thay!"
    echo "          Can cai dat: sudo apt install python3 python3-pip"
fi
echo ""

# 3. pip
echo "[3/8] pip:"
if command -v pip3 &> /dev/null; then
    PIP3=$(which pip3)
    PIP3_VERSION=$(pip3 --version 2>&1 | head -1)
    echo "  [OK] pip3: $PIP3"
    echo "       $PIP3_VERSION"
elif command -v pip &> /dev/null; then
    PIP=$(which pip)
    PIP_VERSION=$(pip --version 2>&1 | head -1)
    echo "  [OK] pip: $PIP"
    echo "       $PIP_VERSION"
else
    echo "  [ERROR] pip khong tim thay!"
    echo "          Can cai dat: sudo apt install python3-pip"
fi
echo ""

# 4. Conda
echo "[4/8] Conda/Anaconda:"
if command -v conda &> /dev/null; then
    CONDA=$(which conda)
    CONDA_VERSION=$(conda --version 2>&1)
    echo "  [OK] conda: $CONDA"
    echo "       Version: $CONDA_VERSION"
    echo "  Environments:"
    conda env list 2>/dev/null | head -5
else
    echo "  [INFO] Conda khong tim thay (khong bat buoc)"
    echo "         Neu can, cai dat:"
    echo "         wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "         bash Miniconda3-latest-Linux-x86_64.sh"
fi
echo ""

# 5. GPU
echo "[5/8] GPU (NVIDIA):"
if command -v nvidia-smi &> /dev/null; then
    echo "  [OK] nvidia-smi tim thay"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s\n       Memory: %s MB\n       Driver: %s\n", $1, $2, $3, $4}'
    echo ""
    CUDA_VERSION_DRIVER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    if [ ! -z "$CUDA_VERSION_DRIVER" ]; then
        echo "  CUDA Version (tu driver): $CUDA_VERSION_DRIVER"
    fi
else
    echo "  [ERROR] nvidia-smi khong tim thay!"
    echo "          Co the GPU chua duoc cai dat hoac driver chua cai"
fi
echo ""

# 6. CUDA Toolkit
echo "[6/8] CUDA Toolkit:"
if command -v nvcc &> /dev/null; then
    NVCC=$(which nvcc)
    NVCC_VERSION=$(nvcc --version 2>&1 | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "  [OK] nvcc: $NVCC"
    echo "       Version: $NVCC_VERSION"
else
    echo "  [INFO] nvcc khong tim thay (khong bat buoc)"
    echo "         PyTorch co the tu download CUDA libraries"
fi
echo ""

# 7. PyTorch
echo "[7/8] PyTorch:"
if python3 -c "import torch" 2>/dev/null; then
    echo "  [OK] PyTorch da cai dat"
    python3 -c "import torch; print(f'       PyTorch version: {torch.__version__}')" 2>/dev/null
    python3 -c "import torch; print(f'       CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
    if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
        python3 -c "import torch; print(f'       CUDA version: {torch.version.cuda}')" 2>/dev/null
        python3 -c "import torch; print(f'       GPU count: {torch.cuda.device_count()}')" 2>/dev/null
        python3 -c "import torch; print(f'       GPU name: {torch.cuda.get_device_name(0)}')" 2>/dev/null
    else
        echo "       [WARNING] PyTorch khong nhan dien duoc CUDA"
        echo "                Can cai lai PyTorch voi CUDA support"
    fi
else
    echo "  [ERROR] PyTorch chua cai dat"
    echo "          De cai dat:"
    echo "          pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
fi
echo ""

# 8. Disk & RAM
echo "[8/8] Disk space & RAM:"
echo "  Disk space:"
df -h / | tail -1 | awk '{print "       Total: " $2 ", Available: " $4 ", Used: " $5}'
if command -v free &> /dev/null; then
    echo "  RAM:"
    free -h | grep Mem | awk '{print "       Total: " $2 ", Available: " $7}'
fi
echo ""

echo "========================================"
echo "  Ket qua Kiem tra"
echo "========================================"
echo ""
echo "Cac buoc tiep theo:"
echo "1. Neu chua co Python: sudo apt install python3 python3-pip"
echo "2. Neu chua co PyTorch: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo "3. Copy code len may remote va setup"
echo ""

