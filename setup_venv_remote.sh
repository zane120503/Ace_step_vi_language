#!/bin/bash
# Script setup virtual environment cho training tren may remote
# Su dung: chmod +x setup_venv_remote.sh && bash setup_venv_remote.sh

set -e  # Exit on error

echo "========================================"
echo "  Setup Virtual Environment"
echo "========================================"
echo ""

# 1. Kiem tra Python
echo "[1/5] Kiem tra Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 khong tim thay!"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "  [OK] $PYTHON_VERSION"
echo ""

# 2. Kiem tra python3-venv
echo "[2/5] Kiem tra python3-venv..."
if ! python3 -m venv --help &> /dev/null; then
    echo "[WARNING] python3-venv khong tim thay, dang cai dat..."
    apt-get update -qq
    apt-get install -y python3-full python3-venv
fi
echo "  [OK] python3-venv san sang"
echo ""

# 3. Tao virtual environment
VENV_DIR="/root/ace_step_env"
echo "[3/5] Tao virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  [INFO] Virtual environment da ton tai tai: $VENV_DIR"
    echo "        Ban co muon xoa va tao lai? (Y/N)"
    read -r confirm
    if [ "$confirm" = "Y" ] || [ "$confirm" = "y" ]; then
        echo "  Dang xoa virtual environment cu..."
        rm -rf "$VENV_DIR"
    else
        echo "  [INFO] Su dung virtual environment hien tai"
        source "$VENV_DIR/bin/activate"
        python3 --version
        echo ""
        exit 0
    fi
fi

echo "  Dang tao virtual environment tai: $VENV_DIR"
python3 -m venv "$VENV_DIR"
echo "  [OK] Virtual environment da tao thanh cong"
echo ""

# 4. Activate virtual environment
echo "[4/5] Kich hoat virtual environment..."
source "$VENV_DIR/bin/activate"
python3 --version
which python3
which pip3
echo ""

# 5. Upgrade pip
echo "[5/5] Upgrade pip..."
pip3 install --upgrade pip
echo ""

echo "========================================"
echo "  Setup Virtual Environment Hoan tat!"
echo "========================================"
echo ""
echo "Virtual environment da duoc tao tai: $VENV_DIR"
echo ""
echo "De su dung virtual environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Cac buoc tiep theo:"
echo "1. Kich hoat virtual environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "2. Cai dat PyTorch:"
echo "   pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121"
echo ""
echo "3. Clone code tu GitHub:"
echo "   cd /root"
echo "   git clone https://github.com/zane120503/Ace_step_vi_language.git ACE-Step"
echo ""
echo "4. Cai dat dependencies:"
echo "   cd /root/ACE-Step"
echo "   pip3 install -r requirements.txt"
echo ""

