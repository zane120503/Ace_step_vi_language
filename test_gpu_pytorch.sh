#!/bin/bash
# Script kiem tra GPU voi PyTorch
# Su dung: chmod +x test_gpu_pytorch.sh && bash test_gpu_pytorch.sh

echo "========================================"
echo "  Kiem Tra GPU voi PyTorch"
echo "========================================"
echo ""

# Kich hoat virtual environment neu chua
if [ -d "/root/ace_step_env" ]; then
    source /root/ace_step_env/bin/activate
fi

echo "[1/4] Kiem tra PyTorch..."
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
echo ""

echo "[2/4] Kiem tra CUDA..."
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "[3/4] Kiem tra GPU..."
python3 -c "import torch; print(f'  GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('  No GPU found')"
echo ""

echo "[4/4] Test GPU computation..."
python3 << EOF
import torch
import warnings

# Suppress warning for testing
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    if torch.cuda.is_available():
        print("  Dang test GPU computation...")
        try:
            # Tao tensor tren GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Thuc hien computation
            z = torch.matmul(x, y)
            
            print(f"  [OK] GPU computation thanh cong!")
            print(f"       Result shape: {z.shape}")
            print(f"       Result device: {z.device}")
            print(f"       Result dtype: {z.dtype}")
            
            # Test memory
            print(f"\n  GPU Memory:")
            print(f"       Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"       Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"  [ERROR] GPU computation that bai: {e}")
    else:
        print("  [WARNING] CUDA khong available, khong the test GPU")
EOF

echo ""
echo "========================================"
echo "  Kiem Tra Hoan Tat!"
echo "========================================"
echo ""
echo "Neu GPU computation thanh cong, ban co the tiep tuc training!"
echo "Neu co loi, co the can PyTorch build moi hon hoac cai CUDA toolkit moi hon."
echo ""

