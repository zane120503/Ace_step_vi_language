"""
🎯 HƯỚNG DẪN SỬ DỤNG:
1. Mở Google Colab: https://colab.research.google.com/
2. Tạo notebook mới
3. Chọn Runtime → Change runtime type → GPU
4. Copy từng cell dưới đây vào Colab và chạy
"""

# ============================================
# CELL 1: Mount Google Drive
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# CELL 2: Clone Repository
# ============================================
!git clone https://github.com/ace-step/ACE-Step.git
%cd ACE-Step

# ============================================
# CELL 3: Cài đặt Dependencies
# ============================================
!pip install -q pytorch-lightning transformers accelerate
!pip install -q -r requirements.txt

# ============================================
# CELL 4: Kiểm tra Dataset và Config
# ============================================
import os
import glob

# Kiểm tra dataset
dataset_path = "/content/drive/MyDrive/ace_step_data/vi_lora_dataset"
if os.path.exists(dataset_path):
    files = os.listdir(dataset_path)
    print(f"✓ Dataset tìm thấy: {len(files)} files")
else:
    print(f"❌ Không tìm thấy dataset tại: {dataset_path}")
    print("   Vui lòng upload dataset lên Google Drive!")

# Kiểm tra và copy config
config_path = "/content/drive/MyDrive/ace_step_data/config/vi_lora_config.json"
if os.path.exists(config_path):
    !cp "{config_path}" config/vi_lora_config.json
    print("✓ Config file đã được copy")
else:
    print(f"❌ Không tìm thấy config tại: {config_path}")
    print("   Vui lòng upload config file lên Google Drive!")

# ============================================
# CELL 5: Tạo Thư Mục Output
# ============================================
checkpoint_dir = "/content/drive/MyDrive/ace_step_outputs/checkpoints"
log_dir = "/content/drive/MyDrive/ace_step_outputs/logs"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print(f"✓ Checkpoint dir: {checkpoint_dir}")
print(f"✓ Log dir: {log_dir}")

# ============================================
# CELL 6: Kiểm tra GPU
# ============================================
import torch

print("🔍 Kiểm tra GPU...")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ GPU tìm thấy: {gpu_name}")
    print(f"✓ VRAM: {gpu_memory:.2f} GB")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    gpu_ok = True
else:
    print("❌ GPU không available!")
    print("   Vui lòng chọn Runtime → Change runtime type → GPU")
    print("   Sau đó restart runtime và chạy lại cell này")
    gpu_ok = False

# ============================================
# CELL 7: Tìm Checkpoint (Nếu Resume)
# ============================================
log_checkpoint_dir = f"{log_dir}/vi_lora/lightning_logs"
checkpoints = glob.glob(f"{log_checkpoint_dir}/*/checkpoints/*.ckpt") if os.path.exists(log_checkpoint_dir) else []

if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"✓ Tìm thấy checkpoint: {latest_checkpoint}")
    ckpt_path = latest_checkpoint
    resume = True
else:
    print("ℹ Chưa có checkpoint, sẽ train từ đầu")
    ckpt_path = None
    resume = False

# ============================================
# CELL 8: Bắt Đầu Training
# ============================================
# Tham số training
dataset_path = "/content/drive/MyDrive/ace_step_data/vi_lora_dataset"
checkpoint_dir = "/content/drive/MyDrive/ace_step_outputs/checkpoints"
log_dir = "/content/drive/MyDrive/ace_step_outputs/logs"

# Build command
cmd = f"""python trainer.py \\
    --num_nodes 1 \\
    --devices 1 \\
    --dataset_path "{dataset_path}" \\
    --exp_name "vi_lora_small" \\
    --lora_config_path "config/vi_lora_config.json" \\
    --learning_rate 1e-4 \\
    --accumulate_grad_batches 4 \\
    --precision 16 \\
    --num_workers 2 \\
    --max_steps 20000 \\
    --every_n_train_steps 100 \\
    --shift 3.0 \\
    --checkpoint_dir "{checkpoint_dir}" \\
    --logger_dir "{log_dir}" \\
    --epochs -1 \\
    --every_plot_step 2000 \\
    --gradient_clip_val 0.5 \\
    --gradient_clip_algorithm "norm" """

# Thêm --ckpt_path nếu có checkpoint
if resume and ckpt_path:
    cmd += f'\\\n    --ckpt_path "{ckpt_path}"'

# Kiểm tra GPU trước khi train
if 'gpu_ok' not in locals() or not gpu_ok:
    print("❌ Không thể train vì GPU không available!")
    print("   Vui lòng chạy CELL 6 (Kiểm tra GPU) trước!")
    print("   Hoặc chọn Runtime → Change runtime type → GPU")
else:
    print("🚀 Bắt đầu training...")
    print("=" * 60)
    print(cmd)
    print("=" * 60)
    
    # Chạy training
    !{cmd}

# ============================================
# CELL 9: Monitor Training (Optional)
# ============================================
# Chạy cell này trong tab mới để xem log real-time
# (Không chạy cùng lúc với training)

# log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
# log_files = glob.glob(f"{log_dir}/*/events.out.tfevents.*")
# if log_files:
#     latest_log = max(log_files, key=os.path.getctime)
#     !tail -f {latest_log}

