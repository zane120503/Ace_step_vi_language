"""
Script để train ACE-Step LoRA trên Google Colab
Copy và chạy từng phần trong Colab notebook
"""

# ============================================
# PHẦN 1: Setup môi trường (chạy 1 lần)
# ============================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository (nếu chưa có)
import os
if not os.path.exists('/content/ACE-Step'):
    !git clone https://github.com/ace-step/ACE-Step.git
    %cd /content/ACE-Step
else:
    %cd /content/ACE-Step
    !git pull

# Cài đặt dependencies
!pip install -q pytorch-lightning transformers accelerate
!pip install -q -r requirements.txt

# ============================================
# PHẦN 2: Tìm checkpoint mới nhất (nếu resume)
# ============================================

import glob

def find_latest_checkpoint(log_dir="/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"):
    """Tìm checkpoint mới nhất để resume"""
    checkpoints = glob.glob(f"{log_dir}/*/checkpoints/*.ckpt")
    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        print(f"✓ Tìm thấy checkpoint: {latest}")
        return latest
    else:
        print("ℹ Chưa có checkpoint, sẽ train từ đầu")
        return None

# ============================================
# PHẦN 3: Train LoRA
# ============================================

def train_lora(
    dataset_path="/content/drive/MyDrive/vi_lora_dataset",
    checkpoint_dir="/content/drive/MyDrive/ace_step_outputs/checkpoints",
    log_dir="/content/drive/MyDrive/ace_step_outputs/logs",
    resume_from_checkpoint=True,
    max_steps=20000,
    every_n_train_steps=500,
    accumulate_grad_batches=4,
    precision=16,
    num_workers=2
):
    """
    Train LoRA trên Colab
    
    Args:
        dataset_path: Đường dẫn đến dataset (trên Google Drive)
        checkpoint_dir: Thư mục lưu checkpoint (trên Google Drive)
        log_dir: Thư mục lưu log (trên Google Drive)
        resume_from_checkpoint: Có tự động resume từ checkpoint mới nhất không
        max_steps: Số step tối đa
        every_n_train_steps: Lưu checkpoint mỗi N steps
        accumulate_grad_batches: Gradient accumulation
        precision: 16 (FP16) hoặc 32 (FP32)
        num_workers: Số worker cho DataLoader (Colab nên dùng 2)
    """
    
    # Tạo thư mục output
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Tìm checkpoint để resume
    ckpt_path = None
    if resume_from_checkpoint:
        ckpt_path = find_latest_checkpoint(log_dir)
    
    # Build command
    cmd = f"""python trainer.py \\
    --num_nodes 1 \\
    --devices 1 \\
    --dataset_path "{dataset_path}" \\
    --exp_name "vi_lora_small" \\
    --lora_config_path "config/vi_lora_config.json" \\
    --learning_rate 1e-4 \\
    --accumulate_grad_batches {accumulate_grad_batches} \\
    --precision {precision} \\
    --num_workers {num_workers} \\
    --max_steps {max_steps} \\
    --every_n_train_steps {every_n_train_steps} \\
    --shift 3.0 \\
    --checkpoint_dir "{checkpoint_dir}" \\
    --logger_dir "{log_dir}" \\
    --epochs -1 \\
    --every_plot_step 2000 \\
    --gradient_clip_val 0.5 \\
    --gradient_clip_algorithm "norm" """
    
    if ckpt_path:
        cmd += f'\\\n    --ckpt_path "{ckpt_path}"'
    
    print("🚀 Bắt đầu training...")
    print(f"Command: {cmd}")
    
    # Chạy training
    !{cmd}

# ============================================
# PHẦN 4: Sử dụng
# ============================================

# Cách 1: Train từ đầu
# train_lora(
#     dataset_path="/content/drive/MyDrive/vi_lora_dataset",
#     resume_from_checkpoint=False
# )

# Cách 2: Resume từ checkpoint mới nhất
# train_lora(
#     dataset_path="/content/drive/MyDrive/vi_lora_dataset",
#     resume_from_checkpoint=True
# )

# Cách 3: Train với tham số tùy chỉnh
# train_lora(
#     dataset_path="/content/drive/MyDrive/vi_lora_dataset",
#     max_steps=50000,
#     every_n_train_steps=200,  # Lưu checkpoint thường xuyên hơn
#     accumulate_grad_batches=8,  # Tăng nếu GPU đủ mạnh
#     num_workers=0  # Giảm nếu bị lỗi
# )

