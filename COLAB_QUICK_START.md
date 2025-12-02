# 🚀 Hướng Dẫn Nhanh - Chạy Training trên Colab (Đã có code trên Git)

## ✅ Checklist - Các Bước Cần Làm

### 🔴 Bắt Buộc (Lần đầu hoặc khi restart Colab)

#### **BƯỚC 1: Mở Colab và Chọn GPU**
1. Truy cập: https://colab.research.google.com/
2. Tạo notebook mới hoặc mở notebook cũ
3. **Runtime** → **Change runtime type** → Chọn **GPU** → **Save**

#### **BƯỚC 2: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### **BƯỚC 3: Clone Repository (Update nếu repo mới)**
```python
# Thay URL bằng repo của bạn
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

**Lưu ý:** Nếu đã clone rồi, chỉ cần:
```python
%cd YOUR_REPO
!git pull  # Update code mới nhất
```

#### **BƯỚC 4: Cài Đặt Dependencies**
```python
print("🔧 Đang fix các dependency conflicts...")

# Bước 1: Uninstall các packages có conflict
!pip uninstall -y numpy protobuf fsspec tensorboard 2>/dev/null || true

# Bước 2: Cài đặt các packages với version cố định
!pip install -q --no-deps "numpy>=1.26.0,<2.1.0"
!pip install -q --no-deps "protobuf>=3.20.3,<6.0.0,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5"
!pip install -q --no-deps "fsspec>=2023.1.0,<=2024.12.0"
!pip install -q --no-deps "tensorboard==2.19.0"
!pip install -q "jedi>=0.16"

# Bước 3: Cài đặt dependencies chính
print("📦 Đang cài đặt dependencies chính...")
!pip install -q pytorch-lightning transformers accelerate peft loguru

# Bước 4: Cài đặt requirements.txt
print("📦 Đang cài đặt requirements.txt...")
!pip install -q --no-deps -r requirements.txt

# Bước 5: Force reinstall các packages quan trọng
print("🔧 Đang lock các packages quan trọng...")
!pip install -q --force-reinstall --no-deps "numpy>=1.26.0,<2.1.0" 2>/dev/null || true
!pip install -q --force-reinstall --no-deps "protobuf>=3.20.3,<6.0.0,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5" 2>/dev/null || true

print("✅ Dependencies đã được cài đặt!")
```

#### **BƯỚC 5: Tìm Dataset**
```python
import os

# Tìm dataset trên Google Drive
drive_root = "/content/drive/MyDrive"
found_datasets = []

for root, dirs, files in os.walk(drive_root):
    if "vi_lora_dataset" in dirs:
        full_path = os.path.join(root, "vi_lora_dataset")
        if os.path.isdir(full_path):
            found_datasets.append(full_path)

if found_datasets:
    # Ưu tiên dataset trong ace_step_data
    dataset_path = None
    for path in found_datasets:
        if "ace_step_data" in path:
            dataset_path = path
            break
    if not dataset_path:
        dataset_path = found_datasets[0]
    print(f"✓ Tìm thấy dataset: {dataset_path}")
else:
    print("❌ Không tìm thấy dataset!")
    print("   Vui lòng upload dataset lên Google Drive")
    raise FileNotFoundError("Dataset not found!")
```

#### **BƯỚC 6: Tạo Thư Mục Output**
```python
import os

checkpoint_dir = "/content/drive/MyDrive/ace_step_outputs/checkpoints"
log_dir = "/content/drive/MyDrive/ace_step_outputs/logs"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print(f"✓ Đã tạo thư mục checkpoint: {checkpoint_dir}")
print(f"✓ Đã tạo thư mục log: {log_dir}")
```

#### **BƯỚC 7: Kiểm tra GPU và RAM**
```python
import torch
import psutil

print("🔍 Kiểm tra GPU...")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ VRAM: {gpu_memory:.2f} GB")
    gpu_ok = True
else:
    print("❌ GPU không available!")
    gpu_ok = False

print("\n🔍 Kiểm tra System RAM...")
ram = psutil.virtual_memory()
print(f"✓ Total RAM: {ram.total / (1024**3):.2f} GB")
print(f"✓ Available RAM: {ram.available / (1024**3):.2f} GB")
print(f"✓ Used RAM: {ram.used / (1024**3):.2f} GB")
```

---

### 🟡 Tùy Chọn (Nếu Resume Training)

#### **BƯỚC 8: Tìm Checkpoint (Nếu Resume)**
```python
import glob
import os

log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
lora_checkpoints = glob.glob(f"{log_dir}/*/checkpoints/*_lora/pytorch_lora_weights.safetensors") if os.path.exists(log_dir) else []

if lora_checkpoints:
    latest_checkpoint = max(lora_checkpoints, key=os.path.getctime)
    checkpoint_dir = os.path.dirname(latest_checkpoint)
    print(f"✓ Tìm thấy LoRA checkpoint mới nhất: {checkpoint_dir}")
    lora_checkpoint_dir = checkpoint_dir
else:
    print("ℹ Chưa có checkpoint, sẽ train từ đầu")
    lora_checkpoint_dir = None
```

**Lưu ý:** Code sẽ tự động tìm và load checkpoint, không cần truyền `--lora_checkpoint_dir` nếu không muốn chỉ định thủ công.

---

### 🟢 Bắt Đầu Training

#### **BƯỚC 9: Chạy Training**
```python
import os

# Kiểm tra các biến cần thiết
if 'dataset_path' not in locals() or dataset_path is None:
    print("❌ Chưa tìm thấy dataset! Vui lòng chạy BƯỚC 5 trước.")
    raise RuntimeError("Dataset not found!")

if 'gpu_ok' not in locals() or not gpu_ok:
    print("❌ GPU chưa được kiểm tra! Vui lòng chạy BƯỚC 7 trước.")
    raise RuntimeError("GPU not checked!")

# Tham số training
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
    --every_n_train_steps 50 \\
    --shift 3.0 \\
    --checkpoint_dir "{checkpoint_dir}" \\
    --logger_dir "{log_dir}" \\
    --epochs -1 \\
    --every_plot_step 2000 \\
    --gradient_clip_val 0.5 \\
    --gradient_clip_algorithm "norm" """

print("🚀 Bắt đầu training...")
print("=" * 60)
print(cmd)
print("=" * 60)

# Chạy training
!{cmd}
```

---

## 📝 Lưu Ý Quan Trọng

### ⚠️ Khi Colab Tự Động Disconnect

**Triệu chứng:**
- Colab tự động disconnect ở bước "Converting transformer to float32..."
- RAM tăng đột ngột

**Giải pháp:**
1. **Chờ code hoàn tất** (có thể mất 2-3 phút ở bước convert)
   - Xem log "Đã convert X/Y modules..." để biết code đang chạy
2. **Nâng cấp Colab Pro+** (50GB RAM) - khuyến nghị
3. **Chạy lại** - code sẽ tự động fallback nếu OOM

### ✅ Sau Khi Training

- Checkpoint được lưu mỗi **50 steps** vào `checkpoints/epoch=X-step=Y_lora/`
- Code tự động resume từ checkpoint mới nhất khi restart
- Checkpoint format: Chỉ lưu LoRA weights (`.safetensors`) → ~10-50MB mỗi checkpoint

---

## 🔄 Workflow Nhanh

**Lần đầu:**
1. Bước 1-7 (bắt buộc)
2. Bước 9 (training)

**Lần sau (đã có dataset và checkpoint):**
1. Bước 1-4 (clone repo, cài dependencies)
2. Bước 5-7 (tìm dataset, tạo thư mục, kiểm tra GPU)
3. Bước 8 (tìm checkpoint nếu resume)
4. Bước 9 (training)

**Nếu Colab disconnect:**
1. Chạy lại Bước 1-4
2. Chạy lại Bước 9 (code tự động resume từ checkpoint mới nhất)

---

## 📚 Xem Hướng Dẫn Chi Tiết

Nếu cần hướng dẫn chi tiết hơn, xem file: `COLAB_TRAIN_STEP_BY_STEP.md`

