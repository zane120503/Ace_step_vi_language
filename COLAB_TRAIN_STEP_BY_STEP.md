# Hướng dẫn Train ACE-Step LoRA trên Google Colab - Từng Bước Chi Tiết

## 📋 Chuẩn bị trước khi bắt đầu

### 1. Cần có:
- ✅ Google Account
- ✅ Google Drive (để lưu dataset và checkpoint)
- ✅ Dataset đã được convert sang HuggingFace format (`vi_lora_dataset`)
- ✅ File config: `config/vi_lora_config.json`

### 2. Upload lên Google Drive:
- Tạo folder `MyDrive/ace_step_data/`
- Upload folder `vi_lora_dataset` vào đó
- Upload file `config/vi_lora_config.json` vào đó

---

## 🚀 BƯỚC 1: Mở Google Colab

1. Truy cập: https://colab.research.google.com/
2. Đăng nhập bằng Google Account
3. Click **"New notebook"** để tạo notebook mới
4. Đặt tên notebook: `ACE-Step LoRA Training`

---

## 🚀 BƯỚC 2: Chọn GPU

1. Click **Runtime** → **Change runtime type**
2. Chọn:
   - **Hardware accelerator**: `GPU`
   - **GPU type**: 
     - **Free**: T4 (tự động)
     - **Pro**: T4/V100 (tùy may mắn)
     - **Pro+**: A100 (tùy may mắn)
3. Click **Save**

---

## 🚀 BƯỚC 3: Mount Google Drive

**Tạo cell mới và chạy:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Kết quả:**
- Sẽ hiện link để authorize
- Click link → chọn Google Account → Copy mã
- Paste mã vào ô input → Enter
- Sẽ thấy: `Mounted at /content/drive`

---

## 🚀 BƯỚC 4: Clone Repository

**Tạo cell mới và chạy:**

```python
!git clone https://github.com/ace-step/ACE-Step.git
%cd ACE-Step
```

**Kết quả:**
- Repository được clone vào `/content/ACE-Step`
- Đã chuyển vào thư mục ACE-Step

---

## 🚀 BƯỚC 5: Cài đặt Dependencies

**Tạo cell mới và chạy:**

```python
!pip install -q pytorch-lightning transformers accelerate
!pip install -q -r requirements.txt
```

**Lưu ý:**
- Có thể mất 5-10 phút
- Nếu có lỗi, thử chạy lại cell

---

## 🚀 BƯỚC 6: Kiểm tra Dataset

**Tạo cell mới và chạy:**

```python
import os

# Kiểm tra dataset có tồn tại không
dataset_path = "/content/drive/MyDrive/ace_step_data/vi_lora_dataset"
if os.path.exists(dataset_path):
    print(f"✓ Dataset tìm thấy tại: {dataset_path}")
    # Đếm số file
    files = os.listdir(dataset_path)
    print(f"✓ Số file trong dataset: {len(files)}")
else:
    print(f"❌ Không tìm thấy dataset tại: {dataset_path}")
    print("   Vui lòng upload dataset lên Google Drive trước!")
```

**Kết quả mong đợi:**
- ✓ Dataset tìm thấy tại: ...
- ✓ Số file trong dataset: ...

---

## 🚀 BƯỚC 7: Kiểm tra Config

**Tạo cell mới và chạy:**

```python
import os

# Kiểm tra config file
config_path = "/content/drive/MyDrive/ace_step_data/config/vi_lora_config.json"
if os.path.exists(config_path):
    print(f"✓ Config file tìm thấy tại: {config_path}")
    # Copy vào thư mục config của repo
    !cp "{config_path}" config/vi_lora_config.json
    print("✓ Đã copy config vào repo")
else:
    print(f"❌ Không tìm thấy config tại: {config_path}")
    print("   Vui lòng upload config file lên Google Drive trước!")
```

**Kết quả mong đợi:**
- ✓ Config file tìm thấy tại: ...
- ✓ Đã copy config vào repo

---

## 🚀 BƯỚC 8: Tạo Thư Mục Output

**Tạo cell mới và chạy:**

```python
import os

# Tạo thư mục trên Google Drive để lưu checkpoint và log
checkpoint_dir = "/content/drive/MyDrive/ace_step_outputs/checkpoints"
log_dir = "/content/drive/MyDrive/ace_step_outputs/logs"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print(f"✓ Đã tạo thư mục checkpoint: {checkpoint_dir}")
print(f"✓ Đã tạo thư mục log: {log_dir}")
```

**Kết quả mong đợi:**
- ✓ Đã tạo thư mục checkpoint: ...
- ✓ Đã tạo thư mục log: ...

---

## 🚀 BƯỚC 9: Tìm Checkpoint (Nếu Resume)

**Tạo cell mới và chạy:**

```python
import glob
import os

# Tìm checkpoint mới nhất (nếu có)
log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
checkpoints = glob.glob(f"{log_dir}/*/checkpoints/*.ckpt") if os.path.exists(log_dir) else []

if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"✓ Tìm thấy checkpoint mới nhất: {latest_checkpoint}")
    print(f"  Sẽ resume từ checkpoint này")
    ckpt_path = latest_checkpoint
else:
    print("ℹ Chưa có checkpoint, sẽ train từ đầu")
    ckpt_path = None
```

**Kết quả:**
- Nếu có checkpoint: ✓ Tìm thấy checkpoint mới nhất: ...
- Nếu chưa có: ℹ Chưa có checkpoint, sẽ train từ đầu

---

## 🚀 BƯỚC 10: Bắt Đầu Training

**Tạo cell mới và chạy:**

```python
import os

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
if 'ckpt_path' in locals() and ckpt_path:
    cmd += f'\\\n    --ckpt_path "{ckpt_path}"'

print("🚀 Bắt đầu training...")
print("=" * 60)
print(cmd)
print("=" * 60)

# Chạy training
!{cmd}
```

**Lưu ý:**
- Training sẽ chạy và hiển thị log
- Có thể mất vài phút để khởi động
- Checkpoint sẽ được lưu mỗi 100 steps

---

## 📊 BƯỚC 11: Monitor Training

**Tạo cell mới và chạy (để xem log):**

```python
# Xem log real-time (chạy cell này trong tab mới để không block)
import time

log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
log_files = glob.glob(f"{log_dir}/*/events.out.tfevents.*")

if log_files:
    latest_log = max(log_files, key=os.path.getctime)
    print(f"📊 Đang theo dõi log: {latest_log}")
    print("=" * 60)
    !tail -f {latest_log}
else:
    print("ℹ Chưa có log file")
```

**Hoặc xem log đơn giản hơn:**

```python
# Xem 50 dòng log cuối cùng
log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
log_files = glob.glob(f"{log_dir}/*/events.out.tfevents.*")

if log_files:
    latest_log = max(log_files, key=os.path.getctime)
    !tail -n 50 {latest_log}
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. Runtime Timeout
- **Colab Free**: ~12 giờ timeout
- **Colab Pro**: ~24 giờ timeout
- **Giải pháp**: 
  - Lưu checkpoint mỗi 100 steps (đã set)
  - Resume từ checkpoint mới nhất khi restart

### 2. Nếu Bị Disconnect
1. Tìm checkpoint mới nhất (Bước 9)
2. Resume training (Bước 10) với checkpoint đó

### 3. Tối Ưu cho GPU
- **T4 (16GB)**: Dùng `--accumulate_grad_batches 4` (đã set)
- **V100 (16GB)**: Có thể tăng lên `8`
- **A100 (40GB)**: Có thể tăng lên `16`

### 4. Nếu Bị OOM (Out of Memory)
- Giảm `--accumulate_grad_batches` xuống `2` hoặc `1`
- Giảm `--num_workers` xuống `0`

---

## 🔄 Resume Training (Sau khi Disconnect)

**Nếu bị disconnect, làm lại từ Bước 9:**

1. Chạy lại cell Bước 9 (Tìm checkpoint)
2. Chạy lại cell Bước 10 (Training) - sẽ tự động resume

---

## 📥 Download Checkpoint về Local

**Sau khi training xong, download checkpoint:**

```python
# Tìm checkpoint mới nhất
import glob
import os

log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
checkpoints = glob.glob(f"{log_dir}/*/checkpoints/*.ckpt")

if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"✓ Checkpoint mới nhất: {latest_checkpoint}")
    print(f"  Đã lưu trên Google Drive, có thể download về local")
else:
    print("ℹ Chưa có checkpoint")
```

**Cách download:**
1. Mở Google Drive
2. Vào folder `ace_step_outputs/logs/vi_lora/lightning_logs/.../checkpoints/`
3. Download file `.ckpt` về máy

---

## 🎯 Checklist Hoàn Thành

- [ ] Đã mount Google Drive
- [ ] Đã clone repository
- [ ] Đã cài đặt dependencies
- [ ] Đã kiểm tra dataset
- [ ] Đã kiểm tra config
- [ ] Đã tạo thư mục output
- [ ] Đã bắt đầu training
- [ ] Training đang chạy (không có lỗi)

---

## 🆘 Troubleshooting

### Lỗi: "Dataset not found"
- Kiểm tra đường dẫn dataset trên Google Drive
- Đảm bảo đã upload đúng folder `vi_lora_dataset`

### Lỗi: "Config not found"
- Kiểm tra file `vi_lora_config.json` trên Google Drive
- Đảm bảo đã copy vào repo (Bước 7)

### Lỗi: "Out of Memory"
- Giảm `--accumulate_grad_batches` xuống `2` hoặc `1`
- Giảm `--num_workers` xuống `0`

### Lỗi: "Runtime disconnected"
- Resume từ checkpoint mới nhất (Bước 9 + 10)

---

## 📝 Ghi Chú

- Checkpoint được lưu mỗi **100 steps** (đã set `--every_n_train_steps 100`)
- Log được lưu trên Google Drive
- Có thể resume bất cứ lúc nào từ checkpoint mới nhất
- Training sẽ tự động dừng khi đạt `max_steps` (20000)

---

**Chúc bạn training thành công! 🎉**

