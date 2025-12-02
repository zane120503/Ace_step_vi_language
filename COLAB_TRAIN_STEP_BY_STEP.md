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
# Clone từ repository tiếng Việt (đã có config và dataset cho tiếng Việt)
!git clone https://github.com/zane120503/Ace_step_vi_language.git
%cd Ace_step_vi_language
```

**Kết quả:**
- Repository được clone vào `/content/ACE-Step`
- Đã chuyển vào thư mục ACE-Step

---

## 🚀 BƯỚC 5: Cài đặt Dependencies

**Tạo cell mới và chạy:**

```python
print("🔧 Đang fix các dependency conflicts...")

# Bước 1: Uninstall các packages có conflict để clean install
!pip uninstall -y numpy protobuf fsspec tensorboard 2>/dev/null || true

# Bước 2: Cài đặt các packages với version cố định (--no-deps để tránh conflicts)
!pip install -q --no-deps "numpy>=1.26.0,<2.1.0"
!pip install -q --no-deps "protobuf>=3.20.3,<6.0.0,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5"
!pip install -q --no-deps "fsspec>=2023.1.0,<=2024.12.0"
!pip install -q --no-deps "tensorboard==2.19.0"
!pip install -q "jedi>=0.16"

# Bước 3: Cài đặt dependencies chính
print("📦 Đang cài đặt dependencies chính...")
!pip install -q pytorch-lightning transformers accelerate

# Bước 4: Cài đặt requirements.txt với --no-deps để tránh conflicts
print("📦 Đang cài đặt requirements.txt (bỏ qua dependency checks)...")
!pip install -q --no-deps -r requirements.txt

# Bước 5: Force reinstall các packages quan trọng với version đúng
print("🔧 Đang lock các packages quan trọng ở version đúng...")
!pip install -q --force-reinstall --no-deps "numpy>=1.26.0,<2.1.0" 2>/dev/null || true
!pip install -q --force-reinstall --no-deps "protobuf>=3.20.3,<6.0.0,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5" 2>/dev/null || true
!pip install -q --force-reinstall --no-deps "fsspec>=2023.1.0,<=2024.12.0" 2>/dev/null || true
!pip install -q --force-reinstall --no-deps "tensorboard==2.19.0" 2>/dev/null || true

print("✓ Đã cài đặt dependencies")
print("ℹ Đã fix và lock các conflicts: numpy, protobuf, tensorboard, fsspec, jedi")
print("⚠️  Một số warnings về gcsfs/fsspec có thể xuất hiện nhưng KHÔNG ảnh hưởng training")
print("⚠️  Các warnings về dependency conflicts có thể bỏ qua nếu training vẫn chạy được")
```

**Lưu ý:**
- Có thể mất 5-10 phút
- Script đã tự động xử lý các dependency conflicts phổ biến
- Có thể vẫn có một số warnings, nhưng **không ảnh hưởng đến training**
- Nếu có lỗi nghiêm trọng khác, thử chạy lại cell

---

## 🚀 BƯỚC 6: Kiểm tra Dataset và Config

**Tạo cell mới và chạy:**

```python
import os
import glob

# Bước 1: Kiểm tra Google Drive đã được mount chưa
if not os.path.exists("/content/drive"):
    print("❌ Google Drive chưa được mount!")
    print("   Vui lòng chạy BƯỚC 3 (Mount Google Drive) trước!")
else:
    print("✓ Google Drive đã được mount")
    
    # Bước 2: Tìm tất cả các folder có tên "vi_lora_dataset" trên Drive
    print("🔍 Đang tìm dataset trên Google Drive...")
    drive_root = "/content/drive/MyDrive"
    
    found_datasets = []
    for root, dirs, files in os.walk(drive_root):
        if "vi_lora_dataset" in dirs:
            full_path = os.path.join(root, "vi_lora_dataset")
            if os.path.isdir(full_path):
                file_count = len(os.listdir(full_path))
                found_datasets.append((full_path, file_count))
                print(f"✓ Tìm thấy: {full_path} ({file_count} files)")
    
    # Bước 3: Chọn dataset phù hợp
    dataset_path = None
    if found_datasets:
        # Ưu tiên dataset trong ace_step_data
        for path, count in found_datasets:
            if "ace_step_data" in path:
                dataset_path = path
                print(f"\n✓ Dataset tìm thấy tại: {path}")
                print(f"✓ Số file trong dataset: {count}")
                break
        
        # Nếu không có trong ace_step_data, dùng dataset đầu tiên
        if not dataset_path:
            dataset_path, count = found_datasets[0]
            print(f"\n✓ Dataset tìm thấy tại: {dataset_path}")
            print(f"✓ Số file trong dataset: {count}")
            if len(found_datasets) > 1:
                print(f"⚠️  Tìm thấy {len(found_datasets)} dataset, đang dùng: {dataset_path}")
    else:
        print("\n❌ Không tìm thấy dataset!")
        print("   Vui lòng upload dataset lên Google Drive")
        print("   Có thể đặt ở bất kỳ đâu trong MyDrive")

# Bước 4: Kiểm tra và copy config
config_paths = [
    "/content/drive/MyDrive/MyDrive/ace_step_data/config/vi_lora_config.json",  # Trường hợp có MyDrive trong MyDrive
    "/content/drive/MyDrive/ace_step_data/config/vi_lora_config.json",
    "/content/drive/MyDrive/config/vi_lora_config.json",
    "/content/drive/MyDrive/vi_lora_config.json",
]

config_found = False
for config_path in config_paths:
    if os.path.exists(config_path):
        !cp "{config_path}" config/vi_lora_config.json
        print(f"✓ Config file đã được copy từ: {config_path}")
        config_found = True
        break

if not config_found:
    print("⚠️  Không tìm thấy config file!")
    print("   Có thể sử dụng config mặc định trong repo")
    if os.path.exists("config/vi_lora_config.json"):
        print("✓ Đã tìm thấy config trong repo")
    else:
        print("❌ Cần tạo hoặc upload config file")
```

**Kết quả mong đợi:**
- ✓ Google Drive đã được mount
- ✓ Dataset tìm thấy tại: `/content/drive/MyDrive/.../vi_lora_dataset`
- ✓ Số file trong dataset: ...
- ✓ Config file đã được copy (nếu có)

---

## 🚀 BƯỚC 7: Tạo Thư Mục Output

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

## 🚀 BƯỚC 8: Kiểm tra GPU và RAM

**Tạo cell mới và chạy:**

```python
import torch
import psutil
import os

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

print("\n🔍 Kiểm tra System RAM...")
ram_total = psutil.virtual_memory().total / (1024**3)
ram_available = psutil.virtual_memory().available / (1024**3)
ram_used = psutil.virtual_memory().used / (1024**3)
print(f"✓ Total RAM: {ram_total:.2f} GB")
print(f"✓ Available RAM: {ram_available:.2f} GB")
print(f"✓ Used RAM: {ram_used:.2f} GB")

if ram_total < 20:
    print("⚠️  RAM hơi thấp (<20GB), code đã được tối ưu để load từng phần")
    print("   Khởi tạo có thể mất 5-10 phút nhưng vẫn chạy được")
else:
    print("✓ RAM đủ để training")
```

**Kết quả mong đợi:**
- ✓ GPU tìm thấy: Tesla T4 (hoặc V100/A100)
- ✓ VRAM: 16.00 GB (hoặc tương ứng)
- ✓ CUDA available: True
- ✓ Total RAM: 12.7 GB (Colab free) hoặc 25-50GB (Colab Pro/Pro+)

---

## 🚀 BƯỚC 9: Tìm Checkpoint (Nếu Resume)

**Tạo cell mới và chạy:**

```python
import glob
import os

# Tìm LoRA adapter checkpoint mới nhất (nếu có)
# Checkpoint format: epoch=X-step=Y_lora/pytorch_lora_weights.safetensors
log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
lora_checkpoints = glob.glob(f"{log_dir}/*/checkpoints/*_lora/pytorch_lora_weights.safetensors") if os.path.exists(log_dir) else []

if lora_checkpoints:
    latest_checkpoint = max(lora_checkpoints, key=os.path.getctime)
    checkpoint_dir = os.path.dirname(latest_checkpoint)  # Lấy thư mục chứa safetensors
    print(f"✓ Tìm thấy LoRA checkpoint mới nhất: {checkpoint_dir}")
    print(f"  File: {os.path.basename(latest_checkpoint)}")
    print(f"  Sẽ load LoRA weights từ checkpoint này khi training")
    # Lưu checkpoint_dir để dùng sau (nếu cần load manual)
    lora_checkpoint_dir = checkpoint_dir
else:
    print("ℹ Chưa có checkpoint, sẽ train từ đầu")
    lora_checkpoint_dir = None

# Lưu ý: Code sẽ tự động resume từ checkpoint nếu có
# Không cần truyền --ckpt_path vì chỉ train LoRA adapter
```

**Kết quả:**
- Nếu có checkpoint: ✓ Tìm thấy LoRA checkpoint mới nhất: ...
- Nếu chưa có: ℹ Chưa có checkpoint, sẽ train từ đầu

**Lưu ý:**
- Checkpoint format mới: Chỉ lưu LoRA adapter weights (`.safetensors`), không lưu full Lightning checkpoint
- Để resume training, code sẽ tự động load LoRA weights nếu tìm thấy trong thư mục checkpoints

---

## 🚀 BƯỚC 10: Bắt Đầu Training

**Tạo cell mới và chạy:**

```python
import os

# Kiểm tra dataset_path đã được tìm thấy ở Bước 6 chưa
if 'dataset_path' not in locals() or dataset_path is None:
    print("🔍 Đang tìm lại dataset...")
    # Tìm lại dataset
    drive_root = "/content/drive/MyDrive"
    found_datasets = []
    
    for root, dirs, files in os.walk(drive_root):
        if "vi_lora_dataset" in dirs:
            full_path = os.path.join(root, "vi_lora_dataset")
            if os.path.isdir(full_path):
                found_datasets.append(full_path)
    
    if found_datasets:
        # Ưu tiên dataset trong ace_step_data
        for path in found_datasets:
            if "ace_step_data" in path:
                dataset_path = path
                break
        if not dataset_path:
            dataset_path = found_datasets[0]
        print(f"✓ Tìm thấy dataset: {dataset_path}")
    else:
        print("❌ Vẫn không tìm thấy dataset!")
        raise FileNotFoundError("Dataset not found!")

# Xác nhận dataset path
print(f"📂 Sử dụng dataset: {dataset_path}")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path không tồn tại: {dataset_path}")

# Tham số training
checkpoint_dir = "/content/drive/MyDrive/ace_step_outputs/checkpoints"
log_dir = "/content/drive/MyDrive/ace_step_outputs/logs"

# Kiểm tra GPU trước khi train
if 'gpu_ok' not in locals() or not gpu_ok:
    print("⚠️  GPU chưa được kiểm tra!")
    print("   Vui lòng chạy BƯỚC 8 (Kiểm tra GPU) trước!")
    raise RuntimeError("GPU not checked!")

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

# Lưu ý: Code sẽ tự động tìm và load LoRA checkpoint nếu có
# Không cần --ckpt_path vì chỉ train LoRA adapter (không phải full model checkpoint)

print("🚀 Bắt đầu training...")
print("=" * 60)
print(cmd)
print("=" * 60)

# Chạy training
!{cmd}
```

**Lưu ý:**
- Training sẽ chạy và hiển thị log
- **Khởi tạo có thể mất 5-10 phút** do code đã được tối ưu để load từng phần (giảm RAM usage)
- Code sẽ tự động clear RAM sau mỗi bước load model → phù hợp với Colab free (12.7GB RAM)
- Checkpoint (LoRA adapter) sẽ được lưu mỗi **50 steps** vào thư mục `checkpoints/epoch=X-step=Y_lora/`
- **Checkpoint format**: Chỉ lưu LoRA weights (`.safetensors`), không lưu full model → tiết kiệm disk space
- Dataset path sẽ tự động sử dụng đường dẫn đã tìm thấy ở Bước 6

---

## 📊 BƯỚC 11: Monitor Training (Optional)

### Monitor RAM Usage

**Tạo cell mới và chạy (để theo dõi RAM trong quá trình training):**

```python
# Monitor RAM usage (chạy cell này trong tab mới để không block training)
import psutil
import time

print("📊 Đang theo dõi RAM usage...")
print("=" * 60)

while True:
    ram = psutil.virtual_memory()
    ram_total = ram.total / (1024**3)
    ram_used = ram.used / (1024**3)
    ram_percent = ram.percent
    
    print(f"RAM: {ram_used:.2f} / {ram_total:.2f} GB ({ram_percent:.1f}%)", end='\r')
    time.sleep(5)  # Update mỗi 5 giây
```

### Xem Training Log

**Tạo cell mới và chạy (để xem log):**

```python
# Xem log real-time (chạy cell này trong tab mới để không block)
import time
import glob
import os

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

**Sau khi training xong, download LoRA checkpoint:**

```python
# Tìm LoRA checkpoint mới nhất
import glob
import os

log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
lora_checkpoints = glob.glob(f"{log_dir}/*/checkpoints/*_lora/pytorch_lora_weights.safetensors")

if lora_checkpoints:
    latest_checkpoint = max(lora_checkpoints, key=os.path.getctime)
    checkpoint_dir = os.path.dirname(latest_checkpoint)
    print(f"✓ LoRA checkpoint mới nhất: {checkpoint_dir}")
    print(f"  File: {os.path.basename(latest_checkpoint)}")
    print(f"  Đã lưu trên Google Drive, có thể download về local")
    print(f"\n📦 Để sử dụng checkpoint này:")
    print(f"   1. Download toàn bộ folder: {checkpoint_dir}")
    print(f"   2. Hoặc chỉ download file: {latest_checkpoint}")
    print(f"   3. Load vào pipeline bằng: pipeline.load_lora('{checkpoint_dir}', lora_weight=1.0)")
else:
    print("ℹ Chưa có checkpoint")
```

**Cách download:**
1. Mở Google Drive
2. Vào folder `ace_step_outputs/logs/vi_lora/lightning_logs/.../checkpoints/`
3. Tìm folder có tên `epoch=X-step=Y_lora/`
4. Download toàn bộ folder hoặc chỉ file `pytorch_lora_weights.safetensors`

**Sử dụng checkpoint:**
- Checkpoint này là LoRA adapter weights, có thể load vào pipeline để inference
- File size: ~10-50MB (rất nhỏ so với full model checkpoint)

---

## 🎯 Checklist Hoàn Thành

- [ ] Đã mount Google Drive (Bước 3)
- [ ] Đã clone repository (Bước 4)
- [ ] Đã cài đặt dependencies (Bước 5)
- [ ] Đã kiểm tra dataset và config (Bước 6)
- [ ] Đã tạo thư mục output (Bước 7)
- [ ] Đã kiểm tra GPU (Bước 8)
- [ ] Đã tìm checkpoint (nếu resume) (Bước 9)
- [ ] Đã bắt đầu training (Bước 10)
- [ ] Training đang chạy (không có lỗi)

---

## 🆘 Troubleshooting

### Lỗi: "Dataset not found"
- Kiểm tra Google Drive đã được mount chưa (Bước 3)
- Đảm bảo đã upload folder `vi_lora_dataset` lên Google Drive
- Script sẽ tự động tìm dataset ở bất kỳ đâu trong MyDrive
- Nếu vẫn không tìm thấy, kiểm tra tên folder có đúng `vi_lora_dataset` không

### Lỗi: "Config not found"
- Kiểm tra file `vi_lora_config.json` trên Google Drive
- Script sẽ tự động tìm và copy config (Bước 6)
- Nếu không tìm thấy, có thể sử dụng config mặc định trong repo

### Lỗi: "Out of Memory" hoặc "Runtime disconnected" khi khởi tạo
- **Nguyên nhân**: Code đang load model vào RAM → Colab free (12.7GB) không đủ
- **Giải pháp**:
  1. Code đã được tối ưu để load từng phần và clear RAM sau mỗi bước
  2. Nếu vẫn bị OOM, nâng cấp lên **Colab Pro+** (50GB RAM)
  3. Hoặc chờ vài phút để code hoàn tất load từng phần (có thể mất 5-10 phút)

### Lỗi: "Out of Memory" khi training
- Giảm `--accumulate_grad_batches` xuống `2` hoặc `1`
- Giảm `--num_workers` xuống `0`
- Giảm `--precision` từ `16` xuống `16-mixed` (nếu có)

### Lỗi: "Runtime disconnected" khi đang train
- Code tự động lưu LoRA checkpoint mỗi 50 steps
- Resume bằng cách chạy lại Bước 10 (code sẽ tự động tìm checkpoint mới nhất)
- Đảm bảo chạy lại Bước 6 để tìm lại dataset_path

---

## 📝 Ghi Chú

- **Checkpoint format**: Chỉ lưu LoRA adapter weights (`.safetensors`), không lưu full model checkpoint
- Checkpoint được lưu mỗi **50 steps** (đã set `--every_n_train_steps 50`)
- **RAM optimization**: Code đã được tối ưu để load từng phần → phù hợp Colab free (12.7GB RAM)
- Log được lưu trên Google Drive
- Có thể resume bất cứ lúc nào từ checkpoint mới nhất (code tự động tìm và load)
- Training sẽ tự động dừng khi đạt `max_steps` (20000)
- **Checkpoint size**: ~10-50MB mỗi checkpoint (rất nhỏ so với full model)

---

## 🎯 Sử dụng Checkpoint sau khi Training

**Sau khi training xong, bạn có thể sử dụng LoRA checkpoint để inference:**

```python
from acestep.pipeline_ace_step import ACEStepPipeline

# Khởi tạo pipeline
pipeline = ACEStepPipeline(checkpoint_dir="/content/drive/MyDrive/ace_step_outputs/checkpoints")

# Load LoRA checkpoint
lora_checkpoint_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs/.../checkpoints/epoch=0-step=100_lora"
pipeline.load_lora(lora_checkpoint_dir, lora_weight=1.0)

# Generate audio
audio = pipeline(
    prompt="Nhạc Việt Nam",
    audio_duration=60.0,
    format="wav"
)
```

**Lưu ý:**
- `lora_weight`: 0.0-1.0 (1.0 = full strength, 0.5 = half strength)
- Checkpoint chỉ chứa LoRA weights, không cần load full model checkpoint

---

**Chúc bạn training thành công! 🎉**

