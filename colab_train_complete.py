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
# Clone từ repository tiếng Việt (đã có config và dataset cho tiếng Việt)
!git clone https://github.com/zane120503/Ace_step_vi_language.git
%cd Ace_step_vi_language

# ============================================
# CELL 3: Cài đặt Dependencies
# ============================================
# Xử lý tất cả dependency conflicts một cách thông minh

print("🔧 Đang fix các dependency conflicts...")

# Bước 1: Uninstall các packages có conflict để clean install
!pip uninstall -y numpy protobuf fsspec tensorboard 2>/dev/null || true

# Bước 2: Cài đặt các packages với version cố định (trước khi cài requirements.txt)
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
!pip install -q --no-deps -r requirements.txt 2>&1 | head -20 || true

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

# ============================================
# CELL 4: Kiểm tra Dataset và Config
# ============================================
import os
import glob

# Bước 1: Kiểm tra Google Drive đã được mount chưa
if not os.path.exists("/content/drive"):
    print("❌ Google Drive chưa được mount!")
    print("   Vui lòng chạy CELL 1 (Mount Google Drive) trước!")
    raise FileNotFoundError("Google Drive not mounted!")

# Bước 2: Tìm tất cả các folder có tên "vi_lora_dataset" trên Drive
print("🔍 Đang tìm dataset trên Google Drive...")
drive_root = "/content/drive/MyDrive"

# Tìm tất cả folder vi_lora_dataset
found_datasets = []
for root, dirs, files in os.walk(drive_root):
    if "vi_lora_dataset" in dirs:
        full_path = os.path.join(root, "vi_lora_dataset")
        if os.path.isdir(full_path):
            file_count = len(os.listdir(full_path))
            found_datasets.append((full_path, file_count))

# Bước 3: Chọn dataset phù hợp
dataset_path = None
if found_datasets:
    # Ưu tiên dataset trong ace_step_data
    for path, count in found_datasets:
        if "ace_step_data" in path:
            dataset_path = path
            print(f"✓ Dataset tìm thấy tại: {path}")
            print(f"✓ Số file trong dataset: {count}")
            break
    
    # Nếu không có trong ace_step_data, dùng dataset đầu tiên
    if not dataset_path:
        dataset_path, count = found_datasets[0]
        print(f"✓ Dataset tìm thấy tại: {path}")
        print(f"✓ Số file trong dataset: {count}")
        if len(found_datasets) > 1:
            print(f"⚠️  Tìm thấy {len(found_datasets)} dataset, đang dùng: {dataset_path}")
            print("   Các dataset khác:")
            for path, count in found_datasets[1:]:
                print(f"     - {path} ({count} files)")
else:
    # Kiểm tra các đường dẫn phổ biến
    possible_paths = [
        "/content/drive/MyDrive/ace_step_data/vi_lora_dataset",
        "/content/drive/MyDrive/MyDrive/ace_step_data/vi_lora_dataset",  # Nếu có folder MyDrive trong MyDrive
        "/content/drive/MyDrive/vi_lora_dataset",
        "/content/drive/MyDrive/data/vi_lora_dataset",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            files = os.listdir(path)
            print(f"✓ Dataset tìm thấy tại: {path}")
            print(f"✓ Số file trong dataset: {len(files)}")
            break

if not dataset_path:
    print("❌ Không tìm thấy dataset!")
    print("\n📋 HƯỚNG DẪN:")
    print("1. Đảm bảo Google Drive đã được mount (CELL 1)")
    print("2. Upload folder 'vi_lora_dataset' lên Google Drive")
    print("3. Có thể đặt ở bất kỳ đâu trong MyDrive")
    print("4. Chạy lại cell này sau khi upload")
    print("\n💡 Hoặc chỉ định đường dẫn thủ công:")
    print("   dataset_path = '/content/drive/MyDrive/your_path/vi_lora_dataset'")

# Kiểm tra và copy config
config_paths = [
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
# Nếu dataset_path đã được tìm thấy ở CELL 4, sử dụng nó
# Nếu không, tự động tìm lại
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
        # Thử các đường dẫn phổ biến
        possible_paths = [
            "/content/drive/MyDrive/MyDrive/ace_step_data/vi_lora_dataset",  # Trường hợp có MyDrive trong MyDrive
            "/content/drive/MyDrive/ace_step_data/vi_lora_dataset",
            "/content/drive/MyDrive/vi_lora_dataset",
            "/content/drive/MyDrive/data/vi_lora_dataset",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                print(f"✓ Tìm thấy dataset: {dataset_path}")
                break
    
    if dataset_path is None or not os.path.exists(dataset_path):
        print("❌ Vẫn không tìm thấy dataset!")
        print("   Vui lòng upload dataset lên Google Drive và chỉ định đường dẫn:")
        print("   dataset_path = '/content/drive/MyDrive/your_path/vi_lora_dataset'")
        raise FileNotFoundError("Dataset not found! Please upload to Google Drive first.")

# Xác nhận dataset path
print(f"📂 Sử dụng dataset: {dataset_path}")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path không tồn tại: {dataset_path}")

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

