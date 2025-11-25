# Hướng dẫn Train ACE-Step LoRA trên Google Colab

## 📋 Yêu cầu

- Google Colab Pro/Pro+ (để có GPU tốt hơn và runtime lâu hơn)
- Google Drive (để lưu checkpoint và dataset)
- Dataset đã được convert sang HuggingFace format

## 🚀 Bước 1: Chuẩn bị Dataset trên Google Drive

1. Upload dataset lên Google Drive:
   - Folder `vi_lora_dataset` (đã convert)
   - Hoặc upload folder `data` và convert trên Colab

2. Upload config file:
   - `config/vi_lora_config.json`

## 🚀 Bước 2: Tạo Notebook Colab mới

1. Mở [Google Colab](https://colab.research.google.com/)
2. Tạo notebook mới
3. Chọn Runtime → Change runtime type → GPU (T4/V100/A100)

## 🚀 Bước 3: Setup môi trường

Chạy các cell sau trong notebook:

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Clone repository
```python
!git clone https://github.com/ace-step/ACE-Step.git
%cd ACE-Step
```

### Cell 3: Cài đặt dependencies
```python
!pip install -r requirements.txt
!pip install pytorch-lightning
!pip install transformers accelerate
```

### Cell 4: Setup dataset (nếu chưa convert)
```python
# Nếu dataset chưa được convert, chạy:
# !python convert2hf_dataset.py --data_dir /content/drive/MyDrive/data --repeat_count 2000 --output_name vi_lora_dataset
```

## 🚀 Bước 4: Train LoRA

### Cell 5: Chạy training
```python
import os

# Tạo thư mục output
os.makedirs("/content/drive/MyDrive/ace_step_outputs/checkpoints", exist_ok=True)
os.makedirs("/content/drive/MyDrive/ace_step_outputs/logs", exist_ok=True)

# Lệnh train
!python trainer.py \
    --num_nodes 1 \
    --devices 1 \
    --dataset_path "/content/drive/MyDrive/vi_lora_dataset" \
    --exp_name "vi_lora_small" \
    --lora_config_path "config/vi_lora_config.json" \
    --learning_rate 1e-4 \
    --accumulate_grad_batches 4 \
    --precision 16 \
    --num_workers 2 \
    --max_steps 20000 \
    --every_n_train_steps 500 \
    --shift 3.0 \
    --checkpoint_dir "/content/drive/MyDrive/ace_step_outputs/checkpoints" \
    --logger_dir "/content/drive/MyDrive/ace_step_outputs/logs" \
    --epochs -1 \
    --every_plot_step 2000 \
    --gradient_clip_val 0.5 \
    --gradient_clip_algorithm "norm"
```

## ⚠️ Lưu ý quan trọng

### 1. Runtime timeout
- Colab free: ~12 giờ timeout
- Colab Pro: ~24 giờ timeout
- **Giải pháp**: Lưu checkpoint thường xuyên (mỗi 500 steps) và resume sau

### 2. Resume từ checkpoint
```python
# Tìm checkpoint mới nhất
import glob
checkpoints = glob.glob("/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs/*/checkpoints/*.ckpt")
latest_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None

# Thêm --ckpt_path nếu có checkpoint
ckpt_arg = f"--ckpt_path {latest_checkpoint}" if latest_checkpoint else ""

!python trainer.py \
    ... (các tham số khác) ... \
    {ckpt_arg}
```

### 3. Tối ưu cho Colab GPU
- **T4 (16GB)**: Dùng `--accumulate_grad_batches 4`, `--precision 16`
- **V100 (16GB)**: Có thể tăng `--accumulate_grad_batches 8`
- **A100 (40GB)**: Có thể tăng batch size và giảm `accumulate_grad_batches`

### 4. Lưu checkpoint lên Drive
- Checkpoint tự động lưu vào `--checkpoint_dir` (đã set là Google Drive)
- Nên backup checkpoint quan trọng vào folder riêng

### 5. Monitor training
```python
# Xem log trong Colab
!tail -f /content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs/*/events.out.tfevents.*
```

## 📊 So sánh Colab vs Local

| Tiêu chí | Colab | Local (RTX 3050) |
|----------|-------|-----------------|
| GPU | T4/V100/A100 | RTX 3050 (6GB) |
| Tốc độ | Nhanh hơn (T4 ~= RTX 3050) | Chậm hơn |
| Thời gian | Giới hạn 12-24h | Không giới hạn |
| Chi phí | Free/Pro ($10/tháng) | Điện + hao mòn |
| Ổn định | Có thể bị disconnect | Ổn định hơn |
| Checkpoint | Cần lưu lên Drive | Lưu local |

## 🎯 Khuyến nghị

1. **Train ban đầu trên Colab**: Để test và xem tốc độ
2. **Train lâu dài trên Local**: Nếu có thời gian và muốn ổn định
3. **Hybrid**: Train trên Colab ban đầu, sau đó download checkpoint về local để tiếp tục

## 🔧 Troubleshooting

### Lỗi: Out of Memory
- Giảm `--accumulate_grad_batches` xuống 2 hoặc 1
- Giảm `--num_workers` xuống 0

### Lỗi: Runtime disconnected
- Resume từ checkpoint mới nhất
- Tăng tần suất lưu checkpoint (`--every_n_train_steps 200`)

### Lỗi: Drive quota full
- Xóa checkpoint cũ
- Chỉ giữ checkpoint mới nhất và các checkpoint quan trọng

