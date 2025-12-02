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
    --every_n_train_steps 50 \
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
- **Giải pháp**: 
  - Lưu checkpoint thường xuyên (mỗi 50 steps - đã set)
  - Code tự động tìm và resume từ checkpoint mới nhất
  - Checkpoint format: Chỉ lưu LoRA adapter (~10-50MB) → tiết kiệm disk space

### 1.5. RAM Usage khi khởi tạo
- **Colab free (12.7GB RAM)**: Code đã được tối ưu để load từng phần → có thể chạy được
- **Khởi tạo mất 5-10 phút**: Code sẽ load từng model và clear RAM sau mỗi bước
- **Nếu vẫn OOM**: Nâng cấp lên Colab Pro+ (50GB RAM) hoặc chờ code hoàn tất load

### 2. Resume từ checkpoint
**Lưu ý**: Code sẽ tự động tìm và load LoRA checkpoint nếu có, không cần `--ckpt_path`

```python
# Kiểm tra checkpoint có tồn tại không (optional)
import glob
import os

log_dir = "/content/drive/MyDrive/ace_step_outputs/logs/vi_lora/lightning_logs"
lora_checkpoints = glob.glob(f"{log_dir}/*/checkpoints/*_lora/pytorch_lora_weights.safetensors")

if lora_checkpoints:
    latest = max(lora_checkpoints, key=os.path.getctime)
    print(f"✓ Tìm thấy LoRA checkpoint: {os.path.dirname(latest)}")
    print("  Code sẽ tự động load khi training")
else:
    print("ℹ Chưa có checkpoint, sẽ train từ đầu")

# Chạy training (code tự động resume nếu có checkpoint)
!python trainer.py \
    ... (các tham số khác) ...
```

**Checkpoint format mới:**
- Chỉ lưu LoRA adapter weights (`.safetensors`)
- Không lưu full Lightning checkpoint (`.ckpt`)
- File size: ~10-50MB mỗi checkpoint

### 3. Tối ưu cho Colab GPU
- **T4 (16GB)**: Dùng `--accumulate_grad_batches 4`, `--precision 16`
- **V100 (16GB)**: Có thể tăng `--accumulate_grad_batches 8`
- **A100 (40GB)**: Có thể tăng batch size và giảm `accumulate_grad_batches`

### 4. Lưu checkpoint lên Drive
- **Checkpoint format**: Chỉ lưu LoRA adapter weights (`.safetensors`), không lưu full model
- Checkpoint tự động lưu vào `--logger_dir/.../checkpoints/epoch=X-step=Y_lora/`
- **File size**: ~10-50MB mỗi checkpoint (rất nhỏ so với full model checkpoint ~GB)
- Nên backup checkpoint quan trọng vào folder riêng
- **Để sử dụng**: Load vào pipeline bằng `pipeline.load_lora(checkpoint_dir, lora_weight=1.0)`

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

### Lỗi: Out of Memory khi khởi tạo
- **Nguyên nhân**: Model 3.3B params cần ~13GB RAM khi load
- **Giải pháp**:
  1. Code đã tối ưu để load từng phần → chờ 5-10 phút để hoàn tất
  2. Nếu vẫn OOM: Nâng cấp lên **Colab Pro+** (50GB RAM)
  3. Hoặc train trên máy local (RTX 3050) như hiện tại

### Lỗi: Out of Memory khi training
- Giảm `--accumulate_grad_batches` xuống 2 hoặc 1
- Giảm `--num_workers` xuống 0
- Giảm `--precision` từ `16` xuống `16-mixed` (nếu có)

### Lỗi: Runtime disconnected
- Code tự động lưu checkpoint mỗi 50 steps
- Resume bằng cách chạy lại cell training (code tự động tìm checkpoint mới nhất)
- Không cần `--ckpt_path` vì chỉ train LoRA adapter

### Lỗi: Drive quota full
- Xóa checkpoint cũ
- Chỉ giữ checkpoint mới nhất và các checkpoint quan trọng

