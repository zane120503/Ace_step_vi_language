# Hướng dẫn Setup Training trên Máy Remote (Debian)

## Thông tin Máy Remote

- **OS**: Debian 6.12.43
- **Python**: 3.13.5
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **CUDA**: 13.0 (từ driver)
- **Disk**: ~119GB trống
- **RAM**: 31GB (19GB available)

---

## Bước 1: Cài đặt PyTorch với CUDA

**⚠️ Lưu ý:** Python 3.13.5 rất mới, PyTorch có thể chưa hỗ trợ đầy đủ. Nếu gặp lỗi, thử cài PyTorch nightly.

Trên máy remote, chạy:

```bash
# Cài PyTorch với CUDA 12.1 (tương thích với CUDA 13.0)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Kiểm tra PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Nếu lỗi "No matching distribution found":**

```bash
# Thử cài PyTorch nightly (hỗ trợ Python 3.13 tốt hơn)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

---

## Bước 2: Clone Code từ GitHub

Trên máy remote:

```bash
cd /root
git clone https://github.com/zane120503/Ace_step_vi_language.git ACE-Step
cd ACE-Step
```

---

## Bước 3: Cài đặt Dependencies

```bash
cd /root/ACE-Step

# Cài đặt dependencies
pip3 install -r requirements.txt
```

**Nếu có lỗi, cài từng package:**

```bash
pip3 install datasets diffusers gradio librosa==0.11.0 loguru matplotlib numpy pytorch-lightning soundfile tqdm transformers==4.50.0 py3langid accelerate peft tensorboard click
```

---

## Bước 4: Copy Dataset (hoặc Tạo mới)

### Cách 1: Copy dataset từ máy local

```bash
# Từ máy Windows, copy dataset
scp -r D:\ACE-Step\vi_lora_dataset root@192.168.11.94:/root/ACE-Step/
```

### Cách 2: Tạo dataset trên máy remote

```bash
# Copy data folder lên máy remote (từ máy Windows)
scp -r D:\ACE-Step\data root@192.168.11.94:/root/ACE-Step/

# Trên máy remote, tạo dataset
cd /root/ACE-Step
python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset
```

---

## Bước 5: Chạy Training

### 5.1. Tạo tmux session (khuyến nghị)

```bash
# Tạo tmux session mới
tmux new -s training

# Trong tmux, chạy training
cd /root/ACE-Step
python3 trainer.py \
    --num_nodes 1 \
    --devices 1 \
    --dataset_path "./vi_lora_dataset" \
    --exp_name "vi_lora_small" \
    --lora_config_path "config/vi_lora_config.json" \
    --learning_rate 1e-4 \
    --accumulate_grad_batches 8 \
    --precision 16 \
    --num_workers 4 \
    --max_steps 20000 \
    --every_n_train_steps 500 \
    --shift 3.0 \
    --checkpoint_dir "./exps/checkpoints/vi_lora" \
    --logger_dir "./exps/logs/vi_lora" \
    --epochs -1 \
    --every_plot_step 2000 \
    --gradient_clip_val 0.5 \
    --gradient_clip_algorithm "norm"

# Tách khỏi tmux: Ctrl+B, sau đó D
# Xem lại tmux: tmux attach -t training
```

### 5.2. Hoặc dùng nohup

```bash
cd /root/ACE-Step

nohup python3 trainer.py \
    --num_nodes 1 \
    --devices 1 \
    --dataset_path "./vi_lora_dataset" \
    --exp_name "vi_lora_small" \
    --lora_config_path "config/vi_lora_config.json" \
    --learning_rate 1e-4 \
    --accumulate_grad_batches 8 \
    --precision 16 \
    --num_workers 4 \
    --max_steps 20000 \
    --every_n_train_steps 500 \
    --shift 3.0 \
    --checkpoint_dir "./exps/checkpoints/vi_lora" \
    --logger_dir "./exps/logs/vi_lora" \
    --epochs -1 \
    --every_plot_step 2000 \
    --gradient_clip_val 0.5 \
    --gradient_clip_algorithm "norm" \
    > training.log 2>&1 &

# Xem log
tail -f training.log
```

---

## Bước 6: Kiểm tra Training Progress

```bash
# Xem log
tail -f training.log

# Hoặc xem TensorBoard (trên máy remote)
cd /root/ACE-Step
tensorboard --logdir ./exps/logs/vi_lora --port 6006 --host 0.0.0.0

# Từ máy local, port forwarding để xem TensorBoard
# ssh -L 6006:localhost:6006 root@192.168.11.94
# Sau đó mở: http://localhost:6006
```

---

## Script Setup Tự động

### Copy script lên máy remote

```bash
# Từ máy Windows (PowerShell)
scp D:\ACE-Step\setup_remote.sh root@192.168.11.94:/tmp/

# Hoặc tạo trực tiếp trên máy remote
```

### Chạy script setup

```bash
# Trên máy remote
cd /tmp
chmod +x setup_remote.sh
bash setup_remote.sh
```

---

## Troubleshooting

### Lỗi "No matching distribution found" khi cài PyTorch

**Giải pháp:** Python 3.13 quá mới, thử cài PyTorch nightly:

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

### Lỗi "Module not found" sau khi cài

```bash
# Kiểm tra Python version
python3 --version

# Cài lại dependencies
pip3 install -r requirements.txt --upgrade --force-reinstall
```

### Lỗi "CUDA out of memory"

- Giảm batch size hoặc tăng `accumulate_grad_batches`
- Giảm `precision` xuống 8 (nếu hỗ trợ)
- Sử dụng gradient checkpointing

---

## Lưu ý Quan trọng

1. **Python 3.13**: Có thể cần cài PyTorch nightly thay vì stable
2. **tmux/screen**: Dùng để giữ training chạy sau khi đóng SSH
3. **Port forwarding**: Dùng để xem TensorBoard từ máy local
4. **Checkpoint**: Được lưu mỗi 500 steps tại `./exps/logs/vi_lora/lightning_logs/{timestamp}/checkpoints/`

---

## Tóm tắt Nhanh

```bash
# 1. Cài PyTorch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# 2. Clone code
cd /root
git clone https://github.com/zane120503/Ace_step_vi_language.git ACE-Step
cd ACE-Step

# 3. Cài dependencies
pip3 install -r requirements.txt

# 4. Tạo tmux và chạy training
tmux new -s training
# Trong tmux:
python3 trainer.py --dataset_path ./vi_lora_dataset --exp_name vi_lora_small ...
```

