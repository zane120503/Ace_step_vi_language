# Hướng dẫn Training trên Máy Remote (SSH)

## Bước 1: Kiểm tra Môi trường Máy Remote

### 1.1. SSH vào máy remote

```bash
ssh user@remote-ip
# Ví dụ: ssh root@192.168.1.100
```

### 1.2. Chạy script kiểm tra

**Cách 1: Copy script lên máy remote**

```bash
# Từ máy local (Windows)
scp D:\ACE-Step\check_remote_environment.sh user@remote-ip:/tmp/

# SSH vào máy remote
ssh user@remote-ip
cd /tmp
chmod +x check_remote_environment.sh
./check_remote_environment.sh
```

**Cách 2: Chạy trực tiếp các lệnh kiểm tra**

```bash
# Kiểm tra Python
python3 --version
which python3

# Kiểm tra pip
pip3 --version
which pip3

# Kiểm tra GPU
nvidia-smi

# Kiểm tra CUDA
nvcc --version
nvidia-smi | grep "CUDA Version"

# Kiểm tra disk space
df -h

# Kiểm tra RAM
free -h
```

---

## Bước 2: Cài đặt Môi trường (nếu thiếu)

### 2.1. Cài đặt Python (nếu chưa có)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip
```

### 2.2. Cài đặt Miniconda (khuyến nghị)

```bash
# Download Miniconda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Cài đặt
bash Miniconda3-latest-Linux-x86_64.sh

# Khởi động lại shell hoặc:
source ~/.bashrc

# Tạo environment mới
conda create -n ace_step python=3.10 -y
conda activate ace_step
```

### 2.3. Cài đặt CUDA và PyTorch

**Kiểm tra CUDA version trên máy remote:**

```bash
nvidia-smi | grep "CUDA Version"
```

**⚠️ QUAN TRỌNG: Debian/Ubuntu với Python 3.13+ bảo vệ system Python**

Nếu gặp lỗi `externally-managed-environment`, bạn **PHẢI** tạo virtual environment trước!

### Tạo Virtual Environment (BẮT BUỘC nếu gặp lỗi externally-managed-environment)

```bash
# 1. Cài python3-venv (nếu chưa có)
apt-get update
apt-get install -y python3-full python3-venv

# 2. Tạo virtual environment tại /root
cd /root
python3 -m venv ace_step_env

# 3. Kích hoạt virtual environment (QUAN TRỌNG!)
source /root/ace_step_env/bin/activate

# Bạn sẽ thấy (ace_step_env) ở đầu prompt sau khi activate
# Ví dụ: (ace_step_env) root@5S-VP:~#

# 4. Kiểm tra (sẽ thấy python3 trong /root/ace_step_env/bin/)
which python3
which pip3
```

**Lưu ý:** Mỗi khi mở terminal mới hoặc tmux session mới, bạn **PHẢI** activate lại:
```bash
source /root/ace_step_env/bin/activate
```

**Cài đặt PyTorch phù hợp:**

```bash
# ĐẢM BẢO đã activate virtual environment trước!
# (Sẽ thấy (ace_step_env) ở đầu prompt)

# Với CUDA 12.1 (nightly - hỗ trợ Python 3.13 tốt hơn)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Hoặc stable version (nếu nightly có vấn đề)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Với CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Kiểm tra PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## Bước 3: Copy Mã nguồn lên Máy Remote

### 3.1. Tạo thư mục trên máy remote

```bash
# SSH vào máy remote
ssh user@remote-ip

# Tạo thư mục
mkdir -p ~/ACE-Step
cd ~/ACE-Step
```

### 3.2. Copy code từ máy local

**Cách 1: Sử dụng scp (từ máy Windows)**

```powershell
# Copy toàn bộ project
scp -r D:\ACE-Step\* user@remote-ip:~/ACE-Step/

# Hoặc chỉ copy file cần thiết
scp D:\ACE-Step\trainer.py user@remote-ip:~/ACE-Step/
scp D:\ACE-Step\requirements.txt user@remote-ip:~/ACE-Step/
scp -r D:\ACE-Step\acestep user@remote-ip:~/ACE-Step/
scp -r D:\ACE-Step\config user@remote-ip:~/ACE-Step/
scp -r D:\ACE-Step\data user@remote-ip:~/ACE-Step/  # Nếu cần
```

**Cách 2: Sử dụng rsync (từ máy Linux/Mac hoặc WSL)**

```bash
rsync -avz --exclude 'exps' --exclude '*.pyc' --exclude '__pycache__' --exclude '.git' D:/ACE-Step/ user@remote-ip:~/ACE-Step/
```

**Cách 3: Sử dụng Git (khuyến nghị)**

```bash
# Trên máy remote
cd ~
git clone https://github.com/zane120503/Ace_step_vi_language.git ACE-Step
cd ACE-Step
```

**Cách 4: Tạo tar.gz và copy**

```powershell
# Trên máy Windows, tạo file nén
cd D:\ACE-Step
tar -czf ace-step.tar.gz --exclude='exps' --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' .

# Copy file nén
scp ace-step.tar.gz user@remote-ip:/tmp/

# Trên máy remote, giải nén
ssh user@remote-ip
cd ~
mkdir -p ACE-Step
cd ACE-Step
tar -xzf /tmp/ace-step.tar.gz
```

---

## Bước 4: Cài đặt Dependencies

```bash
# SSH vào máy remote
ssh user@remote-ip
cd ~/ACE-Step

# Cài đặt dependencies
pip3 install -r requirements.txt

# Hoặc nếu dùng conda
conda activate ace_step
pip install -r requirements.txt
```

---

## Bước 5: Copy Dataset (nếu cần)

```bash
# Từ máy local, copy dataset
scp -r D:\ACE-Step\vi_lora_dataset user@remote-ip:~/ACE-Step/

# Hoặc tạo dataset trên máy remote
# (cần có file data trước)
```

---

## Bước 6: Chạy Training

⚠️ **QUAN TRỌNG:** Đảm bảo đã activate virtual environment trước khi chạy training!

```bash
# SSH vào máy remote
ssh user@remote-ip
cd ~/ACE-Step

# Kích hoạt virtual environment (QUAN TRỌNG!)
source /root/ace_step_env/bin/activate

# Hoặc nếu dùng conda:
# conda activate ace_step

# Chạy training
python3 trainer.py \
    --num_nodes 1 \
    --devices 1 \
    --dataset_path "./vi_lora_dataset" \
    --exp_name "vi_lora_small" \
    --lora_config_path "config/vi_lora_config.json" \
    --learning_rate 1e-4 \
    --accumulate_grad_batches 8 \
    --precision 16 \
    --num_workers 0 \
    --max_steps 20000 \
    --every_n_train_steps 500 \
    --shift 3.0 \
    --checkpoint_dir "./exps/checkpoints/vi_lora" \
    --logger_dir "./exps/logs/vi_lora" \
    --epochs -1 \
    --every_plot_step 2000 \
    --gradient_clip_val 0.5 \
    --gradient_clip_algorithm "norm"
```

---

## Bước 7: Giữ Training Chạy Sau Khi Đóng SSH

### 7.1. Sử dụng screen

```bash
# Cài đặt screen
sudo apt install screen  # Ubuntu/Debian

# Tạo screen session mới
screen -S training

# TRONG screen, kích hoạt virtual environment trước!
source /root/ace_step_env/bin/activate
cd /root/ACE-Step

# Chạy training trong screen
python3 trainer.py ...

# Tách khỏi screen: Ctrl+A, sau đó D
# Xem lại screen: screen -r training
# Liệt kê screens: screen -ls
```

### 7.2. Sử dụng tmux (khuyến nghị)

```bash
# Cài đặt tmux
sudo apt install tmux  # Ubuntu/Debian

# Tạo tmux session mới
tmux new -s training

# TRONG tmux, kích hoạt virtual environment trước!
source /root/ace_step_env/bin/activate
cd /root/ACE-Step

# Chạy training trong tmux
python3 trainer.py ...

# Tách khỏi tmux: Ctrl+B, sau đó D
# Xem lại tmux: tmux attach -t training
# Liệt kê sessions: tmux ls
```

### 7.3. Sử dụng nohup

```bash
# Chạy training với nohup (kích hoạt virtual environment trong lệnh)
nohup bash -c "source /root/ace_step_env/bin/activate && cd /root/ACE-Step && python3 trainer.py ..." > training.log 2>&1 &

# Xem log
tail -f training.log

# Kiểm tra process
ps aux | grep trainer.py
```

---

## Bước 8: Kiểm tra Training Progress

```bash
# SSH vào máy remote
ssh user@remote-ip
cd ~/ACE-Step

# Xem log
tail -f training.log

# Hoặc xem TensorBoard
tensorboard --logdir ./exps/logs/vi_lora --port 6006

# Từ máy local, port forwarding:
# ssh -L 6006:localhost:6006 user@remote-ip
# Sau đó mở: http://localhost:6006
```

---

## Lưu ý Quan trọng

1. **Kiểm tra GPU trước**: Đảm bảo GPU hoạt động tốt
2. **Disk space**: Dataset và checkpoint cần nhiều dung lượng
3. **Network**: Copy code qua mạng có thể mất thời gian
4. **Port forwarding**: Dùng port forwarding để xem TensorBoard từ xa
5. **Backup**: Backup checkpoint thường xuyên

---

## Troubleshooting

### Lỗi "CUDA out of memory"

- Giảm batch size hoặc tăng `accumulate_grad_batches`
- Giảm `precision` xuống 16 hoặc 8
- Sử dụng gradient checkpointing

### Lỗi "Permission denied"

```bash
chmod +x script.sh
chmod -R 755 ~/ACE-Step
```

### Lỗi "Module not found"

```bash
pip3 install -r requirements.txt --upgrade
```

---

## Script Tự động

Sử dụng script `setup_remote_training.sh` để tự động setup:

```bash
# Copy script lên máy remote
scp D:\ACE-Step\setup_remote_training.sh user@remote-ip:/tmp/

# SSH và chạy
ssh user@remote-ip
cd /tmp
chmod +x setup_remote_training.sh
./setup_remote_training.sh
```

