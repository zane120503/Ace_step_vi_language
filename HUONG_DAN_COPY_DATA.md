# Hướng dẫn Copy Data lên Máy Remote

## Vấn đề

Khi clone code từ GitHub, **dataset không có file `.mp3`** vì:
1. File `.mp3` bị ignore trong `.gitignore` (để tránh push file lớn)
2. Folder `vi_lora_dataset` cũng bị ignore (quá lớn, ~407MB)

Vì vậy, bạn cần **copy file `.mp3` từ máy local lên máy remote**.

---

## Giải pháp

### Cách 1: Copy toàn bộ folder `data` (Khuyến nghị)

#### Bước 1: Copy data folder từ máy Windows lên máy remote

**Từ máy Windows (PowerShell hoặc Git Bash):**

```powershell
# Sử dụng script tự động
.\copy_data_to_remote.ps1 -RemoteHost "root@192.168.11.94" -RemotePath "/root/ACE-Step"

# Hoặc copy thủ công
scp -r D:\ACE-Step\data root@192.168.11.94:/root/ACE-Step/
```

**Hoặc từ Git Bash:**

```bash
scp -r D:/ACE-Step/data root@192.168.11.94:/root/ACE-Step/
```

#### Bước 2: Trên máy remote, tạo dataset

```bash
# SSH vào máy remote
ssh root@192.168.11.94

# Di chuyển đến thư mục project
cd /root/ACE-Step

# Tạo dataset từ file .mp3
python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset
```

---

### Cách 2: Copy chỉ file `.mp3` (nhanh hơn nếu đã có file text)

#### Bước 1: Copy file `.mp3` từ máy local

**Từ máy Windows (PowerShell):**

```powershell
# Copy tất cả file .mp3
scp D:\ACE-Step\data\*.mp3 root@192.168.11.94:/root/ACE-Step/data/
```

**Hoặc từ Git Bash:**

```bash
scp D:/ACE-Step/data/*.mp3 root@192.168.11.94:/root/ACE-Step/data/
```

#### Bước 2: Trên máy remote, kiểm tra và tạo dataset

```bash
# SSH vào máy remote
ssh root@192.168.11.94
cd /root/ACE-Step

# Kiểm tra file .mp3
ls -lh data/*.mp3 | wc -l

# Tạo dataset
python3 convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset
```

---

### Cách 3: Copy dataset đã tạo sẵn (nếu đã có)

**Nếu bạn đã có dataset `vi_lora_dataset` trên máy local:**

```powershell
# Copy dataset đã tạo (rất lớn, ~407MB)
scp -r D:\ACE-Step\vi_lora_dataset root@192.168.11.94:/root/ACE-Step/
```

**Lưu ý:** Dataset rất lớn (~407MB), copy có thể mất nhiều thời gian.

---

## Script Tự động

Sử dụng script `copy_data_to_remote.ps1`:

```powershell
# Từ máy Windows (PowerShell)
cd D:\ACE-Step
.\copy_data_to_remote.ps1 -RemoteHost "root@192.168.11.94" -RemotePath "/root/ACE-Step"
```

Script sẽ:
1. Kiểm tra số lượng file MP3 trong folder `data`
2. Xác nhận trước khi copy
3. Copy toàn bộ folder `data` lên máy remote
4. Hướng dẫn các bước tiếp theo

---

## Kiểm tra Sau Khi Copy

Trên máy remote, kiểm tra:

```bash
# SSH vào máy remote
ssh root@192.168.11.94
cd /root/ACE-Step

# Kiểm tra file .mp3
ls -lh data/*.mp3 | head -5

# Đếm số file .mp3
ls -1 data/*.mp3 | wc -l

# Kiểm tra file text
ls -1 data/*_prompt.txt | wc -l
ls -1 data/*_lyrics.txt | wc -l
```

---

## Tạo Dataset trên Máy Remote

Sau khi có đầy đủ file `.mp3` và `.txt`:

```bash
cd /root/ACE-Step

# Tạo dataset
python3 convert2hf_dataset.py \
    --data_dir ./data \
    --repeat_count 2000 \
    --output_name vi_lora_dataset

# Kiểm tra dataset đã tạo
ls -lh vi_lora_dataset/
```

---

## Lưu ý Quan trọng

1. **File .mp3 rất lớn**: Copy có thể mất nhiều thời gian
2. **Kiểm tra kết nối**: Đảm bảo SSH connection ổn định
3. **Disk space**: Đảm bảo máy remote có đủ dung lượng
4. **Tạo dataset sau khi copy**: Cần chạy `convert2hf_dataset.py` sau khi copy

---

## Tóm tắt Quy trình

1. **Clone code từ GitHub** (chỉ có code, không có .mp3)
2. **Copy folder `data`** từ máy local lên máy remote
3. **Tạo dataset** trên máy remote bằng `convert2hf_dataset.py`
4. **Chạy training** với dataset đã tạo

---

## Troubleshooting

### Lỗi "Connection closed" khi copy

```bash
# Kiểm tra SSH connection
ssh root@192.168.11.94 "echo 'OK'"

# Thử copy lại với verbose mode
scp -v -r D:\ACE-Step\data root@192.168.11.94:/root/ACE-Step/
```

### Lỗi "Permission denied"

```bash
# Trên máy remote, kiểm tra quyền
ls -la /root/ACE-Step/data

# Sửa quyền nếu cần
chmod -R 755 /root/ACE-Step/data
```

### File .mp3 không được copy

```bash
# Kiểm tra file .mp3 trên máy local
Get-ChildItem D:\ACE-Step\data\*.mp3 | Select-Object Name, Length

# Kiểm tra file .mp3 trên máy remote
ssh root@192.168.11.94 "ls -lh /root/ACE-Step/data/*.mp3 | head -5"
```

