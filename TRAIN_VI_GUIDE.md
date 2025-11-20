# Hướng dẫn Train LoRA Tiếng Việt cho ACE-Step

## 📋 Tóm tắt nhanh

1. **Chuẩn bị dữ liệu**: 3 file cho mỗi bài hát (MP3 + prompt.txt + lyrics.txt)
2. **Convert dataset**: Chạy `convert2hf_dataset.py`
3. **Train LoRA**: Chạy `trainer.py` với config đã tối ưu cho RTX 3050
4. **Load LoRA**: Sử dụng checkpoint trong web UI

---

## A. Chuẩn bị dữ liệu

### Cấu trúc thư mục `data/`:

```
data/
├── vi_song_001.mp3
├── vi_song_001_prompt.txt
└── vi_song_001_lyrics.txt
├── vi_song_002.mp3
├── vi_song_002_prompt.txt
└── vi_song_002_lyrics.txt
...
```

### Format file:

#### `vi_song_001_prompt.txt`:
```
pop ballad, giọng nữ, piano, guitar, chậm, buồn, 85 bpm, minor key, emotional
```

**Gợi ý tags tiếng Việt:**
- Genre: `pop`, `ballad`, `rock`, `rap`, `electronic`, `folk`, `nhạc trữ tình`
- Giọng: `giọng nam`, `giọng nữ`, `giọng trẻ em`, `hợp xướng`
- Nhạc cụ: `piano`, `guitar`, `trống`, `violin`, `sáo`, `đàn tranh`
- Mood: `vui vẻ`, `buồn`, `lãng mạn`, `mạnh mẽ`, `nhẹ nhàng`
- Tempo: `85 bpm`, `120 bpm`, `chậm`, `nhanh`, `vừa phải`
- Key: `major key`, `minor key`, `C major`, `A minor`

#### `vi_song_001_lyrics.txt`:
```
[Verse 1]
Đêm neon vẫn sáng ngời
Phố xa vang tiếng gọi mời
Nhịp tim theo bước chân ai
Lẫn trong âm sắc nơi này

[Chorus]
Cứ bật lớn để gió hát
Cho ngọn lửa này cháy khát
Trong nhịp điệu ta chung đôi
Đêm ngân vang khúc ca này
```

**Lưu ý:**
- Tên file phải khớp chính xác: `filename.mp3`, `filename_prompt.txt`, `filename_lyrics.txt`
- Lyrics nên có cấu trúc rõ ràng với `[Verse]`, `[Chorus]`, `[Bridge]`
- Sử dụng tiếng Việt có dấu đầy đủ

---

## B. Chạy Training (2 cách)

### Cách 1: Dùng script tự động (Khuyến nghị)

**PowerShell:**
```powershell
.\run_train_vi.ps1
```

**Hoặc Windows Batch:**
```cmd
run_train_vi.bat
```

### Cách 2: Chạy thủ công từng bước

#### Bước 1: Convert dataset
```bash
python convert2hf_dataset.py --data_dir "./data" --repeat_count 2000 --output_name "vi_lora_dataset"
```

**Giải thích:**
- `--repeat_count 2000`: Nhân bản dữ liệu 2000 lần (hữu ích nếu dataset nhỏ)
- `--output_name`: Tên thư mục dataset output

#### Bước 2: Train LoRA
```bash
python trainer.py \
    --num_nodes 1 \
    --devices 1 \
    --dataset_path "./vi_lora_dataset" \
    --exp_name "vi_lora_small" \
    --lora_config_path "config/vi_lora_config.json" \
    --learning_rate 1e-4 \
    --accumulate_grad_batches 8 \
    --precision 16 \
    --num_workers 2 \
    --max_steps 20000 \
    --every_n_train_steps 500 \
    --shift 3.0 \
    --checkpoint_dir "./exps/checkpoints/vi_lora" \
    --logger_dir "./exps/logs/vi_lora" \
    --epochs -1 \
    --every_plot_step 2000 \
    --val_check_interval None \
    --gradient_clip_val 0.5 \
    --gradient_clip_algorithm "norm"
```

---

## C. Tham số tối ưu cho RTX 3050 (6GB VRAM)

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `precision` | `16` | FP16 giảm 50% VRAM |
| `accumulate_grad_batches` | `8` | Mô phỏng batch size lớn mà không tốn VRAM |
| `num_workers` | `2` | Giảm tải CPU |
| `r` (LoRA) | `16` | Rank nhỏ = ít VRAM |
| `max_steps` | `20000` | Đủ để train LoRA, có thể tăng nếu cần |

### Nếu GPU mạnh hơn (RTX 3090/4090):

- Tăng `r` lên `32` hoặc `64` trong `config/vi_lora_config.json`
- Tăng `accumulate_grad_batches` lên `16` hoặc `32`
- Tăng `max_steps` lên `50000` hoặc `100000`
- Tăng `num_workers` lên `4` hoặc `8`

---

## D. Theo dõi Training

### Logs:
- **TensorBoard**: `./exps/logs/vi_lora/`
- **Checkpoints**: `./exps/checkpoints/vi_lora/`

### Xem TensorBoard (nếu có):
```bash
tensorboard --logdir ./exps/logs/vi_lora
```

### Kiểm tra checkpoint:
Sau mỗi `every_n_train_steps` (500 steps), checkpoint sẽ được lưu tại:
```
./exps/checkpoints/vi_lora/vi_lora_small/checkpoints/epoch=*.ckpt
```

---

## E. Load LoRA vào Web UI

Sau khi training xong:

1. **Tìm file checkpoint**: 
   - Thường là file `.ckpt` hoặc `.safetensors` trong `./exps/checkpoints/vi_lora/`

2. **Trong web UI ACE-Step**:
   - Vào phần **LoRA Settings**
   - **LoRA Name or Path**: Nhập đường dẫn đến file checkpoint
   - **LoRA Weight**: Bắt đầu với `0.6-0.8`, điều chỉnh theo kết quả
   - **Generate** với prompt và lyrics tiếng Việt

3. **Ví dụ prompt trong UI**:
   ```
   pop ballad, giọng nữ, piano, guitar, chậm, buồn, 85 bpm
   ```

---

## F. Xử lý lỗi thường gặp

### 1. Out of Memory (OOM)

**Giải pháp:**
- Giảm `accumulate_grad_batches` xuống `4` hoặc `2`
- Giảm `r` trong config xuống `8`
- Đảm bảo `precision=16`
- Đóng các ứng dụng khác đang dùng GPU

### 2. Lỗi convert dataset

**Kiểm tra:**
- Tên file phải đúng pattern: `name.mp3`, `name_prompt.txt`, `name_lyrics.txt`
- Không có ký tự đặc biệt trong tên file
- File encoding phải là UTF-8

### 3. Training chậm

**Tối ưu:**
- Tăng `num_workers` nếu CPU mạnh
- Giảm `every_n_train_steps` để ít checkpoint hơn
- Kiểm tra GPU utilization bằng `nvidia-smi`

### 4. Loss không giảm

**Điều chỉnh:**
- Giảm `learning_rate` xuống `5e-5`
- Tăng `max_steps`
- Kiểm tra chất lượng dữ liệu (prompt và lyrics có khớp audio không)

---

## G. Tips & Best Practices

1. **Dataset chất lượng > số lượng**: 20-50 bài hát chất lượng tốt hơn 100 bài kém chất lượng
2. **Đa dạng phong cách**: Bao gồm nhiều genre, giọng hát, mood khác nhau
3. **Lyrics chính xác**: Lyrics phải khớp với audio, đặc biệt là timing
4. **Prompt mô tả chi tiết**: Càng chi tiết càng tốt
5. **Test thường xuyên**: Sau mỗi 5000 steps, test LoRA để xem tiến độ
6. **Backup checkpoint**: Lưu các checkpoint tốt để có thể rollback

---

## H. Checklist trước khi train

- [ ] Đã chuẩn bị ít nhất 10-20 bài hát trong thư mục `data/`
- [ ] Mỗi bài có đủ 3 file: `.mp3`, `_prompt.txt`, `_lyrics.txt`
- [ ] Tên file đúng pattern
- [ ] Đã tạo `config/vi_lora_config.json`
- [ ] Đã kích hoạt môi trường `ace_step`
- [ ] GPU có đủ VRAM (kiểm tra bằng `nvidia-smi`)
- [ ] Đã backup dữ liệu quan trọng

---

## I. Tài liệu tham khảo

- File gốc: `TRAIN_INSTRUCTION.md`
- Config mẫu: `config/zh_rap_lora_config.json`
- Script convert: `convert2hf_dataset.py`
- Trainer: `trainer.py`

---

**Chúc bạn train thành công! 🎵**

