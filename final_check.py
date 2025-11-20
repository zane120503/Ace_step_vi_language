#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("KIỂM TRA FORMAT CUỐI CÙNG - phep_mau")
print("=" * 70)

data_dir = Path("data")
mp3_file = data_dir / "phep_mau.mp3"
prompt_file = data_dir / "phep_mau_prompt.txt"
lyrics_file = data_dir / "phep_mau_lyrics.txt"

# 1. Kiểm tra file tồn tại
print("\n[1] KIỂM TRA FILE TỒN TẠI:")
all_exist = all([mp3_file.exists(), prompt_file.exists(), lyrics_file.exists()])
print(f"  MP3: {'✓' if mp3_file.exists() else '✗'} {mp3_file.name}")
print(f"  Prompt: {'✓' if prompt_file.exists() else '✗'} {prompt_file.name}")
print(f"  Lyrics: {'✓' if lyrics_file.exists() else '✗'} {lyrics_file.name}")

if not all_exist:
    print("\n❌ THIẾU FILE!")
    sys.exit(1)

# 2. Kiểm tra tên file
print("\n[2] KIỂM TRA TÊN FILE:")
expected_prompt = str(mp3_file).replace(".mp3", "_prompt.txt")
expected_lyrics = str(mp3_file).replace(".mp3", "_lyrics.txt")
name_ok = (str(prompt_file) == expected_prompt and str(lyrics_file) == expected_lyrics)
print(f"  Tên file khớp: {'✓' if name_ok else '✗'}")

# 3. Kiểm tra prompt
print("\n[3] KIỂM TRA FORMAT PROMPT:")
try:
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read().strip()
    
    print(f"  Độ dài: {len(prompt_content)} ký tự")
    
    if len(prompt_content) == 0:
        print("  ❌ File rỗng!")
        print("  ⚠️  CẢNH BÁO: File có thể chưa được lưu. Hãy lưu file trong editor!")
        sys.exit(1)
    
    # Test split
    tags = prompt_content.split(", ")
    print(f"  Số tags (split bằng ', '): {len(tags)}")
    
    if len(tags) > 1:
        print("  ✓ ĐÚNG FORMAT - Tags được split thành công!")
        print(f"  Ví dụ tags: {tags[:3]}")
        prompt_ok = True
    else:
        print("  ❌ SAI FORMAT - Tags không split được")
        # Kiểm tra xem có dấu phẩy không
        if "," in prompt_content:
            tags_comma = prompt_content.split(",")
            print(f"  ⚠️  Phát hiện: Có {len(tags_comma)} phần khi split bằng ','")
            print("     → Cần thêm khoảng trắng sau dấu phẩy")
        prompt_ok = False
        
except Exception as e:
    print(f"  ❌ Lỗi đọc file: {e}")
    prompt_ok = False

# 4. Kiểm tra lyrics
print("\n[4] KIỂM TRA FORMAT LYRICS:")
try:
    with open(lyrics_file, 'r', encoding='utf-8') as f:
        lyrics_content = f.read().strip()
    
    has_verse = "[Verse" in lyrics_content or "[verse" in lyrics_content
    has_chorus = "[Chorus" in lyrics_content or "[chorus" in lyrics_content
    has_bridge = "[Bridge" in lyrics_content or "[bridge" in lyrics_content
    
    print(f"  Có [Verse]: {'✓' if has_verse else '✗'}")
    print(f"  Có [Chorus]: {'✓' if has_chorus else '✗'}")
    print(f"  Có [Bridge]: {'✓' if has_bridge else '✗'}")
    
    if has_verse or has_chorus:
        print("  ✓ Có cấu trúc section")
        lyrics_ok = True
    else:
        print("  ⚠️  Không có cấu trúc section (không bắt buộc)")
        lyrics_ok = True  # Không bắt buộc
        
except Exception as e:
    print(f"  ❌ Lỗi đọc file: {e}")
    lyrics_ok = False

# 5. Tổng kết
print("\n" + "=" * 70)
print("KẾT QUẢ CUỐI CÙNG:")
print("=" * 70)

if all_exist and name_ok and prompt_ok and lyrics_ok:
    print("\n🎉 TẤT CẢ ĐỀU ĐÚNG FORMAT!")
    print("   ✓ File tồn tại")
    print("   ✓ Tên file khớp")
    print("   ✓ Prompt format đúng")
    print("   ✓ Lyrics format đúng")
    print("\n✅ SẴN SÀNG CONVERT DATASET!")
    print("\nChạy lệnh:")
    print("  python convert2hf_dataset.py --data_dir ./data --repeat_count 2000 --output_name vi_lora_dataset")
else:
    print("\n⚠️  CÓ VẤN ĐỀ CẦN SỬA:")
    if not all_exist:
        print("  - Thiếu file")
    if not name_ok:
        print("  - Tên file không khớp")
    if not prompt_ok:
        print("  - Prompt format sai (kiểm tra dấu phẩy + khoảng trắng)")
    if not lyrics_ok:
        print("  - Lyrics có lỗi")

