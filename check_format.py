#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kiểm tra format file dataset"""

import os
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def check_file_format():
    data_dir = Path("data")
    
    # Kiểm tra file phep_mau
    mp3_file = data_dir / "phep_mau.mp3"
    prompt_file = data_dir / "phep_mau_prompt.txt"
    lyrics_file = data_dir / "phep_mau_lyrics.txt"
    
    print("=" * 60)
    print("KIỂM TRA FORMAT FILE: phep_mau")
    print("=" * 60)
    
    # 1. Kiểm tra file tồn tại
    print("\n[1] Kiểm tra file tồn tại:")
    print(f"  ✓ MP3: {mp3_file.exists()} - {mp3_file}")
    print(f"  ✓ Prompt: {prompt_file.exists()} - {prompt_file}")
    print(f"  ✓ Lyrics: {lyrics_file.exists()} - {lyrics_file}")
    
    if not all([mp3_file.exists(), prompt_file.exists(), lyrics_file.exists()]):
        print("  ❌ THIẾU FILE!")
        return False
    
    # 2. Kiểm tra tên file khớp nhau
    print("\n[2] Kiểm tra tên file:")
    expected_prompt = str(mp3_file).replace(".mp3", "_prompt.txt")
    expected_lyrics = str(mp3_file).replace(".mp3", "_lyrics.txt")
    
    if str(prompt_file) == expected_prompt and str(lyrics_file) == expected_lyrics:
        print("  ✓ Tên file khớp nhau")
    else:
        print(f"  ❌ Tên file không khớp!")
        print(f"     Expected prompt: {expected_prompt}")
        print(f"     Actual prompt: {prompt_file}")
        print(f"     Expected lyrics: {expected_lyrics}")
        print(f"     Actual lyrics: {lyrics_file}")
        return False
    
    # 3. Kiểm tra format prompt
    print("\n[3] Kiểm tra format prompt:")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    
    print(f"  Content: {prompt[:50]}...")
    
    # Script split bằng ", " (dấu phẩy + khoảng trắng)
    tags = prompt.split(", ")
    print(f"  Số tags (split bằng ', '): {len(tags)}")
    
    if len(tags) == 1 and tags[0] == prompt:
        print("  ⚠️  CẢNH BÁO: Không tìm thấy dấu phẩy + khoảng trắng!")
        print("     Script sẽ không split được tags đúng cách")
        print("     Đề xuất: Đảm bảo tags cách nhau bằng ', ' (dấu phẩy + khoảng trắng)")
    else:
        print(f"  ✓ Tags được split thành công: {len(tags)} tags")
        print(f"  Ví dụ tags: {tags[:3]}")
    
    # 4. Kiểm tra format lyrics
    print("\n[4] Kiểm tra format lyrics:")
    with open(lyrics_file, "r", encoding="utf-8") as f:
        lyrics = f.read().strip()
    
    has_verse = "[Verse" in lyrics or "[verse" in lyrics
    has_chorus = "[Chorus" in lyrics or "[chorus" in lyrics
    has_bridge = "[Bridge" in lyrics or "[bridge" in lyrics
    
    print(f"  Có [Verse]: {has_verse}")
    print(f"  Có [Chorus]: {has_chorus}")
    print(f"  Có [Bridge]: {has_bridge}")
    
    if has_verse or has_chorus:
        print("  ✓ Có cấu trúc section (Verse/Chorus)")
    else:
        print("  ⚠️  Không có cấu trúc section (không bắt buộc nhưng nên có)")
    
    # 5. Kiểm tra encoding
    print("\n[5] Kiểm tra encoding:")
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            f.read()
        with open(lyrics_file, "r", encoding="utf-8") as f:
            f.read()
        print("  ✓ Encoding UTF-8 hợp lệ")
    except UnicodeDecodeError as e:
        print(f"  ❌ Lỗi encoding: {e}")
        return False
    
    # 6. Tổng kết
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM TRA:")
    print("=" * 60)
    
    issues = []
    if len(tags) == 1:
        issues.append("⚠️  Prompt có thể không split được tags đúng (kiểm tra dấu phẩy + khoảng trắng)")
    
    if issues:
        print("\nCác vấn đề cần lưu ý:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ TẤT CẢ ĐỀU ĐÚNG FORMAT!")
    
    return True

if __name__ == "__main__":
    check_file_format()

