#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Đọc file giống như script convert2hf_dataset.py
with open('data/phep_mau_prompt.txt', 'r', encoding='utf-8') as f:
    prompt = f.read().strip()

print("=" * 60)
print("KIỂM TRA CHI TIẾT PROMPT")
print("=" * 60)
print(f"\nNội dung file (raw):")
print(repr(prompt))
print(f"\nNội dung file (hiển thị):")
print(prompt)
print(f"\nĐộ dài: {len(prompt)} ký tự")

# Test split giống script convert2hf_dataset.py
tags = prompt.split(", ")
print(f"\n{'='*60}")
print("KẾT QUẢ SPLIT (giống script convert2hf_dataset.py):")
print(f"{'='*60}")
print(f"Số tags sau khi split bằng ', ': {len(tags)}")

if len(tags) > 1:
    print("✓ ĐÚNG FORMAT - Tags được split thành công!")
    print(f"\nDanh sách tags ({len(tags)} tags):")
    for i, tag in enumerate(tags, 1):
        print(f"  {i}. {tag.strip()}")
else:
    print("❌ SAI FORMAT - Chỉ có 1 tag (không split được)")
    print("\nNguyên nhân có thể:")
    print("  - Không có dấu phẩy + khoảng trắng giữa các tags")
    print("  - File chỉ có 1 dòng không có dấu phẩy")
    
    # Thử split bằng dấu phẩy đơn
    tags_comma = prompt.split(",")
    if len(tags_comma) > 1:
        print(f"\n⚠️  Phát hiện: File dùng dấu phẩy đơn (không có khoảng trắng)")
        print(f"   Split bằng ',' cho {len(tags_comma)} tags:")
        for i, tag in enumerate(tags_comma[:5], 1):
            print(f"     {i}. {tag.strip()}")
        print("\n   CẦN SỬA: Thêm khoảng trắng sau mỗi dấu phẩy")
        print("   Ví dụ: 'tag1, tag2, tag3' thay vì 'tag1,tag2,tag3'")

print(f"\n{'='*60}")
print("KIỂM TRA LYRICS")
print(f"{'='*60}")

with open('data/phep_mau_lyrics.txt', 'r', encoding='utf-8') as f:
    lyrics = f.read().strip()

has_verse = "[Verse" in lyrics or "[verse" in lyrics
has_chorus = "[Chorus" in lyrics or "[chorus" in lyrics
has_bridge = "[Bridge" in lyrics or "[bridge" in lyrics

print(f"Có [Verse]: {has_verse}")
print(f"Có [Chorus]: {has_chorus}")
print(f"Có [Bridge]: {has_bridge}")

if has_verse or has_chorus:
    print("✓ Lyrics có cấu trúc section")
else:
    print("⚠️  Lyrics không có cấu trúc section (không bắt buộc)")

print(f"\n{'='*60}")
print("TỔNG KẾT")
print(f"{'='*60}")

all_ok = True
if len(tags) <= 1:
    print("❌ PROMPT: Cần sửa format (tags không split được)")
    all_ok = False
else:
    print("✓ PROMPT: Đúng format")

if not (has_verse or has_chorus):
    print("⚠️  LYRICS: Không có cấu trúc section (không bắt buộc)")
else:
    print("✓ LYRICS: Đúng format")

if all_ok:
    print("\n🎉 TẤT CẢ ĐỀU ĐÚNG FORMAT - SẴN SÀNG CONVERT!")
else:
    print("\n⚠️  CẦN SỬA TRƯỚC KHI CONVERT")

