#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

with open('data/phep_mau_prompt.txt', 'r', encoding='utf-8') as f:
    prompt = f.read().strip()

print("Nội dung file:")
print(repr(prompt))
print("\nĐộ dài:", len(prompt))
print("\nKiểm tra split:")

# Test split bằng ", "
tags_comma_space = prompt.split(", ")
print(f"Split bằng ', ' (dấu phẩy + khoảng trắng): {len(tags_comma_space)} tags")
if len(tags_comma_space) > 1:
    print("  ✓ ĐÚNG FORMAT!")
    print(f"  Tags: {tags_comma_space}")
else:
    print("  ❌ SAI FORMAT - chỉ có 1 tag")
    # Thử split bằng dấu phẩy đơn
    tags_comma = prompt.split(",")
    print(f"\nSplit bằng ',' (chỉ dấu phẩy): {len(tags_comma)} tags")
    if len(tags_comma) > 1:
        print("  ⚠️  File dùng dấu phẩy đơn, không có khoảng trắng!")
        print("  Tags:", tags_comma)
        print("\n  CẦN SỬA: Thêm khoảng trắng sau dấu phẩy")
        print("  Ví dụ: 'tag1, tag2, tag3' thay vì 'tag1,tag2,tag3'")

