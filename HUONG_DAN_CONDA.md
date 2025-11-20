# Hướng dẫn Sử dụng Conda trong PowerShell

## Vấn đề

Khi chạy `conda activate` trong PowerShell, có thể gặp lỗi:
```
conda : The term 'conda' is not recognized
```
hoặc
```
CondaError: Run 'conda init' before 'conda activate'
```

## Giải pháp

### Cách 1: Khởi tạo Conda trong PowerShell (Khuyến nghị)

1. **Mở PowerShell với quyền Administrator**

2. **Chạy lệnh init:**
```powershell
& "C:\Users\admin\anaconda3\Scripts\conda.exe" init powershell
```

3. **Đóng và mở lại PowerShell** để áp dụng thay đổi

4. **Kiểm tra:**
```powershell
conda --version
conda activate ace_step
```

### Cách 2: Sử dụng Conda Run (Không cần activate)

Thay vì `conda activate`, sử dụng `conda run`:

```powershell
# Thay vì:
conda activate ace_step
python script.py

# Dùng:
conda run -n ace_step python script.py
```

### Cách 3: Sử dụng Script Helper

Tạo file `activate_env.ps1`:

```powershell
# activate_env.ps1
& "C:\Users\admin\anaconda3\Scripts\conda.exe" run -n ace_step powershell
```

Sau đó chạy:
```powershell
.\activate_env.ps1
```

### Cách 4: Sử dụng Anaconda Prompt

Thay vì PowerShell, sử dụng **Anaconda Prompt** (đã có sẵn conda):

1. Tìm "Anaconda Prompt" trong Start Menu
2. Mở Anaconda Prompt
3. Chạy:
```bash
conda activate ace_step
```

---

## Lưu ý

- Sau khi chạy `conda init powershell`, cần **đóng và mở lại PowerShell** mới có hiệu lực
- Nếu đường dẫn Anaconda khác, thay đổi đường dẫn trong các lệnh trên
- Để kiểm tra đường dẫn Anaconda: `where.exe conda`

---

## Các Lệnh Hữu Ích

```powershell
# Kiểm tra conda đã cài chưa
where.exe conda

# Kiểm tra version
conda --version

# Liệt kê environments
conda env list

# Tạo environment mới
conda create -n ten_env python=3.10

# Xóa environment
conda env remove -n ten_env

# Cài package
conda install -n ace_step package_name
# hoặc
conda run -n ace_step pip install package_name
```

