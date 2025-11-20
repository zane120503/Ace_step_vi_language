# Hướng dẫn Push Code lên Git Repo Mới

## Bước 1: Tạo Repository Mới trên GitHub/GitLab

1. Đăng nhập vào GitHub/GitLab
2. Tạo repository mới (không khởi tạo với README, .gitignore, license)
3. Copy URL của repository mới (ví dụ: `https://github.com/username/repo-name.git`)

## Bước 2: Thay đổi Remote URL

### Cách A: Thay đổi remote hiện tại (origin)

```powershell
cd D:\ACE-Step
git remote set-url origin https://github.com/username/your-repo-name.git
```

### Cách B: Thêm remote mới (giữ cả 2 remote)

```powershell
cd D:\ACE-Step
# Thêm remote mới với tên "myorigin"
git remote add myorigin https://github.com/username/your-repo-name.git

# Xem tất cả remote
git remote -v
```

## Bước 3: Commit Thay đổi (nếu có)

```powershell
# Xem các file đã thay đổi
git status

# Thêm tất cả thay đổi
git add .

# Commit với message
git commit -m "Update code from original repo"
```

## Bước 4: Push lên Repo Mới

```powershell
# Push lên remote mới (thay "main" bằng branch của bạn nếu khác)
git push -u origin main

# Nếu dùng remote tên khác (ví dụ "myorigin")
git push -u myorigin main
```

## Bước 5: Xác nhận

Kiểm tra trên GitHub/GitLab xem code đã được push chưa.

---

## Lưu ý

1. **Nếu repo mới chưa có branch nào**: Push lần đầu cần `-u` để set upstream
2. **Nếu có conflict**: Cần resolve conflict trước khi push
3. **Nếu muốn giữ history từ repo cũ**: Không cần làm gì, history sẽ được giữ nguyên
4. **Nếu muốn bỏ history cũ**: Cần tạo branch mới hoặc squash commits

---

## Các Lệnh Hữu Ích

```powershell
# Xem remote hiện tại
git remote -v

# Xem branch hiện tại
git branch

# Xem commit history
git log --oneline -10

# Xem thay đổi chưa commit
git status

# Xem thay đổi chi tiết
git diff
```

