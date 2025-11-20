# Template mẫu cho bài hát tiếng Việt

## Cấu trúc file

Mỗi bài hát cần 3 file với tên giống nhau (chỉ khác extension):

```
vi_song_001.mp3          <- File audio (MP3 format)
vi_song_001_prompt.txt   <- Mô tả đặc điểm audio
vi_song_001_lyrics.txt   <- Lời bài hát
```

---

## Ví dụ: vi_song_001_prompt.txt

```
pop ballad, giọng nữ miền Nam, piano, guitar acoustic, chậm, buồn, 85 bpm, minor key, emotional, lãng mạn, nhẹ nhàng
```

### Gợi ý tags tiếng Việt:

**Genre:**
- `pop`, `ballad`, `rock`, `rap`, `hip hop`, `electronic`, `folk`, `nhạc trữ tình`, `nhạc vàng`, `nhạc xanh`, `R&B`, `jazz`, `blues`

**Giọng hát:**
- `giọng nam`, `giọng nữ`, `giọng trẻ em`, `hợp xướng`, `giọng nam trầm`, `giọng nữ cao`, `giọng miền Bắc`, `giọng miền Nam`, `giọng miền Trung`
- **Bài hát có 2 giọng**: `giọng nam, giọng nữ, duet, hát đối đáp`, `male vocal, female vocal`

**Nhạc cụ:**
- `piano`, `guitar`, `guitar acoustic`, `guitar điện`, `trống`, `bass`, `violin`, `cello`, `sáo`, `đàn tranh`, `đàn bầu`, `organ`, `synthesizer`, `keyboard`

**Mood/Energy:**
- `vui vẻ`, `buồn`, `lãng mạn`, `mạnh mẽ`, `nhẹ nhàng`, `energetic`, `calm`, `aggressive`, `melancholic`, `upbeat`, `groovy`, `vibrant`, `dynamic`

**Tempo:**
- `85 bpm`, `120 bpm`, `chậm`, `nhanh`, `vừa phải`, `fast tempo`, `slow tempo`, `moderate tempo`

**Key:**
- `major key`, `minor key`, `C major`, `A minor`, `D major`, `E minor`

**Khác:**
- `emotional`, `atmospheric`, `driving`, `melodic`, `rhythmic`, `harmonic`

---

## Ví dụ: vi_song_001_lyrics.txt

```
[Verse 1]
Đêm neon vẫn sáng ngời
Phố xa vang tiếng gọi mời
Nhịp tim theo bước chân ai
Lẫn trong âm sắc nơi này

[Verse 2]
Bassline dội giữa lồng ngực
Hơi thở như muốn bùng nổ
Điện vang khắp lối phố quen
Giấc mơ hoá khúc du ca

[Chorus]
Cứ bật lớn để gió hát
Cho ngọn lửa này cháy khát
Trong nhịp điệu ta chung đôi
Đêm ngân vang khúc ca này

[Verse 3]
Dây đàn ngân tiếng thở dài
Đánh thức giấc ngủ u hoài
Mỗi âm thanh viết câu chuyện
Đêm nay sáng rực niềm tin

[Bridge]
Hòa giọng giữa ngàn thanh sắc
Tiếng vọng xoá tan khoảng cách
Âm vang mãi giữa trời đêm
Tiếng hô khắc khoải dịu êm

[Verse 4]
Phím đàn xoay vũ điệu mới
Gió chiều mang theo tiết tấu
Nắm câu hát giữ trong tay
Khoảnh khắc ấy vụt bay lên

[Chorus]
Cứ bật lớn để gió hát
Cho ngọn lửa này cháy khát
Trong nhịp điệu ta chung đôi
Đêm ngân vang khúc ca này

[Outro]
Đêm ngân vang khúc ca này
```

### Lưu ý về lyrics:

1. **Cấu trúc rõ ràng**: Sử dụng `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`
2. **Tiếng Việt có dấu**: Đảm bảo đầy đủ dấu thanh
3. **Khớp với audio**: Lyrics phải khớp với phần vocal trong file MP3
4. **Format nhất quán**: Mỗi dòng một câu, không cần dấu chấm cuối
5. **Dòng trống**: Để dòng trống giữa các section để dễ đọc

---

## Checklist trước khi convert

- [ ] File MP3 có chất lượng tốt (không bị méo, rè)
- [ ] Tên 3 file khớp nhau (ví dụ: `vi_song_001.*`)
- [ ] File prompt.txt có đủ tags mô tả
- [ ] File lyrics.txt có cấu trúc rõ ràng và khớp với audio
- [ ] Encoding file text là UTF-8
- [ ] Không có ký tự đặc biệt trong tên file (chỉ dùng chữ, số, gạch dưới)

---

## Ví dụ thực tế

Xem file mẫu trong thư mục `data/`:
- `test_track_001.mp3`
- `test_track_001_prompt.txt`
- `test_track_001_lyrics.txt`

Bạn có thể copy và đổi tên để tạo bài mới.

---

## Ví dụ: Bài hát có cả giọng nam và nữ (Duet)

### File: `duet_song_001_prompt.txt`

```
pop ballad, giọng nam, giọng nữ, duet, hát đối đáp, piano, guitar acoustic, chậm, lãng mạn, 90 bpm, male vocal, female vocal, emotional, harmonic
```

**Lưu ý:** Ghi rõ cả hai giọng trong prompt để model học được cả hai.

### File: `duet_song_001_lyrics.txt`

```
[Verse 1 - Nam]
Anh đã từng nghĩ về em
Như một giấc mơ xa xôi
Trong tim anh luôn có em
Dù chưa một lần gặp mặt

[Verse 2 - Nữ]
Em cũng đã từng nghĩ về anh
Như một ngôi sao sáng trên trời
Trong tim em luôn có anh
Dù chưa một lần gặp mặt

[Chorus - Duet]
Cả hai cùng hát
Tình yêu sẽ mãi mãi
Dù xa cách bao lâu
Trái tim vẫn nhớ nhau

[Bridge - Nam]
Anh sẽ tìm em
Dù đường xa vạn dặm
Anh sẽ đến bên em
Để nói lời yêu thương

[Bridge - Nữ]
Em sẽ đợi anh
Dù thời gian trôi qua
Em sẽ ở đây
Để đón anh về nhà

[Chorus - Duet]
Cả hai cùng hát
Tình yêu sẽ mãi mãi
Dù xa cách bao lâu
Trái tim vẫn nhớ nhau
```

**Lưu ý:**
- Ghi rõ `[Verse X - Nam]` hoặc `[Verse X - Nữ]` để model biết phần nào là giọng nào
- Phần `[Chorus - Duet]` hoặc `[Chorus]` là cả hai cùng hát
- Format này giúp model học được cả hai giọng và cách hát đối đáp

### Cấu trúc file:

```
data/
├── duet_song_001.mp3
├── duet_song_001_prompt.txt
└── duet_song_001_lyrics.txt
```

**Không cần tách thành 2 file riêng** - giữ nguyên bài hát gốc và mô tả chi tiết trong prompt + lyrics là đủ.

