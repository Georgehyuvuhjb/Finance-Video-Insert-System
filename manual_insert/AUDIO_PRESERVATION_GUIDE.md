# 音頻保留解決方案

## 問題描述

在高效視頻處理中，原音頻軌道會丟失，因為我們只處理了視頻流。

## 解決方案

### 自動音頻保留

新版本會自動嘗試保留原音頻：

```
🚀 Using efficient segment-based processing with audio preservation...
💡 Memory optimization: Only processing 89/1946 frames (95.4% memory saved)
📺 Processing segment 1/1
🎵 Preserving original audio...
✅ Audio successfully preserved using FFmpeg!
```

### 音頻保留策略

系統按優先級使用以下方法：

#### 1. FFmpeg (首選)
- **優點**: 最快、最可靠
- **要求**: 系統需安裝 FFmpeg
- **使用**: 自動複製音頻流，無重新編碼

```bash
# 安裝 FFmpeg (HPC 環境)
conda install ffmpeg
# 或
apt-get install ffmpeg
```

#### 2. moviepy (備選)
- **優點**: 純 Python 解決方案
- **要求**: pip install moviepy
- **使用**: 當 FFmpeg 不可用時自動切換

```bash
pip install moviepy
```

#### 3. 手動處理 (最後選項)
- **情況**: 無 FFmpeg 和 moviepy 時
- **結果**: 提供手動指令

## 技術實現

### 處理流程

```
1. 使用 OpenCV 處理視頻片段 → temp_video.mp4 (無音頻)
2. 使用 FFmpeg 複製原音頻 → final_video.mp4 (有音頻)
3. 清理臨時文件
```

### FFmpeg 命令

```bash
ffmpeg -y \
  -i processed_video.mp4 \
  -i original_video.mp4 \
  -c:v copy \
  -c:a aac \
  -map 0:v:0 \
  -map 1:a:0 \
  -shortest \
  output_with_audio.mp4
```

### moviepy 代碼

```python
from moviepy.editor import VideoFileClip

# 載入處理後的視頻
video = VideoFileClip('processed_video.mp4')
# 載入原音頻
audio = VideoFileClip('original_video.mp4').audio
# 合併
video.set_audio(audio).write_videofile('output_with_audio.mp4')
```

## 使用效果

### 成功案例
```
🎵 Preserving original audio...
✅ Audio successfully preserved using FFmpeg!
✅ Efficient processing with audio preservation completed successfully!
```

### 備選情況
```
⚠️ FFmpeg not available, using alternative audio preservation method...
Using moviepy for audio preservation...
✅ Audio successfully preserved using moviepy!
```

### 失敗情況
```
📋 Video saved without audio. To manually add audio, use:
1. Using FFmpeg:
   ffmpeg -i processed.mp4 -i original.mp4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 final.mp4

2. Using Python moviepy:
   from moviepy.editor import VideoFileClip
   video = VideoFileClip('processed.mp4')
   audio = VideoFileClip('original.mp4').audio
   video.set_audio(audio).write_videofile('final.mp4')
```

## HPC 環境建議

### 預安裝工具
```bash
# 方法 1: 使用 conda
conda install ffmpeg

# 方法 2: 使用系統包管理器
sudo apt-get install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg      # CentOS/RHEL

# 方法 3: 使用 Python 替代
pip install moviepy
```

### 驗證安裝
```bash
# 檢查 FFmpeg
ffmpeg -version

# 檢查 moviepy
python -c "from moviepy.editor import VideoFileClip; print('moviepy OK')"
```

## 性能比較

| 方法 | 速度 | 記憶體使用 | 依賴 | 推薦度 |
|------|------|------------|------|--------|
| FFmpeg | 最快 | 最低 | 外部工具 | ⭐⭐⭐⭐⭐ |
| moviepy | 中等 | 中等 | Python 包 | ⭐⭐⭐⭐ |
| 手動處理 | N/A | N/A | 無 | ⭐⭐ |

## 故障排除

### 常見問題

1. **"FFmpeg not found"**
   - 解決: 安裝 FFmpeg 或使用 moviepy

2. **"moviepy not available"**
   - 解決: `pip install moviepy`

3. **音頻編碼錯誤**
   - 解決: 檢查原視頻音頻格式，可能需要轉換

4. **音頻時長不匹配**
   - 解決: 系統自動使用 `-shortest` 參數處理

### 檢查音頻軌道

```python
# 檢查視頻是否有音頻
python manual_insert/test_audio_preservation.py
```

這個解決方案確保在高效處理視頻的同時完全保留原音頻軌道。
