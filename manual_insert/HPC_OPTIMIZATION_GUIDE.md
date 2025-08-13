# PyTorch Video Processing - HPC Optimization Guide

## 記憶體優化版本使用說明

### 概述
PyTorch 版本的 manual video inserter 專為 HPC 環境設計，特別針對大型視頻（如 4K）進行了記憶體優化。

### 關鍵優化功能

1. **記憶體高效處理**
   - 自動檢測視頻大小並選擇適當的處理策略
   - 分塊處理大型視頻以避免 GPU 記憶體溢出
   - 動態記憶體清理和監控

2. **GPU 記憶體管理**
   - 保守的 GPU 記憶體分配（60% 而非 80%）
   - 自動調整批次大小基於可用記憶體
   - 定期清理 GPU 快取

3. **HPC 環境兼容性**
   - 純 PyTorch 實現，無需 FFmpeg
   - 支援容器化部署
   - GPU 加速支援

### 使用方法

#### 1. 基本命令行使用（推薦用於 HPC）
```bash
python main.py manual-insert -- \
  --input-video outputs/final.mp4 \
  --output outputs/result.mp4 \
  --add-video overlay.mp4 \
  --insert-time "00:00.05" \
  --clip-start "00:03.00" \
  --position "75%,50%" \
  --size "46%"
```

#### 2. 使用大型視頻配置文件
```bash
python manual_inserter.py --config large_video_config.yaml \
  --input-video main_4k_video.mp4 \
  --output result_4k.mp4
```

### 記憶體使用建議

#### 對於 4K 視頻（3840x2160）
- **推薦 GPU 記憶體**: 至少 16GB
- **批次大小**: 5-10 frames
- **處理模式**: 分塊處理（自動啟用）

#### 對於 1080p 視頻
- **推薦 GPU 記憶體**: 至少 8GB  
- **批次大小**: 25-50 frames
- **處理模式**: 標準處理

#### 對於 720p 或更小視頻
- **推薦 GPU 記憶體**: 4GB+
- **批次大小**: 50+ frames
- **處理模式**: 標準處理

### 環境變數設置

建議在 HPC 環境中設置以下環境變數：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0  # 指定 GPU
```

### 錯誤處理

#### CUDA 記憶體不足錯誤
如果仍然遇到 "CUDA out of memory" 錯誤：

1. **減小批次大小**:
   ```python
   inserter = ManualVideoInserter(batch_size=2, memory_efficient=True)
   ```

2. **使用 CPU 處理**:
   ```python
   inserter = ManualVideoInserter(use_gpu=False)
   ```

3. **降低視頻解析度**:
   - 先使用 OpenCV 或其他工具降低輸入視頻解析度
   - 或使用更小的覆蓋視頻

#### 記憶體監控
系統會自動顯示記憶體使用情況：
```
GPU Memory before processing: Allocated: 2.34 GB, Reserved: 2.89 GB
GPU Memory after loading main video: Allocated: 15.67 GB, Reserved: 16.12 GB
```

### 效能最佳化建議

1. **使用 SSD 存儲**: 改善視頻讀取速度
2. **預處理視頻**: 確保視頻格式和編碼一致
3. **監控記憶體**: 關注 GPU 記憶體使用模式
4. **分階段處理**: 對於非常大的項目，考慮分階段處理

### 已知限制

1. **音頻處理**: PyTorch 版本暫不支援音頻插入
   - 建議先處理視頻，然後使用 FFmpeg 單獨添加音頻

2. **視頻格式**: 推薦使用 MP4 格式以獲得最佳兼容性

3. **記憶體需求**: 4K 視頻處理需要大量 GPU 記憶體

### 技術詳情

#### 分塊處理算法
當估計記憶體需求超過 10GB 時，系統自動啟用分塊處理：
- 將視頻分成小批次處理
- 每個批次處理後清理 GPU 記憶體
- 保持相同的視覺效果但減少記憶體使用

#### 中心座標系統
位置參數使用中心座標：
- `"center"`: 螢幕中心
- `"top-right"`: 右上角（以覆蓋視頻中心為準）
- `"75%,50%"`: 自定義位置（75% 從左，50% 從上）

這與 video-merge 模組保持一致。
