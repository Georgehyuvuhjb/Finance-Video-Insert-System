#!/usr/bin/env python3
"""
Efficient Video Processor - Only processes modified segments
"""

import cv2
import torch
import numpy as np
from tqdm import tqdm
import tempfile
import os

class EfficientVideoProcessor:
    """高效視頻處理器 - 只處理需要修改的片段"""
    
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def process_video_with_overlay_efficient(self, main_video_path, overlay_video_path, 
                                           output_path, start_time, duration, position, size):
        """
        高效處理：只讀取和處理需要修改的片段
        
        Args:
            main_video_path: 主影片路徑
            overlay_video_path: 覆蓋影片路徑
            output_path: 輸出路徑
            start_time: 開始時間（秒）
            duration: 持續時間（秒）
            position: 位置
            size: 大小
        """
        print("Using efficient segment-based processing...")
        
        # 獲取主影片信息
        main_cap = cv2.VideoCapture(main_video_path)
        fps = main_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(main_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(main_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 計算需要修改的幀範圍
        start_frame = int(start_time * fps)
        end_frame = min(total_frames, start_frame + int(duration * fps))
        
        print(f"Main video: {width}x{height}, {fps} fps, {total_frames} frames")
        print(f"Processing only frames {start_frame} to {end_frame} (total: {end_frame - start_frame} frames)")
        print(f"Memory savings: Processing {((end_frame - start_frame) / total_frames) * 100:.1f}% of total frames")
        
        # 設置輸出視頻編碼器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 讀取覆蓋影片
        overlay_frames = self._load_overlay_video(overlay_video_path, duration, fps, size, width, height)
        if overlay_frames is None:
            print("Failed to load overlay video")
            return False
        
        # 計算覆蓋位置
        overlay_h, overlay_w = overlay_frames.shape[1:3]
        x, y = self._calculate_position(position, width, height, overlay_w, overlay_h)
        
        print(f"Overlay: {overlay_w}x{overlay_h} at position ({x}, {y})")
        
        # 分段處理策略
        if start_frame > 0:
            # 1. 複製開始部分（不修改）
            print("Copying beginning segment...")
            self._copy_video_segment(main_cap, out, 0, start_frame)
        
        # 2. 處理需要覆蓋的片段
        print("Processing overlay segment...")
        self._process_overlay_segment(main_cap, out, overlay_frames, start_frame, end_frame, x, y)
        
        # 3. 複製結尾部分（不修改）
        if end_frame < total_frames:
            print("Copying ending segment...")
            self._copy_video_segment(main_cap, out, end_frame, total_frames)
        
        # 清理
        main_cap.release()
        out.release()
        print(f"Efficient processing completed! Output: {output_path}")
        return True
    
    def _copy_video_segment(self, cap, out, start_frame, end_frame):
        """直接複製視頻片段，不進行處理"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        with tqdm(total=end_frame-start_frame, desc="Copying frames") as pbar:
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                pbar.update(1)
    
    def _process_overlay_segment(self, cap, out, overlay_frames, start_frame, end_frame, x, y):
        """只處理需要覆蓋的片段"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        overlay_count = len(overlay_frames)
        overlay_h, overlay_w = overlay_frames.shape[1:3]
        
        with tqdm(total=end_frame-start_frame, desc="Processing overlay") as pbar:
            for frame_idx in range(start_frame, end_frame):
                ret, main_frame = cap.read()
                if not ret:
                    break
                
                # 選擇對應的覆蓋幀（循環）
                overlay_idx = (frame_idx - start_frame) % overlay_count
                overlay_frame = overlay_frames[overlay_idx]
                
                # 應用覆蓋（只修改需要的區域）
                main_frame[y:y+overlay_h, x:x+overlay_w] = overlay_frame
                
                out.write(main_frame)
                pbar.update(1)
    
    def _load_overlay_video(self, overlay_path, duration, main_fps, size, main_width, main_height):
        """載入並預處理覆蓋影片"""
        cap = cv2.VideoCapture(overlay_path)
        if not cap.isOpened():
            return None
        
        # 計算目標大小
        target_width, target_height = self._parse_size(size, main_width, main_height)
        
        frames = []
        frame_count = int(duration * main_fps)
        
        print(f"Loading {frame_count} overlay frames...")
        with tqdm(total=frame_count, desc="Loading overlay") as pbar:
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    # 如果覆蓋影片較短，重新開始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # 調整大小
                if frame.shape[:2] != (target_height, target_width):
                    frame = cv2.resize(frame, (target_width, target_height))
                
                frames.append(frame)
                pbar.update(1)
        
        cap.release()
        return np.array(frames)
    
    def _parse_size(self, size_spec, main_width, main_height):
        """解析大小規格"""
        if '%' in size_spec:
            percent = float(size_spec.replace('%', ''))
            width = int((percent / 100) * main_width)
            # 保持 16:9 比例
            height = int(width * 9 / 16)
            return width, height
        elif 'x' in size_spec:
            width_str, height_str = size_spec.split('x')
            return int(width_str), int(height_str)
        else:
            # 預設 25% 寬度
            width = int(0.25 * main_width)
            height = int(width * 9 / 16)
            return width, height
    
    def _calculate_position(self, position, main_width, main_height, overlay_width, overlay_height):
        """計算覆蓋位置（中心座標轉換為左上角）"""
        if position == "center":
            center_x, center_y = main_width // 2, main_height // 2
        elif position == "top-right":
            center_x = main_width - overlay_width // 2
            center_y = overlay_height // 2
        elif position == "bottom-left":
            center_x, center_y = overlay_width // 2, main_height - overlay_height // 2
        elif ',' in position:
            x_str, y_str = position.split(',')
            if '%' in x_str:
                center_x = int((float(x_str.replace('%', '')) / 100) * main_width)
            else:
                center_x = int(x_str)
            if '%' in y_str:
                center_y = int((float(y_str.replace('%', '')) / 100) * main_height)
            else:
                center_y = int(y_str)
        else:
            center_x, center_y = main_width // 2, main_height // 2
        
        # 轉換中心座標為左上角座標
        x = center_x - overlay_width // 2
        y = center_y - overlay_height // 2
        
        # 確保在邊界內
        x = max(0, min(x, main_width - overlay_width))
        y = max(0, min(y, main_height - overlay_height))
        
        return x, y


def main():
    """測試高效處理器"""
    processor = EfficientVideoProcessor(use_gpu=False)
    
    # 測試參數（模擬用戶的使用案例）
    main_video = "outputs/final.mp4"
    overlay_video = "outputs/videos/172894_finance_large.mp4"
    output_video = "outputs/efficient_result.mp4"
    
    # 只處理 0.05 秒開始的 3 秒片段
    processor.process_video_with_overlay_efficient(
        main_video_path=main_video,
        overlay_video_path=overlay_video,
        output_path=output_video,
        start_time=0.05,  # 0.05 秒開始
        duration=3.0,     # 持續 3 秒
        position="75%,50%",
        size="46%"
    )

if __name__ == "__main__":
    main()
