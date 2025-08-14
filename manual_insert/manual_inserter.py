#!/usr/bin/env python3
"""
Manual Video and Audio Inserter (PyTorch Version)
================================================

A module for manually inserting videos and audio clips into a main video based on user configuration.
Uses PyTorch for video processing to ensure HPC compatibility.

Usage:
    python manual_inserter.py --config config.yaml --input-video main.mp4 --output final.mp4
    python manual_inserter.py --show-recommendations matches.json
"""

import os
import json
import yaml
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
from pathlib import Path
import tempfile
from datetime import datetime
import shutil
from tqdm import tqdm
from PIL import Image
import gc

class PyTorchVideoProcessor:
    """Video processing using PyTorch with GPU acceleration and memory optimization"""
    
    def __init__(self, use_gpu=True, batch_size=25, memory_efficient=True, use_segment_processing=True):
        """
        Initialize the video processor with memory optimization
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration
            batch_size (int): Batch size for processing frames (reduced default for memory efficiency)
            memory_efficient (bool): Enable memory-efficient processing for large videos
            use_segment_processing (bool): Use segment-based processing (only process modified parts)
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.memory_efficient = memory_efficient
        self.use_segment_processing = use_segment_processing
        
        print(f"Using device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Memory efficient mode: {self.memory_efficient}")
        print(f"Segment processing: {self.use_segment_processing}")
        
        # Memory management for CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            # Set conservative memory fraction for large videos
            torch.cuda.set_per_process_memory_fraction(0.6)
            
            # Get and display GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU total memory: {total_memory:.1f} GB")
            
            # Adjust batch size based on available memory
            if total_memory < 16:  # Less than 16GB
                self.batch_size = min(self.batch_size, 10)
                print(f"Adjusted batch size to {self.batch_size} due to limited GPU memory")
        
        # Initialize transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
        ])
        
        self.inverse_transform = T.ToPILImage()
    
    def create_temp_file(self, suffix=".mp4"):
        """創建臨時文件"""
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        return temp_path
    
    def process_video_efficiently(self, main_video_path, overlay_configs, output_path):
        """
        高效處理：只讀取和修改需要的視頻片段，保留原音頻
        
        Args:
            main_video_path: 主影片路徑
            overlay_configs: 覆蓋配置列表
            output_path: 輸出路徑
            
        Returns:
            bool: 處理是否成功
        """
        if not self.use_segment_processing:
            # 使用原有的全量處理方法
            return self._process_video_full_load(main_video_path, overlay_configs, output_path)
        
        print("🚀 Using efficient segment-based processing with audio preservation...")
        
        # 獲取主影片信息
        main_cap = cv2.VideoCapture(main_video_path)
        if not main_cap.isOpened():
            print(f"Error: Cannot open main video {main_video_path}")
            return False
        
        fps = main_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(main_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(main_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Main video: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # 分析所有覆蓋操作，找出需要修改的片段
        segments = self._analyze_overlay_segments(overlay_configs, fps, total_frames)
        
        if not segments:
            print("No overlay segments found, copying original video...")
            return self._copy_entire_video(main_video_path, output_path)
        
        # 計算記憶體節省
        total_modified_frames = sum(seg['end'] - seg['start'] for seg in segments)
        memory_savings = (1 - total_modified_frames / total_frames) * 100
        print(f"💡 Memory optimization: Only processing {total_modified_frames}/{total_frames} frames ({memory_savings:.1f}% memory saved)")
        
        # 創建臨時視頻文件（無音頻）
        temp_video_path = self.create_temp_file(".mp4")
        
        # 設置輸出編碼器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Cannot create temporary video {temp_video_path}")
            main_cap.release()
            return False
        
        try:
            # 處理每個片段
            current_frame = 0
            
            for i, segment in enumerate(segments):
                print(f"\n📺 Processing segment {i+1}/{len(segments)}")
                
                # 複製未修改的部分
                if current_frame < segment['start']:
                    print(f"Copying frames {current_frame} to {segment['start']} (unmodified)")
                    self._copy_video_segment_cv2(main_cap, out, current_frame, segment['start'])
                
                # 處理需要覆蓋的部分
                print(f"Processing overlay frames {segment['start']} to {segment['end']}")
                self._process_overlay_segment_cv2(
                    main_cap, out, segment, width, height, fps
                )
                
                current_frame = segment['end']
            
            # 複製剩餘的未修改部分
            if current_frame < total_frames:
                print(f"Copying remaining frames {current_frame} to {total_frames} (unmodified)")
                self._copy_video_segment_cv2(main_cap, out, current_frame, total_frames)
            
            main_cap.release()
            out.release()
            
            # 使用 FFmpeg 將音頻從原影片複製到處理後的影片
            print("\n🎵 Preserving original audio...")
            success = self._add_audio_to_video(main_video_path, temp_video_path, output_path)
            
            if success:
                print("✅ Efficient processing with audio preservation completed successfully!")
                return True
            else:
                print("⚠️ Video processing completed but audio preservation failed")
                # 如果音頻處理失敗，至少保存無音頻版本
                import shutil
                shutil.copy2(temp_video_path, output_path)
                return True
            
        except Exception as e:
            print(f"Error during efficient processing: {e}")
            return False
            
        finally:
            if main_cap.isOpened():
                main_cap.release()
            if out.isOpened():
                out.release()
    
    def _process_video_full_load(self, main_video_path, overlay_configs, output_path):
        """
        原有的全量處理方法，完整載入視頻進行處理
        
        Args:
            main_video_path: 主影片路徑
            overlay_configs: 覆蓋配置列表
            output_path: 輸出路徑
            
        Returns:
            bool: 處理是否成功
        """
        try:
            print("🔄 Using full video loading method...")
            
            # 讀取主影片
            main_cap = cv2.VideoCapture(main_video_path)
            if not main_cap.isOpened():
                print(f"Error: Cannot open main video {main_video_path}")
                return False
            
            fps = main_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(main_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(main_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 讀取所有主影片幀
            print(f"Loading all {total_frames} frames into memory...")
            main_frames = []
            for i in range(total_frames):
                ret, frame = main_cap.read()
                if not ret:
                    break
                main_frames.append(frame)
            
            main_cap.release()
            
            # 處理每個覆蓋配置
            for config in overlay_configs:
                overlay_path = config.get("overlay_video")
                position = config.get("position", "center")
                start_time = config.get("start_time", "00:00:00")
                duration = config.get("duration", "00:00:10")
                size = config.get("size")
                
                # 讀取覆蓋影片
                start_seconds = self.parse_time_to_seconds(start_time) if isinstance(start_time, str) else start_time
                duration_seconds = self.parse_time_to_seconds(duration) if isinstance(duration, str) else duration
                overlay_frames = self.read_video_frames(overlay_path, start_seconds, duration_seconds)
                if not overlay_frames:
                    print(f"Failed to read overlay video: {overlay_path}")
                    continue
                
                # 執行覆蓋
                main_frames = self.overlay_frames(main_frames, overlay_frames, position, start_time, duration, fps)
            
            # 寫入輸出視頻
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in main_frames:
                out.write(frame)
            
            out.release()
            print("✅ Full video processing completed!")
            return True
            
        except Exception as e:
            print(f"Error in full video processing: {e}")
            return False

    def _add_audio_to_video(self, original_video_path, processed_video_path, output_path):
        """
        使用 FFmpeg 將原音頻添加到處理後的視頻，如果 FFmpeg 不可用則提供替代方案
        
        Args:
            original_video_path: 原始視頻路徑
            processed_video_path: 處理後的視頻路徑（無音頻）
            output_path: 最終輸出路徑
            
        Returns:
            bool: 是否成功
        """
        # 首先嘗試使用 FFmpeg
        if self._try_ffmpeg_audio_preservation(original_video_path, processed_video_path, output_path):
            return True
        
        # FFmpeg 不可用時的替代方案
        print("⚠️ FFmpeg not available, using alternative audio preservation method...")
        return self._try_alternative_audio_preservation(original_video_path, processed_video_path, output_path)
    
    def _try_ffmpeg_audio_preservation(self, original_video_path, processed_video_path, output_path):
        """嘗試使用 FFmpeg 保留音頻"""
        try:
            import subprocess
            
            # 檢查 FFmpeg 是否可用
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            
            # 使用 FFmpeg 複製音頻流
            cmd = [
                'ffmpeg', '-y',
                '-i', processed_video_path,  # 處理後的視頻
                '-i', original_video_path,   # 原始視頻（用於音頻）
                '-c:v', 'copy',              # 複製視頻流（不重新編碼）
                '-c:a', 'aac',               # 音頻編碼
                '-map', '0:v:0',             # 使用第一個輸入的視頻流
                '-map', '1:a:0',             # 使用第二個輸入的音頻流
                '-shortest',                 # 以較短的流為準
                output_path
            ]
            
            print("Executing FFmpeg command to preserve audio...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print("✅ Audio successfully preserved using FFmpeg!")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
            print(f"FFmpeg not available or failed: {e}")
            return False
    
    def _try_alternative_audio_preservation(self, original_video_path, processed_video_path, output_path):
        """使用 moviepy 作為 FFmpeg 的替代方案"""
        try:
            # 嘗試使用 moviepy（如果安裝了的話）
            from moviepy.editor import VideoFileClip
            
            print("Using moviepy for audio preservation...")
            
            # 載入處理後的視頻（無音頻）
            processed_clip = VideoFileClip(processed_video_path)
            
            # 載入原始視頻的音頻
            original_clip = VideoFileClip(original_video_path)
            
            # 將原音頻添加到處理後的視頻
            final_clip = processed_clip.set_audio(original_clip.audio)
            
            # 寫入最終視頻
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            # 清理
            processed_clip.close()
            original_clip.close()
            final_clip.close()
            
            print("✅ Audio successfully preserved using moviepy!")
            return True
            
        except ImportError:
            print("moviepy not available for audio preservation")
            return self._fallback_copy_with_audio_note(original_video_path, processed_video_path, output_path)
        except Exception as e:
            print(f"Error using moviepy: {e}")
            return self._fallback_copy_with_audio_note(original_video_path, processed_video_path, output_path)
    
    def _fallback_copy_with_audio_note(self, original_video_path, processed_video_path, output_path):
        """
        備用方案：複製處理後的視頻並提供音頻處理建議
        """
        import shutil
        try:
            shutil.copy2(processed_video_path, output_path)
            print("📋 Video saved without audio. To manually add audio, use one of these methods:")
            print(f"\n1. Using FFmpeg:")
            print(f"   ffmpeg -i {processed_video_path} -i {original_video_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_path.replace('.mp4', '_with_audio.mp4')}")
            print(f"\n2. Using Python moviepy:")
            print(f"   from moviepy.editor import VideoFileClip")
            print(f"   video = VideoFileClip('{processed_video_path}')")
            print(f"   audio = VideoFileClip('{original_video_path}').audio")
            print(f"   video.set_audio(audio).write_videofile('{output_path.replace('.mp4', '_with_audio.mp4')}')")
            print("\n⚠️ Current output has no audio track")
            return True
        except Exception as e:
            print(f"Error copying video: {e}")
            return False
    
    def _analyze_overlay_segments(self, overlay_configs, fps, total_frames):
        """分析覆蓋配置，找出需要修改的片段"""
        segments = []
        
        for config in overlay_configs:
            start_time = config['start_time']
            duration = config['duration']
            
            start_frame = int(start_time * fps)
            end_frame = min(total_frames, start_frame + int(duration * fps))
            
            if start_frame < end_frame:
                segments.append({
                    'start': start_frame,
                    'end': end_frame,
                    'config': config
                })
        
        # 按開始時間排序
        segments.sort(key=lambda x: x['start'])
        
        # 合併重疊的片段（可選優化）
        merged_segments = []
        for segment in segments:
            if merged_segments and segment['start'] <= merged_segments[-1]['end']:
                # 擴展上一個片段
                merged_segments[-1]['end'] = max(merged_segments[-1]['end'], segment['end'])
                # 合併配置（簡化處理，可以進一步優化）
                if 'configs' not in merged_segments[-1]:
                    merged_segments[-1]['configs'] = [merged_segments[-1]['config']]
                merged_segments[-1]['configs'].append(segment['config'])
            else:
                merged_segments.append(segment)
        
        return merged_segments
    
    def _copy_video_segment_cv2(self, cap, out, start_frame, end_frame):
        """使用 OpenCV 直接複製視頻片段（無處理）"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        with tqdm(total=end_frame-start_frame, desc="Copying frames") as pbar:
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_idx}")
                    break
                out.write(frame)
                pbar.update(1)
    
    def _process_overlay_segment_cv2(self, cap, out, segment, width, height, fps):
        """處理需要覆蓋的片段"""
        start_frame = segment['start']
        end_frame = segment['end']
        
        # 獲取覆蓋配置
        if 'configs' in segment:
            # 處理合併的片段（多個覆蓋）
            configs = segment['configs']
        else:
            configs = [segment['config']]
        
        # 載入所有覆蓋影片
        overlay_data = []
        for config in configs:
            overlay_frames = self._load_overlay_frames_cv2(
                config['source'], config['duration'], fps, 
                config['size'], width, height
            )
            if overlay_frames is not None:
                x, y = self._calculate_position_cv2(
                    config['position'], width, height, 
                    overlay_frames.shape[2], overlay_frames.shape[1]
                )
                overlay_data.append({
                    'frames': overlay_frames,
                    'x': x, 'y': y,
                    'start_time': config['start_time']
                })
        
        # 設置讀取位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 處理每一幀
        with tqdm(total=end_frame-start_frame, desc="Processing overlay") as pbar:
            for frame_idx in range(start_frame, end_frame):
                ret, main_frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_idx}")
                    break
                
                # 應用所有覆蓋
                for overlay in overlay_data:
                    # 計算相對於覆蓋開始的幀索引
                    overlay_start_frame = int(overlay['start_time'] * fps)
                    relative_frame = frame_idx - overlay_start_frame
                    
                    if relative_frame >= 0:
                        overlay_frames = overlay['frames']
                        overlay_idx = relative_frame % len(overlay_frames)
                        overlay_frame = overlay_frames[overlay_idx]
                        
                        x, y = overlay['x'], overlay['y']
                        h, w = overlay_frame.shape[:2]
                        
                        # 確保不超出邊界
                        if (x + w <= width and y + h <= height and 
                            x >= 0 and y >= 0):
                            main_frame[y:y+h, x:x+w] = overlay_frame
                
                out.write(main_frame)
                pbar.update(1)
    
    def _load_overlay_frames_cv2(self, overlay_path, duration, main_fps, size, main_width, main_height):
        """載入覆蓋影片幀"""
        cap = cv2.VideoCapture(overlay_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open overlay video {overlay_path}")
            return None
        
        # 計算目標大小
        target_width, target_height = self.calculate_size(size, main_width, main_height)
        if target_height == -1:
            # 保持寬高比
            target_height = int(target_width * 9 / 16)
        
        frames = []
        frame_count = int(duration * main_fps)
        
        print(f"Loading {frame_count} overlay frames from {overlay_path}")
        with tqdm(total=frame_count, desc="Loading overlay", leave=False) as pbar:
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    # 重新開始（循環）
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
        return np.array(frames) if frames else None
    
    def _calculate_position_cv2(self, position, main_width, main_height, overlay_width, overlay_height):
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
    
    def _copy_entire_video(self, input_path, output_path):
        """直接複製整個影片（無修改）"""
        import shutil
        try:
            shutil.copy2(input_path, output_path)
            print(f"Copied original video to {output_path}")
            return True
        except Exception as e:
            print(f"Error copying video: {e}")
            return False
    
    def read_video_frames(self, video_path, start_time=0.0, duration=None):
        """
        Read video frames into tensor format with memory optimization
        
        Args:
            video_path (str): Path to video file
            start_time (float): Start time in seconds
            duration (float): Duration in seconds (None for entire video)
            
        Returns:
            tuple: (frames_tensor, video_info)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        if duration is not None:
            end_frame = min(total_frames, start_frame + int(duration * fps))
        else:
            end_frame = total_frames
        
        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Estimate memory requirements and use streaming if necessary
        total_frames_to_read = end_frame - start_frame
        estimated_memory_gb = (total_frames_to_read * width * height * 3 * 4) / (1024**3)  # float32
        
        print(f"Reading frames {start_frame} to {end_frame} from {video_path}")
        print(f"Estimated memory requirement: {estimated_memory_gb:.2f} GB")
        
        # If memory requirement is too high, use streaming approach
        if self.memory_efficient and estimated_memory_gb > 20:
            print("Large video detected - using memory-efficient streaming approach")
            return self._read_video_frames_streaming(cap, start_frame, end_frame, width, height, fps)
        
        # Standard approach for smaller videos
        frames = []
        current_frame = start_frame
        
        with tqdm(total=total_frames_to_read, desc="Loading video") as pbar:
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor and move to CPU first to save GPU memory
                frame_tensor = self.transform(Image.fromarray(frame))
                frames.append(frame_tensor.cpu())  # Keep on CPU initially
                
                current_frame += 1
                pbar.update(1)
                
                # Clear GPU cache periodically
                if self.device.type == 'cuda' and current_frame % 100 == 0:
                    torch.cuda.empty_cache()
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames could be read from video")
        
        # Stack frames into tensor (keep on CPU for now)
        frames_tensor = torch.stack(frames)
        print(f"Loaded video tensor shape: {frames_tensor.shape}")
        
        video_info = {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames_to_read
        }
        
        return frames_tensor, video_info
    
    def _read_video_frames_streaming(self, cap, start_frame, end_frame, width, height, fps):
        """
        Streaming approach for very large videos - processes in chunks
        """
        # For streaming approach, we'll return a generator-like object
        # For now, implement a simplified version with smaller chunks
        chunk_size = min(100, end_frame - start_frame)  # Process in smaller chunks
        frames = []
        current_frame = start_frame
        
        print(f"Processing video in chunks of {chunk_size} frames")
        
        with tqdm(total=end_frame-start_frame, desc="Loading video (streaming)") as pbar:
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor
                frame_tensor = self.transform(Image.fromarray(frame))
                frames.append(frame_tensor.cpu())
                
                current_frame += 1
                pbar.update(1)
                
                # Process chunk when it's full
                if len(frames) >= chunk_size:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
        
        if not frames:
            raise ValueError("No frames could be read from video")
        
        frames_tensor = torch.stack(frames)
        print(f"Loaded video tensor shape (streaming): {frames_tensor.shape}")
        
        video_info = {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': len(frames)
        }
        
        return frames_tensor, video_info
        
        if frames:
            frames_tensor = torch.stack(frames)
            return frames_tensor, {"fps": fps, "width": width, "height": height}
        else:
            return None, {"fps": fps, "width": width, "height": height}
    
    def resize_frames(self, frames_tensor, target_size):
        """Resize frames tensor"""
        if frames_tensor is None:
            return None
            
        target_h, target_w = target_size
        frames_tensor = frames_tensor.to(self.device)
        
        # Use PyTorch's interpolate for resizing
        resized_tensor = F.interpolate(
            frames_tensor,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        return resized_tensor
    
    def overlay_frames(self, main_frames, overlay_frames, position, start_time, duration, main_fps):
        """
        Overlay frames at specific position and time with memory optimization
        
        Args:
            main_frames: Main video frames tensor
            overlay_frames: Overlay video frames tensor 
            position: Position specification
            start_time: Start time in seconds
            duration: Duration in seconds
            main_fps: Main video frame rate
            
        Returns:
            Tensor with overlaid frames
        """
        if main_frames is None or overlay_frames is None:
            return main_frames
        
        # Check memory requirements and process accordingly
        main_count, _, main_h, main_w = main_frames.shape
        overlay_count, _, overlay_h, overlay_w = overlay_frames.shape
        
        # Estimate memory requirement for overlay operation
        estimated_memory_gb = (main_count * main_h * main_w * 3 * 4 * 2) / (1024**3)  # *2 for result tensor
        
        if self.memory_efficient and estimated_memory_gb > 10:
            print(f"Large overlay operation detected ({estimated_memory_gb:.2f} GB) - using chunked processing")
            return self._overlay_frames_chunked(main_frames, overlay_frames, position, start_time, duration, main_fps)
        
        # Standard processing for smaller operations
        return self._overlay_frames_standard(main_frames, overlay_frames, position, start_time, duration, main_fps)
    
    def _overlay_frames_standard(self, main_frames, overlay_frames, position, start_time, duration, main_fps):
        """Standard overlay processing"""
        # Move to device
        main_frames = main_frames.to(self.device)
        overlay_frames = overlay_frames.to(self.device)
        
        main_count, _, main_h, main_w = main_frames.shape
        overlay_count, _, overlay_h, overlay_w = overlay_frames.shape
        
        # Calculate time range in frames
        start_frame = int(start_time * main_fps)
        end_frame = int((start_time + duration) * main_fps)
        end_frame = min(end_frame, main_count)
        
        # Calculate center-based position
        x, y = self.calculate_center_position(position, main_w, main_h, overlay_w, overlay_h)
        
        # Ensure overlay fits within main video bounds
        if x + overlay_w > main_w or y + overlay_h > main_h or x < 0 or y < 0:
            # Resize overlay to fit
            max_w = main_w - max(0, x)
            max_h = main_h - max(0, y)
            
            if max_w > 0 and max_h > 0:
                # Maintain aspect ratio
                aspect_ratio = overlay_w / overlay_h
                if max_w / aspect_ratio <= max_h:
                    new_w, new_h = max_w, int(max_w / aspect_ratio)
                else:
                    new_w, new_h = int(max_h * aspect_ratio), max_h
                
                overlay_frames = self.resize_frames(overlay_frames, (new_h, new_w))
                overlay_h, overlay_w = new_h, new_w
                
                # Recalculate position
                x, y = self.calculate_center_position(position, main_w, main_h, overlay_w, overlay_h)
        
        # Ensure position is within bounds
        x = max(0, min(x, main_w - overlay_w))
        y = max(0, min(y, main_h - overlay_h))
        
        # Create result tensor
        result_frames = main_frames.clone()
        
        print(f"Overlaying {overlay_count} frames at position ({x}, {y}) from frame {start_frame} to {end_frame}")
        
        # Apply overlay for the specified time range
        with tqdm(total=end_frame-start_frame, desc="Applying overlay") as pbar:
            for frame_idx in range(start_frame, end_frame):
                if frame_idx >= main_count:
                    break
                
                # Calculate corresponding overlay frame (with looping)
                overlay_idx = (frame_idx - start_frame) % overlay_count
                
                # Apply overlay
                result_frames[frame_idx, :, y:y+overlay_h, x:x+overlay_w] = overlay_frames[overlay_idx]
                
                pbar.update(1)
        
        return result_frames.cpu()  # Move result back to CPU to save GPU memory
    
    def _overlay_frames_chunked(self, main_frames, overlay_frames, position, start_time, duration, main_fps):
        """Memory-efficient chunked overlay processing for large videos"""
        main_count, _, main_h, main_w = main_frames.shape
        overlay_count, _, overlay_h, overlay_w = overlay_frames.shape
        
        # Calculate parameters
        start_frame = int(start_time * main_fps)
        end_frame = int((start_time + duration) * main_fps)
        end_frame = min(end_frame, main_count)
        
        # Calculate center-based position
        x, y = self.calculate_center_position(position, main_w, main_h, overlay_w, overlay_h)
        
        # Handle overlay resizing if needed
        if x + overlay_w > main_w or y + overlay_h > main_h or x < 0 or y < 0:
            max_w = main_w - max(0, x)
            max_h = main_h - max(0, y)
            
            if max_w > 0 and max_h > 0:
                aspect_ratio = overlay_w / overlay_h
                if max_w / aspect_ratio <= max_h:
                    new_w, new_h = max_w, int(max_w / aspect_ratio)
                else:
                    new_w, new_h = int(max_h * aspect_ratio), max_h
                
                overlay_frames = self.resize_frames(overlay_frames, (new_h, new_w))
                overlay_h, overlay_w = new_h, new_w
                x, y = self.calculate_center_position(position, main_w, main_h, overlay_w, overlay_h)
        
        # Ensure position is within bounds
        x = max(0, min(x, main_w - overlay_w))
        y = max(0, min(y, main_h - overlay_h))
        
        # Process in chunks to save memory
        chunk_size = self.batch_size
        result_frames = main_frames.clone()  # Keep on CPU
        
        print(f"Processing overlay in chunks of {chunk_size} frames")
        
        # Move overlay frames to device once
        overlay_frames = overlay_frames.to(self.device)
        
        with tqdm(total=end_frame-start_frame, desc="Applying overlay (chunked)") as pbar:
            for chunk_start in range(start_frame, end_frame, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end_frame)
                
                # Move chunk to device
                chunk_frames = result_frames[chunk_start:chunk_end].to(self.device)
                
                # Apply overlay to chunk
                for frame_idx in range(chunk_start, chunk_end):
                    overlay_idx = (frame_idx - start_frame) % overlay_count
                    chunk_frames[frame_idx - chunk_start, :, y:y+overlay_h, x:x+overlay_w] = overlay_frames[overlay_idx]
                
                # Move result back to CPU
                result_frames[chunk_start:chunk_end] = chunk_frames.cpu()
                
                # Clear GPU memory
                del chunk_frames
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                pbar.update(chunk_end - chunk_start)
        
        return result_frames
        
        with tqdm(total=end_frame-start_frame, desc="Overlaying frames") as pbar:
            for main_idx in range(start_frame, end_frame):
                if main_idx >= main_count:
                    break
                
                # Calculate corresponding overlay frame (with looping if needed)
                relative_time = (main_idx - start_frame) / main_fps
                overlay_idx = int((relative_time % overlay_frame_duration) * main_fps) % overlay_count
                
                # Apply overlay
                if overlay_idx < overlay_count:
                    result_frames[main_idx, :, y:y+overlay_h, x:x+overlay_w] = \
                        overlay_frames[overlay_idx, :, :overlay_h, :overlay_w]
                
                pbar.update(1)
        
        return result_frames
    
    def calculate_center_position(self, position_spec, overlay_size, main_size):
        """Calculate position based on center coordinates"""
        overlay_w, overlay_h = overlay_size
        main_w, main_h = main_size
        
        # Handle preset positions
        preset_positions = {
            'top-left': (overlay_w // 2, overlay_h // 2),
            'top-center': (main_w // 2, overlay_h // 2),
            'top-right': (main_w - overlay_w // 2, overlay_h // 2),
            'center-left': (overlay_w // 2, main_h // 2),
            'center': (main_w // 2, main_h // 2),
            'center-right': (main_w - overlay_w // 2, main_h // 2),
            'bottom-left': (overlay_w // 2, main_h - overlay_h // 2),
            'bottom-center': (main_w // 2, main_h - overlay_h // 2),
            'bottom-right': (main_w - overlay_w // 2, main_h - overlay_h // 2)
        }
        
        if position_spec in preset_positions:
            center_x, center_y = preset_positions[position_spec]
        elif ',' in position_spec:
            x_str, y_str = position_spec.split(',')
            
            # Parse X coordinate (center position)
            if '%' in x_str:
                center_x = int((float(x_str.replace('%', '')) / 100) * main_w)
            else:
                center_x = int(x_str)
            
            # Parse Y coordinate (center position)
            if '%' in y_str:
                center_y = int((float(y_str.replace('%', '')) / 100) * main_h)
            else:
                center_y = int(y_str)
        else:
            # Default to center
            center_x, center_y = main_w // 2, main_h // 2
        
        # Convert center coordinates to top-left coordinates
        x = center_x - overlay_w // 2
        y = center_y - overlay_h // 2
        
        return x, y
    
    def calculate_size(self, size_spec, main_width, main_height):
        """Calculate target size based on size specification"""
        if '%' in size_spec:
            percent = float(size_spec.replace('%', ''))
            width = int((percent / 100) * main_width)
            # Maintain aspect ratio - height will be calculated during resize
            return width, -1
        elif 'x' in size_spec:
            width_str, height_str = size_spec.split('x')
            return int(width_str), int(height_str)
        else:
            try:
                percent = float(size_spec)
                width = int((percent / 100) * main_width)
                return width, -1
            except ValueError:
                # Default size
                return int(0.25 * main_width), -1
    
    def save_frames_as_video(self, frames_tensor, output_path, fps=25.0, audio_path=None):
        """Save frames tensor as video file"""
        if frames_tensor is None:
            print("Error: No frames to save")
            return False
        
        frames_tensor = frames_tensor.cpu()
        frame_count, _, height, width = frames_tensor.shape
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Saving {frame_count} frames to {output_path}")
        
        with tqdm(total=frame_count, desc="Saving video") as pbar:
            for i in range(frame_count):
                # Convert tensor to numpy array
                frame = frames_tensor[i].detach()
                frame_pil = self.inverse_transform(frame)
                frame_np = np.array(frame_pil)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(frame_bgr)
                pbar.update(1)
        
        out.release()
        
        # Add audio if specified
        if audio_path and os.path.exists(audio_path):
            self.add_audio_to_video(output_path, audio_path)
        
        return True
    
    def add_audio_to_video(self, video_path, audio_path):
        """Add audio to video using OpenCV (basic implementation)"""
        print(f"Note: Audio addition requires FFmpeg. Video saved without audio: {video_path}")
        print(f"To add audio manually, use: ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac output_with_audio.mp4")
        return video_path

class ManualVideoInserter:
    """Main class for manual video insertion using PyTorch with efficient segment processing"""
    
    def __init__(self, temp_dir=None, use_gpu=True, batch_size=10, memory_efficient=True, use_segment_processing=True):
        """
        Initialize the manual inserter with memory optimization for large videos
        
        Args:
            temp_dir (str): Directory for temporary files (optional)
            use_gpu (bool): Whether to use GPU acceleration
            batch_size (int): Batch size for processing (reduced for memory efficiency)
            memory_efficient (bool): Enable memory-efficient processing for large videos
            use_segment_processing (bool): Use segment-based processing (recommended for large videos)
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.temp_files = []  # Track temporary files for cleanup
        
        # Use segment processing and smaller batch size by default for large videos
        self.processor = PyTorchVideoProcessor(
            use_gpu=use_gpu, 
            batch_size=batch_size, 
            memory_efficient=memory_efficient,
            use_segment_processing=use_segment_processing
        )
        
        print(f"Initialized ManualVideoInserter with temp directory: {self.temp_dir}")
        print(f"Memory efficient mode: {memory_efficient}")
        print(f"Segment processing: {use_segment_processing}")
        
        # Set environment variable for better memory management
        if use_gpu:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    def monitor_gpu_memory(self, operation_name=""):
        """Monitor and log GPU memory usage"""
        if self.processor.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory {operation_name}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.processor.device.type == 'cuda':
            torch.cuda.empty_cache()
            self.monitor_gpu_memory("after cleanup")
    
    def parse_time_to_seconds(self, time_str):
        """
        Parse time string to seconds
        
        Args:
            time_str (str): Time in MM:SS.ms or HH:MM:SS.ms format
            
        Returns:
            float: Time in seconds
        """
        try:
            if time_str.count(':') == 1:
                # MM:SS.ms format
                minutes, seconds_ms = time_str.split(':')
                minutes = int(minutes)
                
                if '.' in seconds_ms:
                    seconds, ms = seconds_ms.split('.')
                    seconds = int(seconds)
                    ms = int(ms)
                else:
                    seconds = int(seconds_ms)
                    ms = 0
                
                return minutes * 60 + seconds + ms / 100.0
            
            elif time_str.count(':') == 2:
                # HH:MM:SS.ms format
                hours, minutes, seconds_ms = time_str.split(':')
                hours = int(hours)
                minutes = int(minutes)
                
                if '.' in seconds_ms:
                    seconds, ms = seconds_ms.split('.')
                    seconds = int(seconds)
                    ms = int(ms)
                else:
                    seconds = int(seconds_ms)
                    ms = 0
                
                return hours * 3600 + minutes * 60 + seconds + ms / 100.0
            
            else:
                # Try to parse as pure seconds
                return float(time_str)
                
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse time '{time_str}': {e}")
            return 0.0
    
    def _process_audio_insertions(self, video_path, audio_inserts):
        """处理音频插入操作"""
        import subprocess
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        current_video = video_path
        
        try:
            for i, audio_insert in enumerate(audio_inserts):
                source = audio_insert.get('source', '')
                insert_time = audio_insert.get('time', '00:00.00')
                audio_start = audio_insert.get('start', '00:00.00')
                duration = audio_insert.get('duration', '00:05.00')
                volume = audio_insert.get('volume', 1.0)
                mix_mode = audio_insert.get('mix_mode', 'overlay')
                
                if not os.path.exists(source):
                    print(f"⚠️ Audio file not found: {source}")
                    continue
                
                # 准备输出文件
                temp_output = os.path.join(temp_dir, f"temp_audio_{i}.mp4")
                
                # 转换时间格式
                insert_seconds = self.parse_time_to_seconds(insert_time)
                audio_start_seconds = self.parse_time_to_seconds(audio_start)
                duration_seconds = self.parse_time_to_seconds(duration)
                
                print(f"   🎵 Adding audio: {os.path.basename(source)} at {insert_time} (mode: {mix_mode})")
                
                if mix_mode == 'replace':
                    # Replace mode: Mute original audio during the specified duration and add new audio
                    end_seconds = insert_seconds + duration_seconds
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', current_video,
                        '-i', source,
                        '-filter_complex',
                        f'[0:a]volume=enable=\'between(t,{insert_seconds},{end_seconds})\':volume=0[muted];'
                        f'[1:a]atrim=start={audio_start_seconds}:duration={duration_seconds},volume={volume}[trimmed];'
                        f'[trimmed]adelay={insert_seconds*1000}|{insert_seconds*1000}[delayed];'
                        f'[muted][delayed]amix=inputs=2:duration=longest[mixed]',
                        '-map', '0:v', '-map', '[mixed]',
                        '-c:v', 'copy', '-c:a', 'aac',
                        temp_output
                    ]
                else:
                    # Overlay mode: mix with original audio
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', current_video,
                        '-i', source,
                        '-filter_complex',
                        f'[1:a]atrim=start={audio_start_seconds}:duration={duration_seconds}[trimmed];'
                        f'[trimmed]adelay={insert_seconds*1000}|{insert_seconds*1000},volume={volume}[delayed];'
                        f'[0:a][delayed]amix=inputs=2:duration=longest[mixed]',
                        '-map', '0:v', '-map', '[mixed]',
                        '-c:v', 'copy', '-c:a', 'aac',
                        temp_output
                    ]
                
                # 执行FFmpeg命令
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    current_video = temp_output
                    print(f"   ✅ Audio processed successfully")
                else:
                    print(f"   ❌ Audio processing failed: {result.stderr}")
                    continue
            
            # 复制最终结果到目标位置
            if current_video != video_path:
                import shutil
                shutil.copy2(current_video, video_path)
                print(f"✅ Final video with audio saved to: {video_path}")
            
            return video_path
            
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            return video_path
        finally:
            # 清理临时文件
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

    def seconds_to_time_str(self, seconds):
        """
        Convert seconds to MM:SS.ms format
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Time in MM:SS.ms format
        """
        total_seconds = int(seconds)
        ms = int((seconds - total_seconds) * 100)
        minutes = total_seconds // 60
        seconds_remainder = total_seconds % 60
        
        return f"{minutes:02d}:{seconds_remainder:02d}.{ms:02d}"
    
    def load_config(self, config_path):
        """
        Load configuration from YAML file with processing settings support
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration data
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                print(f"Loaded configuration from {config_path}")
                
                # 處理可選的處理設置
                processing_settings = config.get('processing_settings', {})
                if processing_settings:
                    print("Found processing settings in config:")
                    
                    # 更新處理器設置
                    if 'memory_efficient' in processing_settings:
                        self.processor.memory_efficient = processing_settings['memory_efficient']
                        print(f"  Memory efficient: {self.processor.memory_efficient}")
                    
                    if 'use_segment_processing' in processing_settings:
                        self.processor.use_segment_processing = processing_settings['use_segment_processing']
                        print(f"  Segment processing: {self.processor.use_segment_processing}")
                    
                    if 'batch_size' in processing_settings:
                        self.processor.batch_size = processing_settings['batch_size']
                        print(f"  Batch size: {self.processor.batch_size}")
                
                return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}
    
    def show_recommendations(self, matches_json_path):
        """
        Display recommendations from matches.json file
        
        Args:
            matches_json_path (str): Path to matches.json file
        """
        try:
            with open(matches_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("推薦插入點分析")
            print("================\n")
            
            # Handle both old and new JSON formats
            if isinstance(data, dict) and 'sentence_groups' in data:
                # New format with sentence_groups wrapper
                sentence_groups = data['sentence_groups']
            elif isinstance(data, list):
                # Old format - direct array, convert to new format structure
                sentence_groups = []
                for i, group in enumerate(data, 1):
                    converted_group = {
                        'group_id': i,
                        'start_time': None,
                        'end_time': None,
                        'duration': None,
                        'sentences': group.get('sentence_group', {}).get('sentences', []),
                        'recommended_videos': []
                    }
                    
                    # Convert matching_videos to recommended_videos format
                    for video in group.get('matching_videos', []):
                        converted_video = {
                            'video_id': video.get('video_id', 'Unknown'),
                            'distance': video.get('distance', 0),
                            'similarity_score': round(1.0 - video.get('distance', 0), 4),
                            'tags': video.get('tags', '').split(', ') if video.get('tags') else [],
                            'description': video.get('title', f"Video {video.get('video_id', 'Unknown')}")
                        }
                        converted_group['recommended_videos'].append(converted_video)
                    
                    sentence_groups.append(converted_group)
            else:
                print("Error: Unsupported JSON format")
                return
            
            for group in sentence_groups:
                group_id = group.get('group_id', 'Unknown')
                start_time = group.get('start_time')
                end_time = group.get('end_time')
                duration = group.get('duration')
                
                if start_time and end_time:
                    print(f"句子群組 {group_id} (時間: {start_time} - {end_time}, 時長: {duration}):")
                else:
                    print(f"句子群組 {group_id} (時間: None - None, 時長: None):")
                
                print("句子內容:")
                for sentence in group.get('sentences', []):
                    print(f"- {sentence}")
                
                print("\n推薦影片:")
                for i, video in enumerate(group.get('recommended_videos', []), 1):
                    video_id = video.get('video_id', 'Unknown')
                    distance = video.get('distance', 0)
                    tags = video.get('tags', [])
                    description = video.get('description', 'No description available')
                    
                    print(f"{i}. {video_id} (distance: {distance:.4f})")
                    if tags:
                        tags_str = ', '.join(tags) if isinstance(tags, list) else str(tags)
                        print(f"   標籤: {tags_str}")
                
                if duration:
                    print(f"\n可用時長: {duration}")
                else:
                    print(f"\n可用時長: None")
                
                print("\n")
        
        except FileNotFoundError:
            print(f"Error: File {matches_json_path} not found")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {matches_json_path}")
        except Exception as e:
            print(f"Error reading recommendations: {e}")
    
    def create_temp_file(self, suffix=".mp4"):
        """
        Create a temporary file and track it for cleanup
        
        Args:
            suffix (str): File suffix
            
        Returns:
            str: Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, dir=self.temp_dir, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        self.temp_files.append(temp_path)
        return temp_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_file}: {e}")
        
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory {self.temp_dir}: {e}")
    
    def get_video_info(self, video_path):
        """
        Get video information using OpenCV
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            dict: Video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return {}
            
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            cap.release()
            
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps
            }
                
        except Exception as e:
            print(f"Error getting video info for {video_path}: {e}")
            return {}
    
    def process_video_inserts(self, config, input_video_path, output_path):
        """
        Process video insertions with intelligent processing strategy
        
        Args:
            config (dict): Configuration data
            input_video_path (str): Path to main input video
            output_path (str): Path for output video
            
        Returns:
            str: Path to video with inserted clips
        """
        video_inserts = config.get('video_inserts', [])
        if not video_inserts:
            print("No video insertions configured")
            return input_video_path
        
        print(f"Processing {len(video_inserts)} video insertion groups...")
        self.monitor_gpu_memory("before processing")
        
        # 準備覆蓋配置列表
        overlay_configs = []
        
        for i, insert_group in enumerate(video_inserts):
            insert_time = insert_group.get('time', '00:00.00')
            videos = insert_group.get('videos', [])
            
            if not videos:
                print(f"Warning: No videos specified for insertion at {insert_time}")
                continue
            
            insert_seconds = self.parse_time_to_seconds(insert_time)
            
            # 處理每個視頻片段
            for j, video_config in enumerate(videos):
                source = video_config.get('source', '')
                start = video_config.get('start', '00:00.00')
                duration = video_config.get('duration', '00:03.00')
                position = video_config.get('position', 'top-right')
                size = video_config.get('size', '25%')
                
                if not os.path.exists(source):
                    print(f"Warning: Source video {source} not found, skipping")
                    continue
                
                start_seconds = self.parse_time_to_seconds(start)
                duration_seconds = self.parse_time_to_seconds(duration)
                
                # 添加到配置列表
                overlay_configs.append({
                    'source': source,
                    'start_time': insert_seconds,
                    'duration': duration_seconds,
                    'clip_start': start_seconds,
                    'position': position,
                    'size': size
                })
                
                print(f"Configured overlay: {source} at {insert_time} for {duration}")
        
        if not overlay_configs:
            print("No valid overlay configurations found")
            return input_video_path
        
        # 使用高效的片段處理
        success = self.processor.process_video_efficiently(
            input_video_path, overlay_configs, output_path
        )
        
        # 清理記憶體
        self.clear_gpu_memory()
        
        if success:
            print(f"✅ Video processing completed! Output: {output_path}")
            return output_path
        else:
            print("❌ Error during video processing")
            return input_video_path
    
    def process_video_with_config(self, input_video_path, output_path, config):
        """
        Process all insertions with an existing configuration object
        
        Args:
            input_video_path (str): Path to input video
            output_path (str): Path for output video
            config (dict): Configuration data
        """
        if not config:
            print("Error: Invalid configuration")
            return
        
        try:
            # Process video insertions
            video_output = self.process_video_inserts(config, input_video_path, output_path)
            
            # Process audio insertions
            audio_inserts = config.get('audio_inserts', [])
            if audio_inserts:
                print("🎵 Processing audio insertions...")
                video_output = self._process_audio_insertions(video_output, audio_inserts)
                print("✅ Audio processing completed!")
            
            print(f"All video insertions completed. Final video: {video_output}")
            
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()
    
    def process_insertions(self, config_path, input_video_path, output_path):
        """
        Process all insertions according to configuration file
        
        Args:
            config_path (str): Path to configuration file
            input_video_path (str): Path to input video
            output_path (str): Path for output video
        """
        config = self.load_config(config_path)
        if not config:
            print("Error: Could not load configuration")
            return
        
        try:
            # Process video insertions (audio processing removed for PyTorch version)
            video_output = self.process_video_inserts(config, input_video_path, output_path)
            
            # Process audio insertions
            audio_inserts = config.get('audio_inserts', [])
            if audio_inserts:
                print("🎵 Processing audio insertions...")
                video_output = self._process_audio_insertions(video_output, audio_inserts)
                print("✅ Audio processing completed!")
            
            print(f"All video insertions completed. Final video: {video_output}")
            
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()

def main():
    parser = argparse.ArgumentParser(description='Manual Video and Audio Inserter')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--input-video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path for output video file')
    parser.add_argument('--show-recommendations', type=str, help='Show recommendations from matches.json file')
    parser.add_argument('--temp-dir', type=str, help='Directory for temporary files')
    
    # Command line video insertion parameters
    parser.add_argument('--add-video', type=str, help='Path to video file to insert')
    parser.add_argument('--insert-time', type=str, help='Time to insert video in main video (MM:SS.ms format)')
    parser.add_argument('--clip-start', type=str, default='00:00.00', help='Start time in source video (default: 00:00.00)')
    parser.add_argument('--clip-duration', type=str, default='00:03.00', help='Duration to insert (default: 00:03.00)')
    parser.add_argument('--position', type=str, default='top-right', help='Position on screen (default: top-right)')
    parser.add_argument('--size', type=str, default='25%', help='Size as percentage or pixels (default: 25%)')
    parser.add_argument('--loop', action='store_true', help='Loop video if source is shorter than duration')
    
    # Command line audio insertion parameters
    parser.add_argument('--add-audio', type=str, help='Path to audio file to insert')
    parser.add_argument('--audio-time', type=str, help='Time to start audio in main video (MM:SS.ms format)')
    parser.add_argument('--audio-start', type=str, default='00:00.00', help='Start time in source audio (default: 00:00.00)')
    parser.add_argument('--audio-duration', type=str, default='00:05.00', help='Audio duration to insert (default: 00:05.00)')
    parser.add_argument('--volume', type=float, default=0.8, help='Audio volume level (0.0-1.0+, default: 0.8)')
    parser.add_argument('--mix-mode', type=str, choices=['overlay', 'replace'], default='overlay', help='Audio mix mode (default: overlay)')
    
    args = parser.parse_args()
    
    # Initialize inserter with efficient segment processing for large videos
    inserter = ManualVideoInserter(
        temp_dir=args.temp_dir,
        use_gpu=True,
        batch_size=5,  # Small batch size for 4K videos
        memory_efficient=True,
        use_segment_processing=True  # Enable efficient segment processing
    )
    
    try:
        if args.show_recommendations:
            # Show recommendations mode
            inserter.show_recommendations(args.show_recommendations)
        
        elif args.config:
            # Process insertions using configuration file only (self-contained mode)
            if not os.path.exists(args.config):
                print(f"Error: Configuration file {args.config} not found")
                return 1
            
            # Load configuration
            config = inserter.load_config(args.config)
            
            # Check for self-contained configuration
            if 'input_video' in config and 'output_video' in config:
                # Self-contained YAML mode
                input_video = config['input_video']
                output_video = config['output_video']
                
                if not os.path.exists(input_video):
                    print(f"Error: Input video {input_video} (from config) not found")
                    return 1
                
                print(f"🎯 Self-contained YAML mode:")
                print(f"   Input: {input_video}")
                print(f"   Output: {output_video}")
                
                inserter.process_video_with_config(input_video, output_video, config)
                
            elif args.input_video and args.output:
                # Legacy mode with command line input/output
                if not os.path.exists(args.input_video):
                    print(f"Error: Input video {args.input_video} not found")
                    return 1
                
                print(f"📋 Legacy YAML mode:")
                print(f"   Input: {args.input_video}")
                print(f"   Output: {args.output}")
                
                config = inserter.load_config(args.config)
                inserter.process_video_with_config(args.input_video, args.output, config)
                
            else:
                print("Error: Configuration file must either:")
                print("  1. Include 'input_video' and 'output_video' fields (self-contained mode), OR")
                print("  2. Be used with --input-video and --output arguments (legacy mode)")
                return 1
        
        elif args.input_video and args.output and (args.add_video or args.add_audio):
            # Command line insertion mode
            if not os.path.exists(args.input_video):
                print(f"Error: Input video {args.input_video} not found")
                return 1
            
            # Create temporary configuration for command line parameters
            temp_config = {}
            
            # Add video insertion if specified
            if args.add_video:
                if not args.insert_time:
                    print("Error: --insert-time is required when using --add-video")
                    return 1
                
                if not os.path.exists(args.add_video):
                    print(f"Error: Video file {args.add_video} not found")
                    return 1
                
                temp_config['video_inserts'] = [{
                    'time': args.insert_time,
                    'videos': [{
                        'source': args.add_video,
                        'start': args.clip_start,
                        'duration': args.clip_duration,
                        'position': args.position,
                        'size': args.size,
                        'loop': args.loop
                    }]
                }]
                print(f"Adding video: {args.add_video} at {args.insert_time} for {args.clip_duration}")
            
            # Add audio insertion if specified
            if args.add_audio:
                if not args.audio_time:
                    print("Error: --audio-time is required when using --add-audio")
                    return 1
                
                if not os.path.exists(args.add_audio):
                    print(f"Error: Audio file {args.add_audio} not found")
                    return 1
                
                temp_config['audio_inserts'] = [{
                    'time': args.audio_time,
                    'source': args.add_audio,
                    'start': args.audio_start,
                    'duration': args.audio_duration,
                    'volume': args.volume,
                    'mix_mode': args.mix_mode
                }]
                print(f"Adding audio: {args.add_audio} at {args.audio_time} for {args.audio_duration}")
            
            # Save temporary configuration and process
            temp_config_path = inserter.create_temp_file('.yaml')
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(temp_config, f, default_flow_style=False, allow_unicode=True)
            
            inserter.process_insertions(temp_config_path, args.input_video, args.output)
        
        else:
            print("Error: Invalid arguments provided")
            print("\nUsage modes:")
            print("1. Show recommendations: --show-recommendations <matches.json>")
            print("2. Use config file: --config <config.yaml> --input-video <video> --output <output>")
            print("3. Command line insertion: --input-video <video> --output <output> [--add-video <video> --insert-time <time>] [--add-audio <audio> --audio-time <time>]")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        inserter.cleanup_temp_files()
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        inserter.cleanup_temp_files()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
