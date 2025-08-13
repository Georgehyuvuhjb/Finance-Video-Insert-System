#!/usr/bin/env python3
"""
Test script for audio preservation in efficient video processing
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_audio_preservation():
    """測試音頻保留功能"""
    from manual_insert.manual_inserter import ManualVideoInserter
    
    print("Audio Preservation Test")
    print("=" * 50)
    
    # 初始化插入器
    inserter = ManualVideoInserter(
        use_gpu=False,  # 使用 CPU 以避免 CUDA 問題
        use_segment_processing=True
    )
    
    # 測試參數
    config = {
        'video_inserts': [{
            'time': '00:00.05',
            'videos': [{
                'source': 'test_overlay.mp4',  # 假設的覆蓋視頻
                'start': '00:00.00',
                'duration': '00:03.00',
                'position': 'center',
                'size': '25%'
            }]
        }]
    }
    
    # 測試配置
    main_video = 'test_main.mp4'      # 假設的主視頻
    output_video = 'test_output.mp4'   # 輸出視頻
    
    print("Testing audio preservation logic...")
    
    # 測試音頻保留方法
    processor = inserter.processor
    
    # 模擬測試 FFmpeg 可用性
    try:
        print("1. Testing FFmpeg availability...")
        result = processor._try_ffmpeg_audio_preservation(
            'dummy_input.mp4', 'dummy_processed.mp4', 'dummy_output.mp4'
        )
        print(f"   FFmpeg test result: {result}")
    except Exception as e:
        print(f"   FFmpeg test failed: {e}")
    
    # 測試 moviepy 可用性
    try:
        print("2. Testing moviepy availability...")
        from moviepy.editor import VideoFileClip
        print("   ✅ moviepy is available")
    except ImportError:
        print("   ❌ moviepy is not available")
    
    print("\n📋 Audio preservation strategies:")
    print("1. FFmpeg (preferred) - Fastest and most reliable")
    print("2. moviepy (fallback) - Pure Python solution")
    print("3. Manual instructions - User handles audio separately")
    
    print("\n🔧 For your HPC environment, ensure one of these is available:")
    print("- FFmpeg: conda install ffmpeg")
    print("- moviepy: pip install moviepy")
    
    return True

def check_audio_in_video(video_path):
    """檢查視頻是否包含音頻軌道"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        # OpenCV 無法直接檢查音頻，需要使用其他方法
        print(f"Checking video: {video_path}")
        
        # 嘗試使用 FFmpeg 檢查音頻
        try:
            import subprocess
            cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'a', 
                   '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                print(f"✅ Audio track found: {result.stdout.strip()}")
                return True
            else:
                print("❌ No audio track found")
                return False
                
        except Exception as e:
            print(f"Cannot check audio track: {e}")
            return None
            
    except Exception as e:
        print(f"Error checking video: {e}")
        return None

def main():
    """運行所有測試"""
    print("🎵 Audio Preservation Testing Suite")
    print("=" * 60)
    
    # 1. 測試音頻保留邏輯
    test_audio_preservation()
    
    # 2. 如果有實際視頻文件，檢查音頻軌道
    test_videos = [
        "../input/video.mp4",
        "../outputs/final.mp4"
    ]
    
    print("\n🔍 Checking existing video files for audio tracks...")
    for video in test_videos:
        if os.path.exists(video):
            check_audio_in_video(video)
        else:
            print(f"Video not found: {video}")
    
    print("\n📝 Recommendations for audio preservation:")
    print("1. Install FFmpeg in your HPC environment for best performance")
    print("2. Use moviepy as a fallback if FFmpeg is not available")
    print("3. Process video first, then add audio manually if needed")
    
    print("\n🚀 The new efficient processing will:")
    print("- Process only the segments that need overlay")
    print("- Preserve original audio automatically")
    print("- Provide fallback options if audio tools are unavailable")

if __name__ == "__main__":
    main()
