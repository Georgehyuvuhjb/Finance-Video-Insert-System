#!/usr/bin/env python3
"""
FFmpeg Basic Caption Generator - 最兼容版本
使用最基本的方法添加字幕
"""
import subprocess
import tempfile
import os
import sys

def create_simple_srt(script_file, output_srt):
    """創建簡單的SRT文件"""
    sys.path.insert(0, '/home/22055747d/insert_2.2')
    from auto_caption.subtitle_parser import SubtitleParser
    
    parser = SubtitleParser()
    segments = parser.parse_script_file(script_file)
    
    with open(output_srt, 'w', encoding='utf-8') as f:
        def format_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
        
        for i, segment in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(segment.start_time)} --> {format_srt_time(segment.end_time)}\n")
            f.write(f"{segment.text}\n\n")
    
    print(f"Created SRT: {output_srt}")
    return len(segments)

def main():
    input_video = '/home/22055747d/insert_2.2/outputs/video_merge/final_video_pytorch.mp4'
    script_file = '/home/22055747d/insert_2.2/outputs/tts/script/script.txt'
    output_video = '/home/22055747d/insert_2.2/outputs/auto_caption/ffmpeg_basic.mp4'
    
    # 創建SRT文件
    srt_file = '/tmp/basic_subtitles.srt'
    num_segments = create_simple_srt(script_file, srt_file)
    print(f"Created {num_segments} subtitle segments")
    
    # 檢查SRT文件內容
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"SRT file size: {len(content)} characters")
        print("SRT preview:")
        print(content[:300] + "...")
    
    # 嘗試不同的FFmpeg命令
    commands_to_try = [
        # 方法1: 使用filename參數
        [
            'ffmpeg', '-y',
            '-i', input_video,
            '-vf', f'subtitles=filename={srt_file}',
            '-c:v', 'mpeg4', '-c:a', 'copy',
            output_video.replace('.mp4', '_method1.mp4')
        ],
        # 方法2: 使用簡單路徑
        [
            'ffmpeg', '-y',
            '-i', input_video,
            '-vf', f'subtitles={srt_file}',
            '-c:v', 'mpeg4', '-c:a', 'copy',
            output_video.replace('.mp4', '_method2.mp4')
        ],
        # 方法3: 使用轉義路徑
        [
            'ffmpeg', '-y',
            '-i', input_video,
            '-vf', f'subtitles={srt_file.replace(":", "\\:")}',
            '-c:v', 'mpeg4', '-c:a', 'copy',
            output_video.replace('.mp4', '_method3.mp4')
        ],
        # 方法4: 不使用濾鏡，直接燒錄（如果支持）
        [
            'ffmpeg', '-y',
            '-i', input_video,
            '-i', srt_file,
            '-c:v', 'mpeg4', '-c:a', 'copy',
            '-c:s', 'mov_text',
            output_video.replace('.mp4', '_method4.mp4')
        ]
    ]
    
    success = False
    
    for i, cmd in enumerate(commands_to_try, 1):
        print(f"\n=== 嘗試方法 {i} ===")
        print("命令:")
        print(" ".join(cmd))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            print(f"返回碼: {result.returncode}")
            if result.stdout:
                print("stdout:", result.stdout[:200])
            if result.stderr:
                print("stderr:", result.stderr[:500])
            
            if result.returncode == 0:
                output_file = cmd[-1]
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"✅ 成功！輸出文件: {output_file}")
                    print(f"文件大小: {file_size:,} 字節 ({file_size/1024/1024:.1f} MB)")
                    success = True
                    break
            else:
                print(f"❌ 方法 {i} 失敗")
        
        except subprocess.TimeoutExpired:
            print(f"❌ 方法 {i} 超時")
        except Exception as e:
            print(f"❌ 方法 {i} 錯誤: {e}")
    
    # 清理
    if os.path.exists(srt_file):
        os.unlink(srt_file)
        print(f"清理SRT文件: {srt_file}")
    
    if success:
        print("\n🎉 至少有一種方法成功了！")
        return 0
    else:
        print("\n😞 所有方法都失敗了")
        print("可能需要使用OpenCV版本或安裝支持字幕的FFmpeg")
        return 1

if __name__ == "__main__":
    sys.exit(main())
