#!/usr/bin/env python3
"""
FFmpeg-based Caption Generator
==============================

High-performance subtitle generator using FFmpeg for video processing.
Alternative implementation for environments where FFmpeg GPU acceleration is available.

Features:
- FFmpeg-based video processing
- SRT subtitle generation
- Hardware acceleration support
- High-quality output
- Automatic subtitle styling
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

# Import our subtitle parser
from subtitle_parser import SubtitleParser, SubtitleSegment


class FFmpegCaptionGenerator:
    """FFmpeg-based caption generator"""

    def __init__(self,
                 input_video: str,
                 output_video: str,
                 subtitle_segments: List[SubtitleSegment],
                 font_size: int = 24,
                 font_color: str = "white",
                 position: str = "bottom",
                 margin_bottom: int = 50,
                 max_width: int = 80,
                 line_spacing: float = 1.2,
                 quality: str = "high",
                 use_gpu: bool = True):

        self.input_video = input_video
        self.output_video = output_video
        self.subtitle_segments = subtitle_segments
        self.font_size = font_size
        self.font_color = font_color
        self.position = position
        self.margin_bottom = margin_bottom
        self.max_width = max_width
        self.line_spacing = line_spacing
        self.quality = quality
        self.use_gpu = use_gpu

        # Check FFmpeg availability
        self.ffmpeg_available = self.check_ffmpeg()
        self.available_encoders = self.check_available_encoders()

        # Temporary files
        self.temp_srt_file = None

    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("FFmpeg is available")
                return True
            else:
                print("FFmpeg not found in system PATH")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("FFmpeg not found or not accessible")
            return False

    def check_available_encoders(self) -> List[str]:
        """Check available video encoders"""
        if not self.ffmpeg_available:
            return []

        available_encoders = []
        encoders_to_check = ['libx264', 'libopenh264',
                             'mpeg4', 'libxvid', 'libvpx']

        try:
            result = subprocess.run(['ffmpeg', '-encoders'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout
                for encoder in encoders_to_check:
                    if encoder in output:
                        available_encoders.append(encoder)
                        print(f"Video encoder available: {encoder}")
        except subprocess.TimeoutExpired:
            print("Timeout checking video encoders")

        if not available_encoders:
            print("No video encoders found, will try defaults")

        return available_encoders

    def wrap_text(self, text: str, max_width_chars: int) -> List[str]:
        """
        Wrap text to multiple lines based on character count

        Args:
            text: Input text
            max_width_chars: Maximum characters per line

        Returns:
            List of text lines
        """
        if len(text) <= max_width_chars:
            return [text]

        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_width_chars:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word longer than max width, force break
                    lines.append(word[:max_width_chars])
                    current_line = word[max_width_chars:]

        if current_line:
            lines.append(current_line)

        return lines

    def create_srt_file(self) -> str:
        """
        Create SRT subtitle file from segments

        Returns:
            Path to created SRT file
        """
        # Create temporary SRT file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.srt',
                                                delete=False, encoding='utf-8')
        self.temp_srt_file = temp_file.name

        def format_srt_time(seconds: float) -> str:
            """Format seconds to SRT time format (HH:MM:SS,mmm)"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

        # Write SRT content
        for i, segment in enumerate(self.subtitle_segments, 1):
            # Wrap text if needed
            wrapped_lines = self.wrap_text(segment.text, self.max_width)
            wrapped_text = '\n'.join(wrapped_lines)

            # Write SRT entry
            temp_file.write(f"{i}\n")
            temp_file.write(
                f"{format_srt_time(segment.start_time)} --> {format_srt_time(segment.end_time)}\n")
            temp_file.write(f"{wrapped_text}\n\n")

        temp_file.close()

        # Verify SRT file was created correctly
        if os.path.exists(self.temp_srt_file):
            with open(self.temp_srt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Created SRT file: {self.temp_srt_file}")
                print(f"SRT file size: {len(content)} characters")
                # Show first few lines for debugging
                lines = content.split('\n')[:10]
                print("SRT file preview:")
                for line in lines:
                    print(f"  {line}")
        else:
            print(f"Warning: SRT file creation failed")

        return self.temp_srt_file

    def get_video_info(self) -> dict:
        """Get video information using FFprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', self.input_video
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)

                # Find video stream
                video_stream = None
                for stream in info['streams']:
                    if stream['codec_type'] == 'video':
                        video_stream = stream
                        break

                if video_stream:
                    return {
                        'width': int(video_stream.get('width', 1920)),
                        'height': int(video_stream.get('height', 1080)),
                        'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                        'duration': float(info['format'].get('duration', 0))
                    }

        except Exception as e:
            print(f"Error getting video info: {e}")

        # Default values if probe fails
        return {'width': 1920, 'height': 1080, 'fps': 30, 'duration': 0}

    def build_ffmpeg_command(self, srt_file: str) -> List[str]:
        """
        Build FFmpeg command for adding subtitles

        Args:
            srt_file: Path to SRT subtitle file

        Returns:
            FFmpeg command as list of strings
        """
        # Base command - most basic approach
        cmd = ['ffmpeg', '-y', '-i', self.input_video]

        # Ultra-simple subtitle filter - escape the path properly
        srt_path = srt_file.replace(
            '\\', '/').replace(':', '\\:')  # Escape for FFmpeg
        cmd.extend(['-vf', f'subtitles=filename={srt_path}'])

        # Simple encoding settings with fallback encoders
        if self.quality == "high":
            # Try best available encoder
            if 'libx264' in self.available_encoders:
                cmd.extend(['-c:v', 'libx264', '-crf', '18'])
            elif 'libopenh264' in self.available_encoders:
                # libopenh264 doesn't support crf 18
                cmd.extend(['-c:v', 'libopenh264', '-crf', '23'])
            elif 'mpeg4' in self.available_encoders:
                cmd.extend(['-c:v', 'mpeg4', '-qscale:v', '4'])
            elif 'libxvid' in self.available_encoders:
                cmd.extend(['-c:v', 'libxvid', '-qscale:v', '4'])
            else:
                # Fallback to default
                cmd.extend(['-c:v', 'mpeg4', '-qscale:v', '4'])
        else:
            # Normal quality
            if 'libx264' in self.available_encoders:
                cmd.extend(['-c:v', 'libx264', '-crf', '23'])
            elif 'libopenh264' in self.available_encoders:
                cmd.extend(['-c:v', 'libopenh264', '-crf', '28'])
            elif 'mpeg4' in self.available_encoders:
                cmd.extend(['-c:v', 'mpeg4', '-qscale:v', '6'])
            else:
                # Fallback to default
                cmd.extend(['-c:v', 'mpeg4', '-qscale:v', '6'])

        # Copy audio without re-encoding
        cmd.extend(['-c:a', 'copy'])

        # Output file - ensure it has proper extension
        output_file = self.output_video
        if not output_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            # Add .mp4 extension if none provided
            output_file += '.mp4'
            print(f"Adding .mp4 extension to output: {output_file}")
            self.output_video = output_file  # Update the instance variable

        cmd.append(output_file)

        return cmd

    # build_subtitle_filter method removed - using simple subtitles filter in build_ffmpeg_command instead

    def find_font_file(self) -> Optional[str]:
        """Find a suitable font file for subtitle rendering (for logging purposes only)"""
        font_paths = [
            # Linux fonts (common locations)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            # Windows fonts
            "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
            "C:/Windows/Fonts/simsun.ttc",  # SimSun
            "C:/Windows/Fonts/simhei.ttf",  # SimHei
            # macOS fonts
            "/System/Library/Fonts/PingFang.ttc",
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                print(f"Font available: {font_path}")
                return font_path

        print("No specific fonts found, using system default")
        return None

    def get_color_hex(self, color: str) -> str:
        """Get hex color code for FFmpeg"""
        color_codes = {
            'white': 'FFFFFF',
            'black': '000000',
            'yellow': 'FFFF00'
        }
        return color_codes.get(color, 'FFFFFF')

    def process_video(self):
        """Process video with FFmpeg"""
        if not self.ffmpeg_available:
            raise RuntimeError("FFmpeg is not available")

        print("Starting FFmpeg video processing...")

        try:
            # Create SRT file
            srt_file = self.create_srt_file()

            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_video)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Build FFmpeg command
            cmd = self.build_ffmpeg_command(srt_file)

            print("FFmpeg command:")
            print(" ".join(cmd))
            print("\nProcessing video...")

            # Run FFmpeg with better error handling
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Capture both stdout and stderr
            stdout, stderr = process.communicate()

            # Print all output for debugging
            if stdout:
                print("FFmpeg stdout:")
                print(stdout)

            if stderr:
                print("FFmpeg stderr:")
                print(stderr)

            # Check return code
            if process.returncode == 0:
                print(f"\nVideo processing completed successfully!")
                print(f"Output saved to: {self.output_video}")
                return True
            else:
                print(
                    f"\nFFmpeg failed with return code: {process.returncode}")
                print("This might be due to:")
                print("1. Unsupported subtitle filter options")
                print("2. Font-related issues")
                print("3. Input video format compatibility")
                print("4. Missing subtitle filter support in FFmpeg")
                return False

        except Exception as e:
            print(f"Error during FFmpeg processing: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup temporary files
            if self.temp_srt_file and os.path.exists(self.temp_srt_file):
                try:
                    os.unlink(self.temp_srt_file)
                    print(f"Cleaned up temporary SRT file")
                except Exception as e:
                    print(f"Warning: Could not delete temporary file: {e}")

        return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="FFmpeg-based caption generator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--input-video', required=True,
                        help='Input video file path')
    parser.add_argument('--script', required=True,
                        help='Script file with timestamps')

    # Optional arguments
    parser.add_argument('--output', default='outputs/auto_caption/final_video_ffmpeg.mp4',
                        help='Output video file path')
    parser.add_argument('--font-size', type=int, default=24,
                        help='Font size for subtitles')
    parser.add_argument('--font-color', default='white',
                        choices=['white', 'black', 'yellow'],
                        help='Font color for subtitles')
    parser.add_argument('--position', default='bottom',
                        choices=['bottom', 'top', 'center'],
                        help='Subtitle position')
    parser.add_argument('--margin-bottom', type=int, default=50,
                        help='Bottom margin in pixels')
    parser.add_argument('--max-width', type=int, default=80,
                        help='Maximum characters per line')
    parser.add_argument('--line-spacing', type=float, default=1.2,
                        help='Line spacing multiplier')
    parser.add_argument('--quality', choices=['normal', 'high'], default='high',
                        help='Output video quality')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return 1

    if not os.path.exists(args.script):
        print(f"Error: Script file not found: {args.script}")
        return 1

    # Parse subtitle file
    print("Parsing subtitle file...")
    subtitle_parser = SubtitleParser()
    try:
        segments = subtitle_parser.parse_script_file(args.script)
        if not segments:
            print("Error: No valid subtitle segments found")
            return 1

        # Show subtitle statistics
        stats = subtitle_parser.get_subtitle_stats()
        print(f"Loaded {stats['total_segments']} subtitle segments")
        print(
            f"Total subtitle duration: {stats['total_subtitle_duration']:.2f} seconds")

    except Exception as e:
        print(f"Error parsing subtitle file: {e}")
        return 1

    # Create generator
    generator = FFmpegCaptionGenerator(
        input_video=args.input_video,
        output_video=args.output,
        subtitle_segments=segments,
        font_size=args.font_size,
        font_color=args.font_color,
        position=args.position,
        margin_bottom=args.margin_bottom,
        max_width=args.max_width,
        line_spacing=args.line_spacing,
        quality=args.quality,
        use_gpu=not args.no_gpu
    )

    # Process video
    try:
        success = generator.process_video()
        return 0 if success else 1
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
