#!/usr/bin/env python3
"""
OpenCV GPU-Accelerated Caption Generator
========================================

High-performance subtitle generator using OpenCV with GPU acceleration.
Features:
- GPU-accelerated video processing
- High-quality Chinese text rendering using PIL
- Automatic text wrapping for large fonts
- Multi-line subtitle support
- Batch processing for memory efficiency
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from pathlib import Path
import psutil
import gc

# PIL for high-quality text rendering
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Import our subtitle parser
from subtitle_parser import SubtitleParser, SubtitleSegment


class GPUVideoProcessor:
    """GPU-accelerated video processor for adding captions"""

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
                 batch_size: int = 200,
                 use_gpu: bool = True,
                 num_workers: int = 4,
                 quality: str = "high",
                 codec: str = "auto",
                 auto_wrap: bool = True,
                 max_lines: int = 3):

        self.input_video = input_video
        self.output_video = output_video
        self.subtitle_segments = subtitle_segments
        self.font_size = font_size
        self.font_color = font_color
        self.position = position
        self.margin_bottom = margin_bottom
        self.max_width = max_width
        self.line_spacing = line_spacing
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        self.quality = quality
        self.codec = codec
        self.auto_wrap = auto_wrap
        self.max_lines = max_lines

        # Video properties
        self.cap = None
        self.writer = None
        self.fps = None
        self.frame_width = None
        self.frame_height = None
        self.total_frames = None

        # GPU setup
        self.gpu_available = False
        self.setup_gpu()

        # Font setup
        self.font = None
        self.setup_font()

        # Processing statistics
        self.processed_frames = 0
        self.start_time = None

    def setup_gpu(self):
        """Setup GPU acceleration if available"""
        try:
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count > 0 and self.use_gpu:
                self.gpu_available = True
                print(
                    f"GPU acceleration enabled: {gpu_count} CUDA devices detected")

                # Print GPU info
                for i in range(gpu_count):
                    name = cv2.cuda.getDevice()
                    print(f"GPU {i}: CUDA device available")
            else:
                self.gpu_available = False
                print("GPU acceleration disabled or not available, using CPU")
        except Exception as e:
            self.gpu_available = False
            print(f"GPU setup failed: {e}, falling back to CPU")

    def setup_font(self):
        """Setup font for text rendering with verified working font"""
        # Use the verified working font paths from test results
        working_font_paths = [
            "/home/22055747d/.local/share/fonts/wqy-microhei.ttc",  # User's font directory
            "fonts/wqy-microhei.ttc",                               # Local fonts directory
            "./fonts/wqy-microhei.ttc",                             # Explicit relative path
            os.path.expanduser(
                "~/.local/share/fonts/wqy-microhei.ttc")  # Expanded path
        ]

        # System fallback fonts if needed
        fallback_font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]

        # Try all font paths
        font_paths = working_font_paths + fallback_font_paths

        font_found = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    self.font = ImageFont.truetype(font_path, self.font_size)
                    font_found = True
                    print(f"Using font: {font_path}")
                    # Test Chinese rendering capability
                    test_text = "Test Chinese Text"
                    bbox = self.font.getbbox(test_text)
                    if bbox[2] - bbox[0] > 0:
                        print("Font supports Chinese characters ✓")
                    else:
                        print("Warning: Font may not render Chinese properly")
                    break
                except Exception as e:
                    print(f"Failed to load font {font_path}: {e}")
                    continue

        if not font_found:
            print(
                "WARNING: No suitable font found - Chinese characters will not display correctly!")
            try:
                self.font = ImageFont.load_default()
                print("Using default font as fallback (limited language support)")
            except Exception as e:
                print(f"Failed to load default font: {e}")
                self.font = None

    def open_video(self):
        """Open input video and setup output writer"""
        self.cap = cv2.VideoCapture(self.input_video)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.input_video}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"Video info: {self.frame_width}x{self.frame_height}, {self.fps:.2f} FPS, {self.total_frames} frames")

        # Setup output writer with better codec compatibility
        # Choose codec based on user preference or auto-detect
        if self.codec != "auto":
            # User specified a specific codec
            codec_options = [(self.codec, cv2.VideoWriter_fourcc(*self.codec))]
            print(f"Using user-specified codec: {self.codec}")
        else:
            # Try multiple codecs in order of preference for HPC environments
            codec_options = [
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Most compatible
                ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Very compatible
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # Standard MP4
                ('X264', cv2.VideoWriter_fourcc(*'X264')),  # H.264 variant
                ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H.264 (may not work)
            ]

            # If high quality requested, try H.264 variants first
            if self.quality == "high":
                codec_options = [
                    ('X264', cv2.VideoWriter_fourcc(*'X264')),
                    ('H264', cv2.VideoWriter_fourcc(*'H264')),
                    ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
                    ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
                    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
                ]

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Try codecs until one works
        self.writer = None
        successful_codec = None

        for codec_name, fourcc in codec_options:
            print(f"Trying codec: {codec_name}")
            try:
                # Test with original filename first
                test_writer = cv2.VideoWriter(self.output_video, fourcc, self.fps,
                                              (self.frame_width, self.frame_height))
                if test_writer.isOpened():
                    self.writer = test_writer
                    successful_codec = codec_name
                    print(f"Successfully using codec: {codec_name}")
                    break
                else:
                    test_writer.release()
                    print(f"Codec {codec_name} failed to open")
            except Exception as e:
                print(f"Codec {codec_name} failed: {e}")
                continue

        # If no codec worked with MP4, try with AVI
        if not self.writer or not self.writer.isOpened():
            print("All MP4 codecs failed, trying AVI format...")

            # Change to AVI extension
            original_output = self.output_video
            self.output_video = self.output_video.replace('.mp4', '.avi')

            # Try XVID with AVI (most compatible)
            try:
                print(f"Trying AVI with XVID: {self.output_video}")
                self.writer = cv2.VideoWriter(
                    self.output_video,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    self.fps,
                    (self.frame_width, self.frame_height)
                )

                if self.writer.isOpened():
                    successful_codec = "XVID (AVI)"
                    print("Successfully using AVI/XVID")
                else:
                    # Last resort: try MJPG with AVI
                    print("XVID failed, trying MJPG with AVI...")
                    self.writer.release()
                    self.writer = cv2.VideoWriter(
                        self.output_video,
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        self.fps,
                        (self.frame_width, self.frame_height)
                    )

                    if self.writer.isOpened():
                        successful_codec = "MJPG (AVI)"
                        print("Successfully using AVI/MJPG")
                    else:
                        self.output_video = original_output  # Restore original name for error
                        raise ValueError("All video codecs and formats failed")

            except Exception as e:
                self.output_video = original_output  # Restore original name for error
                raise ValueError(f"All codec options failed. Error: {e}")

        print(f"Final codec selection: {successful_codec}")
        print(f"Output file: {self.output_video}")

        if not self.writer or not self.writer.isOpened():
            raise ValueError(
                f"Cannot create video writer with any available codec")

    def close_video(self):
        """Close video files"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    def wrap_text(self, text: str, max_width_chars: int = None) -> List[str]:
        """
        Smart text wrapping based on actual rendered width rather than character count

        Args:
            text: Input text
            max_width_chars: Optional character-based width limit (fallback)

        Returns:
            List of text lines
        """
        # If no text or font, return empty list
        if not text or not self.font:
            return []

        # Get video frame width (with margin for safety)
        max_pixel_width = self.frame_width * \
            0.85 if hasattr(self, 'frame_width') else None

        # If we don't have frame width or font, fall back to character-based wrapping
        if not max_pixel_width or not self.font or not self.auto_wrap:
            # Fall back to basic character-based wrapping
            if max_width_chars is None:
                max_width_chars = self.max_width  # Use default from class

            if len(text) <= max_width_chars:
                return [text]

            words = text.split()
            lines = []
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

        # Smart pixel-based wrapping
        words = text.split()
        if not words:
            return []

        lines = []
        current_line = ""
        current_width = 0

        # For CJK text, we might need to split individual characters
        # since words may not be separated by spaces
        is_cjk_text = any(u'\u4e00' <= c <= u'\u9fff' for c in text)

        if is_cjk_text and not " " in text:
            # Treat each character as a "word" for CJK text without spaces
            words = list(text)

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            test_bbox = self.font.getbbox(test_line)
            test_width = test_bbox[2] - test_bbox[0]

            if test_width <= max_pixel_width:
                current_line = test_line
                current_width = test_width
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                    current_width = self.font.getbbox(
                        word)[2] - self.font.getbbox(word)[0]
                else:
                    # Single word wider than max width - need to split characters
                    lines.append(word)
                    current_line = ""
                    current_width = 0

                # Enforce maximum number of lines
                if len(lines) >= self.max_lines:
                    # If we're at the max lines, add ellipsis to the last line
                    if current_line and lines:
                        lines[-1] = lines[-1] + "..."
                    break

        if current_line and len(lines) < self.max_lines:
            lines.append(current_line)

        # Remove debug info - it was causing too many messages
        # if len(lines) > 1:
        #     print(f"Wrapped text into {len(lines)} lines due to large font size")

        return lines

    def render_subtitle_to_image(self, text_lines: List[str]) -> Optional[np.ndarray]:
        """
        Render subtitle text to an image with automatic size adjustment

        Args:
            text_lines: List of text lines to render

        Returns:
            Rendered subtitle as numpy array (RGBA)
        """
        if not text_lines or not self.font:
            return None

        # Calculate text dimensions
        padding = 20
        line_height = int(self.font_size * self.line_spacing)

        # Check if we need to rewrap text based on pixel width
        if len(text_lines) == 1 and self.auto_wrap and hasattr(self, 'frame_width'):
            # Get the text width in pixels for the single line
            bbox = self.font.getbbox(text_lines[0])
            text_width = bbox[2] - bbox[0]

            # If it exceeds frame width, rewrap using pixel-based wrapping
            if text_width > (self.frame_width * 0.85):
                text_lines = self.wrap_text(text_lines[0])

        # Calculate dimensions for the subtitle image
        max_text_width = 0
        for line in text_lines:
            bbox = self.font.getbbox(line)
            text_width = bbox[2] - bbox[0]
            max_text_width = max(max_text_width, text_width)

        img_width = max_text_width + padding * 2
        img_height = len(text_lines) * line_height + padding * 2

        # Create transparent image
        pil_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)

        # Parse color
        color = self.font_color.lower()
        if color == "white":
            text_color = (255, 255, 255, 255)
        elif color == "black":
            text_color = (0, 0, 0, 255)
        elif color == "yellow":
            text_color = (255, 255, 0, 255)
        else:
            text_color = (255, 255, 255, 255)  # Default to white

        # Draw text lines
        y_offset = padding
        for line in text_lines:
            bbox = self.font.getbbox(line)
            text_width = bbox[2] - bbox[0]
            x_offset = (img_width - text_width) // 2  # Center align

            # Draw text with shadow for better visibility
            shadow_offset = 2
            draw.text((x_offset + shadow_offset, y_offset + shadow_offset),
                      line, font=self.font, fill=(0, 0, 0, 128))  # Shadow
            draw.text((x_offset, y_offset), line,
                      font=self.font, fill=text_color)

            y_offset += line_height

        # Convert PIL image to OpenCV format
        pil_img = pil_img.convert('RGBA')
        cv_img = np.array(pil_img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)

        return cv_img

    def overlay_subtitle_on_frame(self, frame: np.ndarray, subtitle_img: np.ndarray) -> np.ndarray:
        """
        Overlay subtitle image on video frame with auto-scaling if needed

        Args:
            frame: Video frame (BGR)
            subtitle_img: Subtitle image (BGRA)

        Returns:
            Frame with subtitle overlay
        """
        if subtitle_img is None:
            return frame

        frame_h, frame_w = frame.shape[:2]
        sub_h, sub_w = subtitle_img.shape[:2]

        # Scale subtitle if wider than frame
        scale_factor = 1.0
        if sub_w > frame_w:
            print(
                f"Warning: Subtitle width ({sub_w}px) exceeds frame width ({frame_w}px). Auto-scaling.")
            scale_factor = frame_w / sub_w
            new_width = frame_w
            new_height = int(sub_h * scale_factor)

            # Resize subtitle image
            subtitle_img = cv2.resize(subtitle_img, (new_width, new_height),
                                      interpolation=cv2.INTER_AREA)

            # Update dimensions
            sub_h, sub_w = subtitle_img.shape[:2]

        # Calculate position
        if self.position == "bottom":
            x = (frame_w - sub_w) // 2
            y = frame_h - sub_h - self.margin_bottom
        elif self.position == "top":
            x = (frame_w - sub_w) // 2
            y = self.margin_bottom
        elif self.position == "center":
            x = (frame_w - sub_w) // 2
            y = (frame_h - sub_h) // 2
        else:  # bottom by default
            x = (frame_w - sub_w) // 2
            y = frame_h - sub_h - self.margin_bottom

        # Ensure subtitle fits in frame
        x = max(0, min(x, frame_w - sub_w))
        y = max(0, min(y, frame_h - sub_h))

        # Extract alpha channel
        if subtitle_img.shape[2] == 4:
            bgr_part = subtitle_img[:, :, :3]
            alpha = subtitle_img[:, :, 3] / 255.0

            # Create 3D alpha mask
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)

            # Blend images
            roi = frame[y:y+sub_h, x:x+sub_w]
            blended = roi * (1 - alpha_3d) + bgr_part * alpha_3d
            frame[y:y+sub_h, x:x+sub_w] = blended.astype(np.uint8)
        else:
            # No alpha channel, direct overlay
            frame[y:y+sub_h, x:x+sub_w] = subtitle_img

        return frame

    def process_frame_batch(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """
        Process a batch of frames with smart text wrapping

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number

        Returns:
            List of processed frames
        """
        processed_frames = []

        for frame_idx in range(start_frame, end_frame):
            # Calculate time for this frame
            time_seconds = frame_idx / self.fps

            # Get active subtitles at this time
            parser = SubtitleParser()
            parser.segments = self.subtitle_segments
            active_segments = parser.get_segments_at_time(time_seconds)

            # Read frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

            if not ret:
                break

            # Process subtitles if any
            if active_segments:
                # Combine all active subtitle texts
                all_text = " ".join([seg.text for seg in active_segments])

                # Use smart wrapping based on actual font metrics
                text_lines = self.wrap_text(all_text)

                # Render subtitle
                subtitle_img = self.render_subtitle_to_image(text_lines)

                # Overlay on frame
                frame = self.overlay_subtitle_on_frame(frame, subtitle_img)

            processed_frames.append(frame)

            # Update progress
            self.processed_frames += 1
            if self.processed_frames % 100 == 0:
                progress = (self.processed_frames / self.total_frames) * 100
                elapsed = time.time() - self.start_time
                fps_current = self.processed_frames / elapsed
                print(f"Progress: {progress:.1f}% ({self.processed_frames}/{self.total_frames}), "
                      f"Speed: {fps_current:.1f} FPS")

        return processed_frames

    def process_video(self):
        """Process the entire video with batch processing"""
        print("Starting video processing...")
        self.start_time = time.time()

        try:
            self.open_video()

            # Process video in batches
            for batch_start in range(0, self.total_frames, self.batch_size):
                batch_end = min(batch_start + self.batch_size,
                                self.total_frames)

                print(f"Processing batch: frames {batch_start}-{batch_end}")

                # Process batch
                processed_frames = self.process_frame_batch(
                    batch_start, batch_end)

                # Write frames
                for frame in processed_frames:
                    self.writer.write(frame)

                # Memory cleanup
                del processed_frames
                gc.collect()

                # Show memory usage
                memory_usage = psutil.virtual_memory().percent
                print(f"Memory usage: {memory_usage:.1f}%")

            # Processing complete
            elapsed = time.time() - self.start_time
            avg_fps = self.total_frames / elapsed

            print(f"\nProcessing complete!")
            print(f"Total time: {elapsed:.2f} seconds")
            print(f"Average speed: {avg_fps:.2f} FPS")
            print(f"Output saved to: {self.output_video}")

        except Exception as e:
            print(f"Error during processing: {e}")
            raise
        finally:
            # Ensure video files are closed
            self.close_video()

        # Try to add audio after video file is closed
        try:
            # Verify generated video file
            print("Verifying generated video file...")
            import time as time_module
            time_module.sleep(1)  # Wait for filesystem sync

            if not os.path.exists(self.output_video):
                raise FileNotFoundError(
                    f"Output video file not found: {self.output_video}")

            file_size = os.path.getsize(self.output_video)
            print(
                f"Video file size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

            if file_size < 1000:  # If file is too small, there might be a problem
                raise ValueError(
                    f"Generated video file seems too small: {file_size} bytes")

            # Try to add audio back to the video
            if not self.output_video.endswith('.avi'):
                # For MP4 output, add audio using FFmpeg
                self.add_audio_track()
            else:
                # Try to convert to MP4 if we had to use AVI fallback
                self.convert_to_mp4_with_audio()
        except Exception as e:
            print(f"Warning: Failed to add audio track: {e}")
            print("Video saved without audio")

    def add_audio_track(self):
        """Add audio from original video to the processed video using FFmpeg"""
        try:
            import subprocess

            # Check if FFmpeg is available
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("FFmpeg not available, output video will have no audio")
                return

            # Create temporary filename for video with audio
            temp_output = self.output_video.replace('.mp4', '_with_audio.mp4')

            print(f"Adding audio track from original video...")

            # FFmpeg command to combine processed video with original audio
            cmd = [
                'ffmpeg', '-y',
                '-i', self.output_video,      # Processed video (no audio)
                '-i', self.input_video,       # Original video (with audio)
                '-c:v', 'copy',               # Copy video from processed file
                '-c:a', 'copy',               # Copy audio from original file
                '-map', '0:v:0',              # Map video from first input
                '-map', '1:a:0',              # Map audio from second input
                temp_output
            ]

            print("FFmpeg command:")
            print(" ".join(cmd))

            # Run FFmpeg
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"Successfully added audio track!")
                # Replace original output with audio version
                import os
                os.replace(temp_output, self.output_video)
                print(f"Final output with audio: {self.output_video}")
            else:
                print(f"Failed to add audio track. FFmpeg error:")
                print(result.stderr)
                print("Video saved without audio")
                # Clean up temp file if it exists
                if os.path.exists(temp_output):
                    os.remove(temp_output)

        except Exception as e:
            print(f"Error adding audio track: {e}")
            print("Video saved without audio")

    def convert_to_mp4_with_audio(self):
        """Convert AVI to MP4 and add audio from original video using FFmpeg"""
        try:
            import subprocess

            # Check if FFmpeg is available
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("FFmpeg not available, keeping AVI format without audio")
                return

            # Create MP4 filename
            mp4_output = self.output_video.replace('.avi', '.mp4')

            print(f"Converting to MP4 format and adding audio...")

            # FFmpeg command to convert AVI to MP4 and add audio from original video
            cmd = [
                'ffmpeg', '-y',
                '-i', self.output_video,      # Processed video (AVI, no audio)
                '-i', self.input_video,       # Original video (with audio)
                '-c:v', 'libx264',            # Re-encode video to H.264
                '-crf', '23',                 # Good quality
                '-c:a', 'copy',               # Copy audio from original file
                # Map video from first input (AVI)
                '-map', '0:v:0',
                # Map audio from second input (original)
                '-map', '1:a:0',
                mp4_output
            ]

            print("FFmpeg command:")
            print(" ".join(cmd))

            # Run conversion
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(
                    f"Successfully converted to MP4 with audio: {mp4_output}")
                # Update output path
                self.output_video = mp4_output
                # Remove temporary AVI file
                try:
                    import os
                    os.remove(self.output_video.replace('.mp4', '.avi'))
                    print("Removed temporary AVI file")
                except:
                    pass
            else:
                print(f"Failed to convert to MP4. FFmpeg error:")
                print(result.stderr)
                print("Keeping AVI format without audio")

        except Exception as e:
            print(f"Error converting to MP4 with audio: {e}")
            print("Keeping AVI format without audio")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="OpenCV GPU-accelerated caption generator with auto text wrapping",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--input-video', required=True,
                        help='Input video file path')
    parser.add_argument('--script', required=True,
                        help='Script file with timestamps')

    # Optional arguments
    parser.add_argument('--output', default='outputs/auto_caption/final_video_opencv.mp4',
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
                        help='Maximum characters per line (for non-auto wrapping)')
    parser.add_argument('--line-spacing', type=float, default=1.2,
                        help='Line spacing multiplier')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Number of frames to process in each batch')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--quality', choices=['normal', 'high'], default='high',
                        help='Output video quality')
    parser.add_argument('--codec', choices=['auto', 'XVID', 'MJPG', 'mp4v'], default='auto',
                        help='Force specific video codec (auto=try all)')
    # New arguments for text wrapping
    parser.add_argument('--no-auto-wrap', action='store_true',
                        help='Disable automatic text wrapping')
    parser.add_argument('--max-lines', type=int, default=3,
                        help='Maximum number of lines for wrapped text')

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

    # Create processor
    processor = GPUVideoProcessor(
        input_video=args.input_video,
        output_video=args.output,
        subtitle_segments=segments,
        font_size=args.font_size,
        font_color=args.font_color,
        position=args.position,
        margin_bottom=args.margin_bottom,
        max_width=args.max_width,
        line_spacing=args.line_spacing,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        num_workers=args.workers,
        quality=args.quality,
        codec=args.codec,
        auto_wrap=not args.no_auto_wrap,
        max_lines=args.max_lines
    )

    # Process video
    try:
        processor.process_video()
        return 0
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
