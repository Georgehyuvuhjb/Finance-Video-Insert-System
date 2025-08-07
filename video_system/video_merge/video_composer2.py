import json
import os
import argparse
import subprocess
import re
import tempfile
from tqdm import tqdm
import cv2
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

def parse_position(position_str, main_width, main_height, overlay_width, overlay_height):
    """
    Parse position string and calculate coordinates for overlay center
    
    Args:
        position_str: Position string in format "x,y" or "x%,y%"
        main_width: Width of main video
        main_height: Height of main video
        overlay_width: Width of overlay video
        overlay_height: Height of overlay video
        
    Returns:
        tuple: (x, y) coordinates for overlay top-left position
    """
    # Parse position
    try:
        x_str, y_str = position_str.split(",")
        x_str = x_str.strip()
        y_str = y_str.strip()
        
        # Parse X coordinate
        if "%" in x_str:
            x_percent = float(x_str.rstrip("%")) / 100
            center_x = int(main_width * x_percent)
        else:
            center_x = int(x_str)
            
        # Parse Y coordinate
        if "%" in y_str:
            y_percent = float(y_str.rstrip("%")) / 100
            center_y = int(main_height * y_percent)
        else:
            center_y = int(y_str)
        
        # Convert center position to top-left position for overlay
        x = center_x - overlay_width // 2
        y = center_y - overlay_height // 2
        
        return (x, y)
    except Exception as e:
        print(f"Error parsing position: {e}, using center position")
        # Default to center
        x = (main_width - overlay_width) // 2
        y = (main_height - overlay_height) // 2
        return (x, y)

def parse_size(size_str, main_width, main_height, original_width, original_height):
    """
    Parse size specification and calculate target dimensions while preserving aspect ratio
    
    Args:
        size_str: Size string like "25%" or "320px" or "30%w" or "240px-h"
        main_width: Width of the main video
        main_height: Height of the main video
        original_width: Original width of the overlay video
        original_height: Original height of the overlay video
        
    Returns:
        tuple: (target_width, target_height) for the resized overlay
    """
    # Original aspect ratio
    aspect_ratio = original_width / original_height if original_height > 0 else 16/9
    
    # Default size (25% of main video width)
    if not size_str:
        target_width = int(main_width * 0.25)
        target_height = int(target_width / aspect_ratio)
        return target_width, target_height
    
    size_str = size_str.strip()
    
    # Handle percentage of main video width (e.g. "25%")
    if size_str.endswith("%") and not any(x in size_str for x in ['w', 'h', 'x']):
        percent = float(size_str.rstrip("%")) / 100
        target_width = int(main_width * percent)
        target_height = int(target_width / aspect_ratio)
        return target_width, target_height
    
    # Handle width specification (e.g. "320px" or "30%w")
    elif any(x in size_str for x in ['w', 'px']) and 'h' not in size_str:
        if 'w' in size_str:
            # Percentage of main width
            size_str = size_str.rstrip('w')
            if size_str.endswith("%"):
                percent = float(size_str.rstrip("%")) / 100
                target_width = int(main_width * percent)
            else:
                target_width = int(size_str)
        else:
            # Absolute pixels
            size_str = size_str.rstrip('px')
            target_width = int(size_str)
            
        target_height = int(target_width / aspect_ratio)
        return target_width, target_height
    
    # Handle height specification (e.g. "240px-h" or "20%h")
    elif 'h' in size_str:
        size_str = size_str.rstrip('h').rstrip('-')
        if size_str.endswith("%"):
            # Percentage of main height
            percent = float(size_str.rstrip("%")) / 100
            target_height = int(main_height * percent)
        else:
            # Absolute pixels
            size_str = size_str.rstrip('px')
            target_height = int(size_str)
            
        target_width = int(target_height * aspect_ratio)
        return target_width, target_height
    
    # Handle absolute size (fallback)
    else:
        try:
            # Try parsing as a simple number (treated as width)
            target_width = int(size_str)
            target_height = int(target_width / aspect_ratio)
            return target_width, target_height
        except ValueError:
            # Default if parsing fails
            print(f"Could not parse size '{size_str}', using default (25% of width)")
            target_width = int(main_width * 0.25)
            target_height = int(target_width / aspect_ratio)
            return target_width, target_height

def run_ffmpeg_command(cmd, description="FFmpeg operation", show_output=False):
    """Run FFmpeg command with better error handling"""
    try:
        if show_output:
            result = subprocess.run(cmd, check=True)
        else:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: Exit code {e.returncode}")
        if e.stderr:
            print(f"Error details: {e.stderr.decode('utf-8', errors='replace')}")
        return False

class VideoComposer:
    """A class for composing videos based on matched sentences and video segments"""
    
    def __init__(self, json_input, transcript_file, output_file="composed_video.mp4", 
                 position="center", size="25%", distance_threshold=2.0, debug=False,
                 audio_file=None, audio_start=0.0):
        """
        Initialize the video composer
        
        Args:
            json_input (str): Path to JSON file containing matched videos
            transcript_file (str): Path to transcript file with timings
            output_file (str): Path to output video file
            position (str): Position string to insert video clips (center of overlay)
            size (str): Size string for overlay videos
            distance_threshold (float): Maximum distance threshold for video inclusion
            debug (bool): Whether to show detailed debug info
        """
        self.json_input = json_input
        self.transcript_file = transcript_file
        self.output_file = output_file
        self.position = position
        self.size = size
        self.distance_threshold = distance_threshold
        self.debug = debug
        self.audio_file = audio_file
        self.audio_start = audio_start
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create temp directories with unique names
        self.temp_dir = tempfile.mkdtemp(prefix="temp_videos_")
        self.segments_dir = tempfile.mkdtemp(prefix="video_segments_")
        
        # Load matched videos data
        with open(json_input, 'r', encoding='utf-8') as f:
            self.matched_data = json.load(f)
        
        # Load transcript data
        self.transcript = self.load_transcript(transcript_file)
        
        print(f"Loaded {len(self.matched_data)} sentence groups and {len(self.transcript)} transcript entries")
    
    def load_transcript(self, transcript_file):
        """Load transcript data from file"""
        transcript = []
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse line format: start_time_end_time_text
                match = re.match(r'(\d+:\d+\.\d+)_(\d+:\d+\.\d+)_(.+)', line)
                if match:
                    start_time, end_time, text = match.groups()
                    
                    # Convert time format (MM:SS.ms) to seconds
                    def time_to_seconds(time_str):
                        parts = time_str.split(':')
                        if len(parts) == 2:
                            minutes, seconds = parts
                            return float(minutes) * 60 + float(seconds)
                        else:
                            return float(time_str)
                    
                    start_seconds = time_to_seconds(start_time)
                    end_seconds = time_to_seconds(end_time)
                    
                    transcript.append({
                        'start_time': start_seconds,
                        'end_time': end_seconds,
                        'text': text
                    })
        
        return transcript
    
    def download_videos(self):
        """Download videos that meet the distance threshold"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        downloaded_videos = []
        
        for group in tqdm(self.matched_data, desc="Downloading videos"):
            # Get top video that meets the distance threshold
            videos = group["matching_videos"]
            eligible_videos = [v for v in videos if v["distance"] < self.distance_threshold]
            
            if not eligible_videos:
                print(f"No eligible videos found for sentence: {group['sentence_group']['sentences'][0][:30]}...")
                continue
            
            # Use the top video
            top_video = eligible_videos[0]
            video_id = top_video["video_id"]
            video_url = top_video["url"]
            
            # Local path for downloaded video
            local_path = os.path.join(self.temp_dir, f"video_{video_id}.mp4")
            
            # Download if not already present
            if not os.path.exists(local_path):
                try:
                    print(f"Downloading video {video_id} from {video_url}")
                    
                    # Try different download methods
                    download_success = False
                    
                    # Try wget first
                    try:
                        result = subprocess.run(["wget", "-q", "-O", local_path, video_url], 
                                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        download_success = True
                    except Exception as e:
                        if self.debug:
                            print(f"wget download failed: {e}")
                    
                    # Try curl if wget failed
                    if not download_success:
                        try:
                            result = subprocess.run(["curl", "-s", "-L", "-o", local_path, video_url], 
                                                  check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            download_success = True
                        except Exception as e:
                            if self.debug:
                                print(f"curl download failed: {e}")
                    
                    # Try Python's urllib if both wget and curl failed
                    if not download_success:
                        try:
                            import urllib.request
                            urllib.request.urlretrieve(video_url, local_path)
                            download_success = True
                        except Exception as e:
                            print(f"All download methods failed for {video_id}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error downloading video {video_id}: {e}")
                    continue
            
            # Calculate segment duration based on text length
            text_length = sum(len(s) for s in group["sentence_group"]["sentences"])
            needed_duration = max(3.0, min(15.0, text_length * 0.15))
            
            # Check if the video exists and get properties
            if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000:
                print(f"Downloaded file is invalid or too small: {local_path}")
                continue
                
            # Check video duration and properties using MoviePy instead of OpenCV
            try:
                clip = VideoFileClip(local_path)
                
                if clip.duration < 0.1:  # Sanity check
                    print(f"Video {video_id} is too short or invalid")
                    clip.close()
                    continue
                    
                width, height = clip.size
                fps = clip.fps
                duration = clip.duration
                
                # Close clip to free resources
                clip.close()
                
                # Limit segment duration to video length
                segment_duration = min(needed_duration, duration)
                
                # Add to downloaded videos
                downloaded_videos.append({
                    "video_id": video_id,
                    "local_path": local_path,
                    "start_time": 0,  # Start from beginning
                    "duration": segment_duration,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "sentence_group": group["sentence_group"],
                    "distance": top_video["distance"]
                })
            except Exception as e:
                print(f"Error inspecting video {video_id}: {e}")
                if os.path.exists(local_path):
                    # Try with ffprobe as fallback
                    try:
                        cmd = ["ffprobe", "-v", "error", "-show_entries", 
                               "stream=width,height,r_frame_rate,duration", 
                               "-of", "default=noprint_wrappers=1:nokey=1", local_path]
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
                        if result.returncode == 0:
                            lines = result.stdout.strip().split('\n')
                            if len(lines) >= 4:
                                width = int(float(lines[0]))
                                height = int(float(lines[1]))
                                
                                # Parse framerate (can be in format "num/den")
                                fps_parts = lines[2].split('/')
                                if len(fps_parts) == 2:
                                    fps = float(fps_parts[0]) / float(fps_parts[1])
                                else:
                                    fps = float(fps_parts[0])
                                
                                duration = float(lines[3])
                                
                                # Limit segment duration to video length
                                segment_duration = min(needed_duration, duration)
                                
                                downloaded_videos.append({
                                    "video_id": video_id,
                                    "local_path": local_path,
                                    "start_time": 0,
                                    "duration": segment_duration,
                                    "width": width,
                                    "height": height,
                                    "fps": fps,
                                    "sentence_group": group["sentence_group"],
                                    "distance": top_video["distance"]
                                })
                                continue
                    except Exception as probe_err:
                        print(f"Fallback probe failed for video {video_id}: {probe_err}")
        
        print(f"Downloaded {len(downloaded_videos)} videos")
        return downloaded_videos
    
    def process_videos(self, downloaded_videos):
        """Process videos directly without extracting segments"""
        processed_videos = []
        
        for video in tqdm(downloaded_videos, desc="Processing videos"):
            video_id = video["video_id"]
            local_path = video["local_path"]
            
            # Verify the video can be opened
            try:
                clip = VideoFileClip(local_path)
                clip.close()
                processed_videos.append(video)
            except Exception as e:
                print(f"Error: Cannot process video {video_id}: {e}")
        
        print(f"Successfully processed {len(processed_videos)} videos")
        return processed_videos
    
    def match_segments_to_transcript(self, videos):
        """
        Match videos to transcript entries based on sentence content matching.
        
        This function creates a proper mapping between transcript entries and video segments
        by matching the actual text content rather than using simple sequential indexing.
        This fixes the issue where videos would repeat incorrectly due to misaligned mapping.
        
        Args:
            videos: List of video data with sentence groups
            
        Returns:
            List of matched segments with transcript and video pairs
        """
        # Sort videos by their order
        videos.sort(key=lambda x: x["sentence_group"]["start_index"])
        
        matched_segments = []
        
        # Create a mapping from transcript text to video group
        text_to_video_map = {}
        for video in videos:
            sentences = video["sentence_group"]["sentences"]
            for sentence in sentences:
                # Use cleaned sentence text as key
                cleaned_sentence = sentence.strip()
                text_to_video_map[cleaned_sentence] = video
        
        # Match each transcript entry to the correct video based on text content
        for transcript_entry in self.transcript:
            # Extract text from transcript entry
            transcript_text = transcript_entry['text'].strip()
            
            # Find matching video
            matching_video = text_to_video_map.get(transcript_text)
            
            if matching_video:
                matched_segments.append({
                    "transcript": transcript_entry,
                    "video": matching_video
                })
                if self.debug:
                    print(f"Matched transcript: '{transcript_text[:30]}...' -> Video {matching_video['video_id']}")
            else:
                # If no exact match found, try to find the best semantic match
                # by checking if transcript text is contained in any sentence
                best_match = None
                for video in videos:
                    for sentence in video["sentence_group"]["sentences"]:
                        if transcript_text in sentence or sentence in transcript_text:
                            best_match = video
                            break
                    if best_match:
                        break
                
                if best_match:
                    matched_segments.append({
                        "transcript": transcript_entry,
                        "video": best_match
                    })
                    if self.debug:
                        print(f"Fuzzy matched transcript: '{transcript_text[:30]}...' -> Video {best_match['video_id']}")
                else:
                    print(f"Warning: No matching video found for transcript: '{transcript_text[:50]}...'")
        
        print(f"Successfully matched {len(matched_segments)} transcript entries to videos")
        return matched_segments
    
    def compose_video(self, input_video, matched_segments):
        """
        Compose final video by inserting segments at the right times
        
        Args:
            input_video (str): Path to the input video file
            matched_segments (list): List of matched segments and transcript entries
        """
        # Get input video dimensions
        main_clip = VideoFileClip(input_video)
        main_width, main_height = main_clip.size
        main_fps = main_clip.fps
        
        print(f"Loading input video: {input_video} ({main_width}x{main_height}, {main_fps} fps)")
        
        # Prepare video clips to add
        composed_clips = [main_clip]
        
        # For each matched segment, add it to the composed clips
        for match in tqdm(matched_segments, desc="Composing video"):
            transcript = match["transcript"]
            video_data = match["video"]
            
            start_time = transcript["start_time"]
            end_time = transcript["end_time"]
            video_path = video_data["local_path"]
            
            try:
                # Get the clip
                clip_duration = min(end_time - start_time, video_data["duration"])
                if clip_duration <= 0:
                    print(f"Warning: Invalid clip duration: {clip_duration}s")
                    continue
                
                # Load video clip, trimming if needed
                segment_clip = VideoFileClip(video_path).subclip(0, clip_duration)
                
                # Get dimensions
                original_width = video_data["width"]
                original_height = video_data["height"]
                
                # Calculate target size while preserving aspect ratio
                target_width, target_height = parse_size(
                    self.size, main_width, main_height, original_width, original_height
                )
                
                # Resize the clip
                segment_clip = segment_clip.resize(width=target_width, height=target_height)
                
                # Calculate position based on the center coordinates
                x, y = parse_position(
                    self.position, main_width, main_height, target_width, target_height
                )
                
                # Position the clip
                segment_clip = segment_clip.set_position((x, y))
                
                # Set the time range
                segment_clip = segment_clip.set_start(start_time).set_duration(clip_duration)
                
                # Add to composed clips
                composed_clips.append(segment_clip)
                
                print(f"Added video clip at {start_time:.2f}-{end_time:.2f}s, position {x},{y}, size {target_width}x{target_height}")
                
            except Exception as e:
                print(f"Error adding video {video_path}: {str(e)}")
        
        # Create composite clip
        print("Creating final composition...")
        final_clip = CompositeVideoClip(composed_clips)
        
        # Add audio if specified
        if self.audio_file and os.path.exists(self.audio_file):
            print(f"Adding audio track: {self.audio_file}")
            from moviepy.editor import AudioFileClip
            
            try:
                audio_clip = AudioFileClip(self.audio_file)
                
                # Apply audio start offset if specified
                if self.audio_start > 0:
                    audio_clip = audio_clip.subclip(self.audio_start)
                
                # Set audio to final clip, matching video duration
                final_clip = final_clip.set_audio(audio_clip.subclip(0, final_clip.duration))
                
            except Exception as e:
                print(f"Warning: Failed to add audio: {e}")
        
        # Write output with progress bar
        print(f"Writing output to {self.output_file}")
        final_clip.write_videofile(
            self.output_file, 
            codec="libx264", 
            audio_codec="aac",
            threads=4,  # Use multiple CPU cores
            logger='bar'  # Show progress bar
        )
        
        # Close clips to free resources
        for clip in composed_clips:
            try:
                clip.close()
            except:
                pass
        
        print(f"Video composition complete: {self.output_file}")
        
        # Clean up temporary directories
        try:
            import shutil
            for path in [self.temp_dir, self.segments_dir]:
                if os.path.exists(path):
                    shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directories: {e}")

def main():
    parser = argparse.ArgumentParser(description='Compose video from matched segments and input video')
    parser.add_argument('--json', required=True, help='Path to JSON file with matched videos')
    parser.add_argument('--transcript', required=True, help='Path to transcript file')
    parser.add_argument('--input-video', required=True, help='Path to input video')
    parser.add_argument('--output', default='composed_video.mp4', help='Path to output video')
    parser.add_argument('--position', default='70%,50%', 
                       help='Position to insert video clips (center of overlay). Can be "x,y" coordinates or percentage (e.g., "50%%,50%%")')
    parser.add_argument('--size', default='52%', 
                       help='Size of inserted videos (preserves aspect ratio). Examples: ' 
                       '"25%%" (25%% of main video width), '
                       '"320px" (320 pixels wide), '
                       '"30%%w" (30%% of main video width), '
                       '"240px-h" (240 pixels tall), '
                       '"20%%h" (20%% of main video height)')
    parser.add_argument('--distance-threshold', type=float, default=2.0, 
                       help='Maximum distance threshold for video inclusion')
    parser.add_argument('--debug', action='store_true', help='Show detailed debug information')
    parser.add_argument('--audio', help='Path to audio file to add to the final video')
    parser.add_argument('--audio-start', type=float, default=0.0, 
                       help='Start time for audio in seconds (default: 0.0)')

    args = parser.parse_args()
    
    # Create composer and process video
    composer = VideoComposer(
        args.json, 
        args.transcript, 
        args.output, 
        args.position,
        args.size,
        args.distance_threshold,
        args.debug,
        audio_file=args.audio,
        audio_start=args.audio_start
    )
    
    # Download eligible videos
    downloaded_videos = composer.download_videos()
    
    # Process videos (skip extraction, use original files)
    processed_videos = composer.process_videos(downloaded_videos)
    
    # Match videos to transcript
    matched_segments = composer.match_segments_to_transcript(processed_videos)
    
    # Compose final video
    composer.compose_video(args.input_video, matched_segments)

if __name__ == "__main__":
    main()