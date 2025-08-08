import os
import json
import argparse
import re
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import glob
import gc

class PyTorchVideoProcessor:
    """Video processing using PyTorch with GPU acceleration and memory optimization"""
    
    def __init__(self, use_gpu=True, batch_size=50, maintain_resolution=True):
        """
        Initialize the video processor with memory optimization
        
        Args:
            use_gpu: Whether to use GPU
            batch_size: Number of frames to process at once
            maintain_resolution: Whether to maintain original resolution
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.maintain_resolution = maintain_resolution
        print(f"Using device: {self.device}")
        print(f"Batch size: {self.batch_size}, Maintain original resolution: {self.maintain_resolution}")
        
        # Initialize transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
        ])
        
        self.inverse_transform = T.ToPILImage()
    
    def read_video_in_batches(self, video_path, start_frame=0, max_frames=None):
        """Read video file in batches to avoid memory issues"""
        print(f"Reading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit number of frames if specified
        if max_frames is not None:
            frame_count = min(frame_count, start_frame + max_frames)
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        video_metadata = {
            "fps": fps, 
            "width": width, 
            "height": height, 
            "frame_count": frame_count
        }
        
        return cap, video_metadata
    
    def process_frame_batch(self, cap, batch_size):
        """Process a batch of frames"""
        frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            frame_tensor = self.transform(Image.fromarray(frame))
            frames.append(frame_tensor)
        
        # Stack frames into a tensor if we have any
        if frames:
            return torch.stack(frames).to(self.device)
        return None
    
    def resize_batch(self, batch_tensor, target_size):
        """Resize a batch of frames"""
        _, c, h, w = batch_tensor.shape
        target_h, target_w = target_size
        
        if h == target_h and w == target_w:
            return batch_tensor
        
        # Use PyTorch's interpolate for resizing
        resized_tensor = F.interpolate(
            batch_tensor, 
            size=(target_h, target_w),
            mode='bilinear', 
            align_corners=False
        )
        
        return resized_tensor
    
    def overlay_batch(self, main_batch, overlay_batch, position):
        """Overlay one batch onto another batch at position"""
        main_frames, _, main_h, main_w = main_batch.shape
        overlay_frames, _, overlay_h, overlay_w = overlay_batch.shape
        x, y = position
        
        # Make sure position is valid
        x = max(0, min(x, main_w - overlay_w))
        y = max(0, min(y, main_h - overlay_h))
        
        # Create a copy of the main tensor to modify
        result_tensor = main_batch.clone()
        
        # Get the number of frames to process
        frames_to_process = min(main_frames, overlay_frames)
        
        # Process each frame
        for i in range(frames_to_process):
            # Extract the region to overlay
            main_region = result_tensor[i, :, y:y+overlay_h, x:x+overlay_w]
            
            # Apply overlay with alpha blending
            alpha = 1.0  # Full opacity
            result_tensor[i, :, y:y+overlay_h, x:x+overlay_w] = \
                (1.0 - alpha) * main_region + alpha * overlay_batch[i]
        
        return result_tensor
    
    def write_batch_to_video(self, video_writer, batch_tensor):
        """Write a batch of frames to video file"""
        batch_size = batch_tensor.shape[0]
        
        for i in range(batch_size):
            # Get frame and convert to numpy
            frame = batch_tensor[i].cpu().detach()
            frame_pil = self.inverse_transform(frame)
            frame_np = np.array(frame_pil)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Write frame
            video_writer.write(frame_bgr)

class PyTorchVideoComposer:
    """A class for composing videos with PyTorch GPU acceleration and memory optimization"""
    
    def __init__(self, json_input, transcript_file, output_file="composed_video.mp4", 
                 position="center", size="25%", distance_threshold=2.0, videos_dir="videos", 
                 use_gpu=True, batch_size=50, audio_file=None, audio_start=0.0):
        """Initialize the video composer with PyTorch"""
        self.json_input = json_input
        self.transcript_file = transcript_file
        self.output_file = output_file
        self.position = position
        self.size = size
        self.distance_threshold = distance_threshold
        self.videos_dir = videos_dir
        self.batch_size = batch_size
        self.audio_file = audio_file
        self.audio_start = audio_start
        
        # Initialize PyTorch video processor
        self.processor = PyTorchVideoProcessor(
            use_gpu=use_gpu, 
            batch_size=batch_size, 
            maintain_resolution=True  # Keep original resolution
        )
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
    
    def parse_position(self, position_str, main_width, main_height, overlay_width, overlay_height):
        """Parse position string and calculate coordinates for overlay center"""
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
    
    def parse_size(self, size_str, main_width, main_height, original_width, original_height):
        """Parse size string and calculate target dimensions while preserving aspect ratio"""
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
    
    def find_videos(self):
        """Find pre-downloaded videos that match the required IDs"""
        found_videos = []
        
        # Get all video files in the videos directory
        video_files = glob.glob(os.path.join(self.videos_dir, "*.mp4"))
        video_dict = {}
        
        # Create a dictionary of videos by ID
        for video_path in video_files:
            # Extract video ID from filename (format: ID_keyword_quality.mp4)
            filename = os.path.basename(video_path)
            parts = filename.split('_', 1)  # Split at first underscore
            if len(parts) >= 1:
                video_id = parts[0]
                try:
                    # Make sure ID is a number
                    video_id = int(video_id)
                    video_dict[str(video_id)] = video_path
                except ValueError:
                    pass
        
        print(f"Found {len(video_dict)} videos in directory {self.videos_dir}")
        
        for group in tqdm(self.matched_data, desc="Finding matching videos"):
            # Get videos that meet the distance threshold
            videos = group["matching_videos"]
            eligible_videos = [v for v in videos if v["distance"] < self.distance_threshold]
            
            if not eligible_videos:
                print(f"No eligible videos found for sentence: {group['sentence_group']['sentences'][0][:30]}...")
                continue
            
            # Try each eligible video until we find one that exists in our directory
            found_match = False
            for video_data in eligible_videos:
                video_id = str(video_data["video_id"])
                
                if video_id in video_dict:
                    video_path = video_dict[video_id]
                    
                    # Check video properties with OpenCV
                    try:
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            print(f"Could not open video: {video_path}")
                            continue
                        
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        cap.release()
                        
                        # Calculate segment duration based on text length
                        text_length = sum(len(s) for s in group["sentence_group"]["sentences"])
                        needed_duration = max(3.0, min(15.0, text_length * 0.15))
                        segment_duration = min(needed_duration, duration)
                        
                        found_videos.append({
                            "video_id": video_id,
                            "local_path": video_path,
                            "start_time": 0,
                            "duration": segment_duration,
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "frame_count": frame_count,
                            "sentence_group": group["sentence_group"],
                            "distance": video_data["distance"]
                        })
                        
                        found_match = True
                        break
                    except Exception as e:
                        print(f"Error checking video {video_id}: {e}")
            
            if not found_match:
                print(f"No matching video found in directory for sentence group {group['sentence_group']['start_index']}")
        
        print(f"Found {len(found_videos)} matching videos in directory")
        return found_videos
    
    def match_videos_to_transcript(self, videos):
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
        videos.sort(key=lambda x: x["sentence_group"]["start_index"])
        
        matched_segments = []
        
        # Create a mapping from transcript text to video group
        text_to_video_map = {}
        for i, video in enumerate(videos):
            sentences = video["sentence_group"]["sentences"]
            for sentence in sentences:
                # Use cleaned sentence text as key
                cleaned_sentence = sentence.strip()
                # Add segment ID for tracking playback position across batches
                video["segment_id"] = f"segment_{i}"
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
                    "video": matching_video,
                    "id": matching_video["segment_id"]  # Add ID for tracking
                })
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
                        "video": best_match,
                        "id": best_match["segment_id"]  # Add ID for tracking
                    })
                    print(f"Fuzzy matched transcript: '{transcript_text[:30]}...' -> Video {best_match['video_id']}")
                else:
                    print(f"Warning: No matching video found for transcript: '{transcript_text[:50]}...'")
        
        print(f"Successfully matched {len(matched_segments)} transcript entries to videos")
        return matched_segments
    
    def compose_video(self, input_video, matched_segments):
        """
        Compose final video by inserting segments at the right times using PyTorch
        with memory optimization through batch processing and continuity between batches
        """
        try:
            # Open main video with batch processing
            main_cap, main_metadata = self.processor.read_video_in_batches(input_video)
            main_width = main_metadata["width"]
            main_height = main_metadata["height"]
            main_fps = main_metadata["fps"]
            main_frame_count = main_metadata["frame_count"]
            
            print(f"Main video properties: {main_width}x{main_height}, {main_fps} fps, {main_frame_count} frames")
            
            # Prepare segments by frame indices
            frame_segments = []
            for match in matched_segments:
                transcript = match["transcript"]
                video_data = match["video"]
                
                start_time = transcript["start_time"]
                end_time = transcript["end_time"]
                
                # Calculate frame indices
                start_frame = int(start_time * main_fps)
                end_frame = int(end_time * main_fps)
                
                # Ensure valid frame indices
                start_frame = max(0, min(start_frame, main_frame_count - 1))
                end_frame = max(start_frame + 1, min(end_frame, main_frame_count))
                
                if start_frame >= end_frame:
                    print(f"Warning: Invalid frame range: {start_frame}-{end_frame}, skipping")
                    continue
                
                frame_segments.append({
                    "segment_id": match["id"],  # Add unique identifier
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "video_data": video_data
                })
            
            # Sort segments by start frame
            frame_segments.sort(key=lambda x: x["start_frame"])
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_file, fourcc, main_fps, (main_width, main_height))
            
            if not out.isOpened():
                raise ValueError(f"Could not open output video file: {self.output_file}")
            
            # Maintain overlay video state between batches
            active_overlays = {}  # Map segment_id to (cap, frame_position, target_size, position)
            
            # Process the main video in batches
            current_frame = 0
            current_segment_idx = 0
            
            with tqdm(total=main_frame_count, desc="Processing video") as pbar:
                while current_frame < main_frame_count:
                    # Calculate batch end frame
                    batch_end = min(current_frame + self.batch_size, main_frame_count)
                    
                    # Read batch from main video
                    main_batch = self.processor.process_frame_batch(
                        main_cap, batch_end - current_frame
                    )
                    
                    if main_batch is None:
                        break
                    
                    # Find segments that overlap with current batch
                    overlapping_segments = []
                    
                    # Check all remaining segments
                    temp_idx = current_segment_idx
                    while temp_idx < len(frame_segments):
                        segment = frame_segments[temp_idx]
                        
                        # Check if segment overlaps with current batch
                        if segment["start_frame"] < batch_end and segment["end_frame"] > current_frame:
                            overlapping_segments.append(segment)
                        
                        # If segment starts after this batch, we can stop checking
                        if segment["start_frame"] >= batch_end:
                            break
                            
                        temp_idx += 1
                    
                    # Apply overlays for each overlapping segment
                    for segment in overlapping_segments:
                        segment_id = segment["segment_id"]
                        segment_start = max(segment["start_frame"], current_frame)
                        segment_end = min(segment["end_frame"], batch_end)
                        
                        if segment_start >= segment_end:
                            continue
                        
                        # Adjust indices for batch
                        batch_start_idx = segment_start - current_frame
                        batch_end_idx = segment_end - current_frame
                        
                        # Get the relevant portion of main batch
                        main_portion = main_batch[batch_start_idx:batch_end_idx]
                        
                        try:
                            video_data = segment["video_data"]
                            overlay_path = video_data["local_path"]
                            
                            # Calculate frames needed from overlay
                            overlay_frames_needed = batch_end_idx - batch_start_idx
                            
                            # Check if we have an active overlay for this segment
                            if segment_id in active_overlays:
                                # Continue from previous batch
                                overlay_cap, frame_position, target_size, position = active_overlays[segment_id]
                                
                                # Verify the cap is still valid, if not reopen it
                                if not overlay_cap.isOpened():
                                    overlay_cap = cv2.VideoCapture(overlay_path)
                                    overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                            else:
                                # First time seeing this segment, initialize overlay video
                                overlay_cap = cv2.VideoCapture(overlay_path)
                                
                                if not overlay_cap.isOpened():
                                    print(f"Could not open overlay video: {overlay_path}")
                                    continue
                                
                                # Get overlay video dimensions
                                overlay_width = int(overlay_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                overlay_height = int(overlay_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                
                                # Calculate target size
                                target_size = self.parse_size(
                                    self.size, main_width, main_height,
                                    overlay_width, overlay_height
                                )
                                
                                # Make dimensions even
                                target_width = target_size[0] + (target_size[0] % 2)
                                target_height = target_size[1] + (target_size[1] % 2)
                                target_size = (target_width, target_height)
                                
                                # Calculate position
                                position = self.parse_position(
                                    self.position, main_width, main_height,
                                    target_width, target_height
                                )
                                
                                frame_position = 0
                            
                            # Read overlay frames
                            overlay_frames = []
                            frames_read = 0
                            total_frames = int(overlay_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            
                            for _ in range(overlay_frames_needed):
                                ret, frame = overlay_cap.read()
                                frames_read += 1
                                
                                if not ret:
                                    # If we run out of frames, loop back to beginning
                                    overlay_cap.release()
                                    overlay_cap = cv2.VideoCapture(overlay_path)
                                    frame_position = 0
                                    ret, frame = overlay_cap.read()
                                    if not ret:
                                        break
                                
                                # Resize frame to target size
                                frame = cv2.resize(frame, target_size)
                                
                                # Convert to tensor
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_tensor = self.processor.transform(Image.fromarray(frame))
                                overlay_frames.append(frame_tensor)
                            
                            # Update frame position for next batch
                            if frames_read < total_frames:
                                # Only update if we haven't looped
                                frame_position += frames_read
                            else:
                                # If we looped, calculate correct position
                                frame_position = frames_read % total_frames
                            
                            # Store state for next batch
                            active_overlays[segment_id] = (overlay_cap, frame_position, target_size, position)
                            
                            if overlay_frames:
                                # Stack frames into tensor
                                overlay_tensor = torch.stack(overlay_frames).to(self.processor.device)
                                
                                # Apply overlay
                                main_portion = self.processor.overlay_batch(
                                    main_portion, overlay_tensor, position
                                )
                                
                                # Update main batch with overlaid portion
                                main_batch[batch_start_idx:batch_end_idx] = main_portion
                                
                                # Free memory
                                del overlay_frames, overlay_tensor
                                torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Error processing overlay: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Check if any segments are completely processed
                    segments_to_remove = []
                    for segment_id, (cap, _, _, _) in active_overlays.items():
                        # Find the segment data
                        segment_data = None
                        for seg in frame_segments:
                            if seg["segment_id"] == segment_id:
                                segment_data = seg
                                break
                        
                        # Check if segment is fully processed
                        if segment_data and segment_data["end_frame"] <= batch_end:
                            segments_to_remove.append(segment_id)
                            cap.release()  # Release the video capture
                    
                    # Remove completed segments
                    for segment_id in segments_to_remove:
                        del active_overlays[segment_id]
                    
                    # Move segments that are completely processed
                    while (current_segment_idx < len(frame_segments) and 
                           frame_segments[current_segment_idx]["end_frame"] <= batch_end):
                        current_segment_idx += 1
                    
                    # Write batch to output
                    self.processor.write_batch_to_video(out, main_batch)
                    
                    # Update progress
                    current_frame = batch_end
                    pbar.update(len(main_batch))
                    
                    # Free memory
                    del main_batch
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Close files and release resources
            main_cap.release()
            out.release()
            
            # Clean up any remaining active overlays
            for segment_id, (cap, _, _, _) in active_overlays.items():
                cap.release()
            
            # Add audio if specified
            if self.audio_file and os.path.exists(self.audio_file):
                print(f"Adding audio track: {self.audio_file}")
                self._add_audio_track()
            
            print(f"Video composition complete: {self.output_file}")
            
        except Exception as e:
            print(f"Error in video composition: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_audio_track(self):
        """Add audio track to the composed video using ffmpeg"""
        try:
            import subprocess
            
            # Create temporary output file
            temp_output = self.output_file.replace('.mp4', '_temp.mp4')
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', self.output_file,  # Input video
                '-i', self.audio_file,   # Input audio
                '-c:v', 'copy',          # Copy video stream without re-encoding
                '-c:a', 'aac',           # Encode audio as AAC
                '-shortest',             # End when shortest stream ends
                temp_output
            ]
            
            # Add audio start offset if specified
            if self.audio_start > 0:
                cmd.insert(4, '-ss')
                cmd.insert(5, str(self.audio_start))
            
            # Run ffmpeg
            print("Running ffmpeg to add audio...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original file with audio-enhanced version
                os.replace(temp_output, self.output_file)
                print("Audio track added successfully")
            else:
                print(f"Error adding audio: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
        except Exception as e:
            print(f"Error in audio processing: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Compose video with PyTorch GPU acceleration and memory optimization')
    parser.add_argument('--json', required=True, help='Path to JSON file with matched videos')
    parser.add_argument('--transcript', required=True, help='Path to transcript file')
    parser.add_argument('--input-video', required=True, help='Path to input video')
    parser.add_argument('--output', default='composed_video_pytorch.mp4', help='Path to output video')
    parser.add_argument('--videos-dir', required=True, help='Directory containing pre-downloaded videos')
    parser.add_argument('--position', default='70%,50%', 
                       help='Position to insert video clips (center of overlay)')
    parser.add_argument('--size', default='54%', help='Size of inserted videos (preserves aspect ratio)')
    parser.add_argument('--distance-threshold', type=float, default=2.0, 
                       help='Maximum distance threshold for video inclusion')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing even if GPU is available')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of frames to process in each batch')
    parser.add_argument('--audio', help='Path to audio file to add to the final video')
    parser.add_argument('--audio-start', type=float, default=0.0, 
                       help='Start time for audio in seconds (default: 0.0)')

    args = parser.parse_args()
    
    # Create composer and process video
    composer = PyTorchVideoComposer(
        args.json, 
        args.transcript, 
        args.output, 
        args.position,
        args.size,
        args.distance_threshold,
        args.videos_dir,
        use_gpu=not args.cpu,
        batch_size=args.batch_size,
        audio_file=args.audio,
        audio_start=args.audio_start
    )
    
    # Find pre-downloaded videos
    found_videos = composer.find_videos()
    
    # Match videos to transcript
    matched_segments = composer.match_videos_to_transcript(found_videos)
    
    # Compose final video
    composer.compose_video(args.input_video, matched_segments)

if __name__ == "__main__":
    main()