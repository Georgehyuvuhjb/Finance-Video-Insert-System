import os
import json
import cv2
import torch
import argparse
import glob
from PIL import Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor


class VideoTagger:
    def __init__(self, model_path='Bllossom/llama-3.2-Korean-Bllossom-AICA-5B', device=None):
        """Initialize the video tagger with local MLLAMA model"""
        print(f"Loading model from {model_path}...")
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model with appropriate precision based on device
        if self.device == 'cuda':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map='auto',
            low_cpu_mem_usage=True  # Optimize memory usage
        )
        self.processor = MllamaProcessor.from_pretrained(model_path)
        print("Model loaded successfully")

    def extract_frames(self, video_path, num_frames=1):
        """Extract multiple frames from the video

        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract (default: 1)

        Returns:
            List of (image, frame_number) tuples
        """
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"Error: Could not determine frame count for {video_path}")
            video.release()
            return []

        # Adjust num_frames if it exceeds total frames
        num_frames = min(num_frames, total_frames)

        # Calculate frame positions to extract
        if num_frames == 1:
            # For a single frame, take the middle frame
            frame_positions = [total_frames // 2]
        else:
            # For multiple frames, distribute evenly across the video
            step = total_frames / num_frames
            frame_positions = [int(i * step) for i in range(num_frames)]

            # Ensure we don't exceed total_frames
            frame_positions = [min(pos, total_frames - 1)
                               for pos in frame_positions]

        extracted_frames = []

        # Extract each frame
        for frame_pos in frame_positions:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            success, frame = video.read()

            if success:
                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                extracted_frames.append((image, frame_pos))
            else:
                print(
                    f"Warning: Failed to extract frame at position {frame_pos}")

        video.release()
        return extracted_frames

    def get_image_tags(self, image, debug=True):
        """Get tags for the image using the local MLLAMA model with improved prompt"""

        # Use a very specific prompt that forces tag-only output
        prompt_template = "List highly related single-word tags for this image, separated by commas ONLY. Response must only contain tags. No other text."

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt_template}
                ]
            },
        ]

        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # Process inputs
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate output with optimized parameters
        with torch.inference_mode():  # Use inference mode for memory efficiency
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                eos_token_id=self.processor.tokenizer.convert_tokens_to_ids(
                    '<|eot_id|>'),
                use_cache=True
            )

        # Decode output
        result = self.processor.decode(output[0])

        # Print the raw model output for debugging
        if debug:
            print("\n--- MODEL RAW OUTPUT ---")
            print(result)
            print("------------------------\n")

        # Completely revised tag extraction approach
        try:
            # First look for assistant's response which typically contains the tags
            if "<|start_header_id|>assistant<|end_header_id|>" in result:
                parts = result.split(
                    "<|start_header_id|>assistant<|end_header_id|>")
                if len(parts) > 1:
                    assistant_response = parts[1].strip()
                    if debug:
                        print(f"Assistant response: '{assistant_response}'")
                else:
                    assistant_response = result
            else:
                assistant_response = result

            # Look for comma-separated content in the response
            if "," in assistant_response:
                # Find all text segments with commas - these are likely the tags
                potential_tag_segments = []
                lines = assistant_response.split("\n")
                for line in lines:
                    if "," in line:
                        potential_tag_segments.append(line)

                if potential_tag_segments:
                    # Use the segment with the most commas (likely the tag list)
                    best_segment = max(potential_tag_segments,
                                       key=lambda x: x.count(","))
                    # Clean up the segment
                    cleaned_segment = best_segment

                    # Remove common tokens
                    tokens_to_remove = ["<|eot_id|>", "<|end_header_id|>"]
                    for token in tokens_to_remove:
                        cleaned_segment = cleaned_segment.replace(token, "")

                    if debug:
                        print(f"Best tag segment: '{cleaned_segment}'")

                    # Split by comma and clean each tag
                    raw_tags = [tag.strip()
                                for tag in cleaned_segment.split(",")]

                    # Final cleanup of each tag
                    cleaned_tags = []
                    for tag in raw_tags:
                        tag = tag.strip("'\" [](){}-.")
                        # Only add non-empty tags with at least 2 chars
                        if tag and len(tag) > 1:
                            # Convert to lowercase for consistency
                            cleaned_tags.append(tag.lower())

                    if debug:
                        print(f"Extracted tags: {cleaned_tags}")

                    return cleaned_tags
                else:
                    if debug:
                        print("No comma-separated segments found")
            else:
                # If no commas found, try to extract words as tags
                words = assistant_response.split()
                cleaned_words = [word.strip(
                    "'\"[](){}-.,:;<>").lower() for word in words]
                cleaned_tags = [word for word in cleaned_words if len(
                    word) > 2 and not word.startswith("<|")]

                if debug:
                    print(
                        f"No commas found, using words as tags: {cleaned_tags[:10]}")

                return cleaned_tags[:10]  # Return up to 10 words as tags

            return []

        except Exception as e:
            print(f"Warning: Could not parse tags from result")
            print(f"Error: {str(e)}")

            # Last resort - grab anything that looks like words between commas or spaces
            try:
                # Remove special tokens and get all alphanumeric words
                clean_text = result
                for token in ["<|eot_id|>", "<|end_header_id|>", "<|start_header_id|>", "user", "assistant"]:
                    clean_text = clean_text.replace(token, " ")

                words = [word.strip("'\"[](){}-.,:;<>").lower()
                         for word in clean_text.split()]
                valid_words = [
                    word for word in words if word.isalnum() and len(word) > 2]

                if debug:
                    print(f"Last resort tags: {valid_words[:10]}")

                return valid_words[:10]  # Return up to 10 words as tags
            except:
                return []  # Return empty list if all parsing fails


def process_tags(existing_tags, new_tags):
    """
    Process and merge tags properly

    Args:
        existing_tags: Existing tags (string with comma-separated tags or list of tags)
        new_tags: List of new tags to add

    Returns:
        List of merged unique tags
    """
    # Initialize list of existing tags
    processed_existing_tags = []

    # Handle different formats of existing tags
    if isinstance(existing_tags, list):
        # If existing_tags is already a list, use it directly
        processed_existing_tags = existing_tags
    elif isinstance(existing_tags, str):
        # If it's a comma-separated string, split by commas
        processed_existing_tags = [tag.strip().lower()
                                   for tag in existing_tags.split(',')]

    # Filter out very short and empty tags
    processed_existing_tags = [
        tag for tag in processed_existing_tags if tag and len(tag) > 1]

    # Add new tags, avoiding duplicates
    final_tags = processed_existing_tags.copy()
    for new_tag in new_tags:
        new_tag = new_tag.lower().strip()
        if new_tag and len(new_tag) > 1 and new_tag not in processed_existing_tags:
            final_tags.append(new_tag)

    return final_tags


def load_metadata_database(metadata_path="outputs/video_metadata.json"):
    """
    Load the video metadata database

    Args:
        metadata_path: Path to the metadata JSON file

    Returns:
        Dictionary of video metadata or empty dict if file not found
    """
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Metadata file not found: {metadata_path}")
            return {}
    except Exception as e:
        print(f"Error loading metadata database: {str(e)}")
        return {}


def save_metadata_database(metadata_db, metadata_path="outputs/video_metadata.json"):
    """
    Save the updated metadata database

    Args:
        metadata_db: The metadata database dictionary
        metadata_path: Path to the metadata JSON file
    """
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_db, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved updated metadata to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata database: {str(e)}")


def parse_video_filename(filename):
    """
    Parse the video filename to get video ID and quality

    Args:
        filename: Video filename in format ID_keyword_quality.mp4

    Returns:
        Tuple of (video_id, quality) or (None, None) if parsing fails
    """
    try:
        basename = os.path.basename(filename)
        name_parts = basename.split('.')[0].split('_')
        if len(name_parts) >= 3:
            video_id = name_parts[0]
            quality = name_parts[-1]
            return video_id, quality
        return None, None
    except Exception:
        return None, None


def process_videos(videos_dir="videos", metadata_path="outputs/video_metadata.json", model_path='Bllossom/llama-3.2-Korean-Bllossom-AICA-5B',
                   device='auto', num_frames=1, debug=True, specific_video=None):
    """
    Process all videos in the directory or a specific video if provided

    Args:
        videos_dir: Directory containing the videos
        metadata_path: Path to the metadata JSON file
        model_path: Path to the MLLAMA model
        device: Device to use for inference
        num_frames: Number of frames to extract and analyze per video
        debug: Enable debug output
        specific_video: Optional specific video ID to process
    """
    # Initialize the tagger
    tagger = VideoTagger(model_path=model_path, device=device)

    # Load the metadata database
    metadata_db = load_metadata_database(metadata_path)
    if not metadata_db:
        print("No metadata available. Please download videos first.")
        return

    # Find videos to process
    if specific_video:
        video_files = []
        for file in os.listdir(videos_dir):
            if file.endswith(".mp4") and file.startswith(f"{specific_video}_"):
                video_files.append(os.path.join(videos_dir, file))
    else:
        video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

    print(f"Found {len(video_files)} video files to process")

    # Process each video
    for i, video_path in enumerate(video_files):
        video_id, quality = parse_video_filename(video_path)

        if not video_id or video_id not in metadata_db:
            print(
                f"\nSkipping video {video_path} - could not match to metadata")
            continue

        print(
            f"\nProcessing video {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"Video ID: {video_id}, Quality: {quality}")

        # Extract existing metadata
        video_metadata = metadata_db[video_id]

        # Extract existing tags
        existing_tags = video_metadata.get("tags", "")
        print(f"Existing tags: {existing_tags}")

        # Extract frames from the video
        frames = tagger.extract_frames(video_path, num_frames=num_frames)
        if not frames:
            print(f"Warning: Could not extract frames from video {video_path}")
            continue

        # Process each frame and collect all tags
        all_new_tags = []
        for frame_idx, (frame, frame_number) in enumerate(frames):
            print(
                f"Analyzing frame {frame_idx+1}/{len(frames)} (position: {frame_number})...")

            # Get tags for the frame
            frame_tags = tagger.get_image_tags(frame, debug=debug)
            print(
                f"Generated {len(frame_tags)} tags from frame {frame_idx+1}: {frame_tags}")

            # Add to all tags
            all_new_tags.extend(frame_tags)

        # Remove duplicates from all_new_tags while preserving order
        unique_new_tags = []
        seen_tags = set()
        for tag in all_new_tags:
            if tag.lower() not in seen_tags:
                unique_new_tags.append(tag)
                seen_tags.add(tag.lower())

        print(
            f"Total unique tags generated from all frames: {len(unique_new_tags)}")
        print(f"Combined new tags: {unique_new_tags}")

        # Process and merge tags
        combined_tags = process_tags(existing_tags, unique_new_tags)
        print(f"Final combined tags: {combined_tags}")

        # Update the metadata with the new tags
        metadata_db[video_id]["tags"] = ", ".join(combined_tags)

        print(
            f"Updated metadata for video ID {video_id} with {len(unique_new_tags)} new tags ({len(combined_tags)} total tags)")

        # Free up memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save the updated metadata database
    save_metadata_database(metadata_db, metadata_path)
    print("\nAll videos processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos and update tags in metadata database")
    parser.add_argument("--videos_dir", default="outputs/videos",
                        help="Directory containing video files (default: videos)")
    parser.add_argument("--metadata", default="outputs/video_metadata.json",
                        help="Path to metadata JSON file (default: outputs/video_metadata.json)")
    parser.add_argument("--model_path", default="Bllossom/llama-3.2-Korean-Bllossom-AICA-5B",
                        help="Path to model or model ID on Hugging Face")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                        help="Device to use for inference (default: auto)")
    parser.add_argument("--frames", type=int, default=1,
                        help="Number of frames to extract and analyze per video (default: 1)")
    parser.add_argument("--video_id",
                        help="Process only a specific video ID")
    parser.add_argument("--smaller_model", action="store_true",
                        help="Use a smaller model for faster processing")
    parser.add_argument("--no_debug", action="store_true",
                        help="Disable debug output from model responses")

    args = parser.parse_args()

    # If smaller model option is selected, use a lighter model
    if args.smaller_model:
        model_path = "joaogante/phi-2-vision"  # A smaller alternative model
        print("Using smaller model for faster processing")
    else:
        model_path = args.model_path

    process_videos(
        videos_dir=args.videos_dir,
        metadata_path=args.metadata,
        model_path=model_path,
        device=args.device,
        num_frames=args.frames,
        debug=not args.no_debug,
        specific_video=args.video_id
    )
