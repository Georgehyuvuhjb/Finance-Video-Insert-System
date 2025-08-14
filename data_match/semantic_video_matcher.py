import json
import faiss
import pickle
import argparse
import os
import numpy as np
import re
from sentence_transformers import SentenceTransformer


class SemanticVideoMatcher:
    """A class that groups sentences by meaning and finds matching videos"""

    def __init__(self, vectors_dir="outputs/vectors", model_name=None):
        """
        Initialize the matcher with pre-computed vectors

        Args:
            vectors_dir (str): Directory containing vector data
            model_name (str): Optional model name to override the one in metadata
        """
        self.vectors_dir = vectors_dir

        # Load embedding metadata
        metadata_path = os.path.join(vectors_dir, "embedding_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Embedding metadata not found at {metadata_path}. Run video_vectorizer.py first.")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # Load model
        self.model_name = model_name if model_name else self.metadata["model_name"]
        print(f"Loading model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)

        # Load ID mapping
        mapping_path = os.path.join(vectors_dir, "video_id_mapping.json")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.id_mapping = json.load(f)

        # Load FAISS index (using HNSW for speed)
        index_path = os.path.join(vectors_dir, "video_index_hnsw.faiss")
        if not os.path.exists(index_path):
            # Fall back to flat index if HNSW not available
            index_path = os.path.join(vectors_dir, "video_index_flat.faiss")

        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)

        # Load video metadata
        self.video_metadata = None  # Will be loaded on demand

    def load_video_metadata(self, metadata_path="video_metadata.json"):
        """Load video metadata from JSON file"""
        if self.video_metadata is None:
            print(f"Loading video metadata from {metadata_path}...")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.video_metadata = json.load(f)

    def split_text_into_sentences(self, text):
        """
        Split text into sentences (supports multiple languages including Chinese)

        Args:
            text (str): Input text

        Returns:
            list: List of sentences
        """
        # Split by common sentence ending punctuation
        sentences = re.split(r'([。！？\.\!?])', text)

        # Rejoin sentence endings with their sentences
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
            else:
                result.append(sentences[i])

        # Add any remaining content
        if len(sentences) % 2 == 1:
            result.append(sentences[-1])

        # Filter out empty sentences
        result = [s.strip() for s in result if s.strip()]

        # If the text contains commas but no periods, try splitting by commas
        if len(result) <= 1 and '，' in text:
            result = [s.strip() for s in text.split('，') if s.strip()]

        return result

    def group_sentences_by_meaning(self, sentences, max_sentences_per_group=3, similarity_threshold=0.5):
        """
        Group semantically similar adjacent sentences together

        Args:
            sentences (list): List of sentences
            max_sentences_per_group (int): Maximum number of sentences per group
            similarity_threshold (float): Threshold for grouping sentences

        Returns:
            list: List of sentence groups
        """
        if not sentences:
            return []

        print(
            f"Grouping {len(sentences)} sentences by adjacent meaning (threshold: {similarity_threshold})...")

        # Generate sentence embeddings
        sentence_embeddings = self.model.encode(
            sentences, show_progress_bar=False)

        if len(sentences) <= max_sentences_per_group:
            # If there are few sentences, return them as a single group
            return [{'sentences': sentences, 'start_index': 0, 'end_index': len(sentences) - 1}]

        # Group adjacent sentences by similarity
        groups = []
        current_group = [0]  # Start with first sentence

        for i in range(1, len(sentences)):
            # Check similarity with the last sentence in current group
            last_sentence_idx = current_group[-1]

            # Compute cosine similarity between current sentence and last sentence in group
            similarity = np.dot(sentence_embeddings[i], sentence_embeddings[last_sentence_idx]) / (
                np.linalg.norm(
                    sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[last_sentence_idx])
            )

            # Debug output
            print(
                f"  Sentence {i} vs {last_sentence_idx}: similarity = {similarity:.3f}")

            # If similar enough and group not too large, add to current group
            if similarity >= similarity_threshold and len(current_group) < max_sentences_per_group:
                current_group.append(i)
                print(
                    f"    -> Added to current group (size: {len(current_group)})")
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                    print(
                        f"    -> Started new group. Previous group had {len(current_group)} sentences")
                current_group = [i]

        # Add the last group
        if current_group:
            groups.append(current_group)

        # Convert to the expected format
        sorted_groups = []
        for indices in groups:
            sorted_groups.append({
                'sentences': [sentences[idx] for idx in indices],
                'start_index': min(indices),
                'end_index': max(indices)
            })

        print(
            f"Created {len(sorted_groups)} sentence groups (adjacent grouping)")
        return sorted_groups

    def find_matching_videos_for_group(self, sentence_group, top_k=2, metadata_path="video_metadata.json", distance_threshold=2.0):
        """
        Find matching videos for a sentence group

        Args:
            sentence_group (dict): Dict with 'sentences' key containing sentences
            top_k (int): Number of videos to return
            metadata_path (str): Path to video metadata
            distance_threshold (float): Maximum distance threshold for filtering videos

        Returns:
            list: List of matching videos
        """
        # Ensure metadata is loaded
        self.load_video_metadata(metadata_path)

        # Combine all sentences in the group
        combined_text = " ".join(sentence_group['sentences'])

        # Encode the combined text
        query_vector = self.model.encode([combined_text])[0]
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Search for similar videos - FAISS returns distances (smaller = more similar)
        distances, indices = self.index.search(query_vector, top_k)

        # Sort results by distance (ascending order - smallest distance first)
        combined = list(zip(distances[0], indices[0]))
        combined.sort(key=lambda x: x[0])  # Sort by distance (ascending)

        # Format results and apply distance threshold filtering
        results = []
        filtered_count = 0

        for distance, idx in combined:
            # Apply distance threshold filter
            if distance > distance_threshold:
                filtered_count += 1
                continue

            idx_str = str(int(idx))
            video_id = self.id_mapping[idx_str]

            # Get video details
            video_data = self.video_metadata.get(video_id, {})

            # Prepare thumbnail URL
            thumbnail_url = ""
            if "videos" in video_data and "small" in video_data["videos"]:
                thumbnail_url = video_data["videos"]["small"].get(
                    "thumbnail", "")

            results.append({
                "video_id": video_id,
                # Keep original distance value (smaller = more similar)
                "distance": float(distance),
                "title": f"Video {video_id}",
                "tags": video_data.get("tags", "")[:100] + "..." if len(video_data.get("tags", "")) > 100 else video_data.get("tags", ""),
                "url": video_data.get("videos", {}).get("small", {}).get("url", ""),
                "thumbnail": thumbnail_url
            })

        if filtered_count > 0:
            print(
                f"  Filtered out {filtered_count} videos with distance > {distance_threshold}")

        if not results:
            print(
                f"  No videos found within distance threshold {distance_threshold}")

        return results

    def parse_timestamp_text(self, text):
        """
        Parse text with timestamp format: start_time_end_time_sentence

        Args:
            text (str): Input text with timestamps

        Returns:
            list: List of sentences with timing information
        """
        lines = text.strip().split('\n')
        sentences_with_timing = []

        for line in lines:
            if not line.strip():
                continue

            # Parse format: MM:SS.ms_MM:SS.ms_sentence
            parts = line.split('_', 2)
            if len(parts) >= 3:
                start_time = parts[0]
                end_time = parts[1]
                sentence = parts[2]

                sentences_with_timing.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "sentence": sentence,
                    "duration": self.calculate_duration(start_time, end_time)
                })
            else:
                # If no timestamp, treat as regular sentence
                sentences_with_timing.append({
                    "start_time": None,
                    "end_time": None,
                    "sentence": line,
                    "duration": None
                })

        return sentences_with_timing

    def calculate_duration(self, start_time, end_time):
        """
        Calculate duration between two timestamps in MM:SS.ms format

        Args:
            start_time (str): Start time in MM:SS.ms format
            end_time (str): End time in MM:SS.ms format

        Returns:
            str: Duration in MM:SS.ms format
        """
        try:
            # Parse start time
            start_parts = start_time.split(':')
            start_minutes = int(start_parts[0])
            start_seconds_parts = start_parts[1].split('.')
            start_seconds = int(start_seconds_parts[0])
            start_ms = int(start_seconds_parts[1]) if len(
                start_seconds_parts) > 1 else 0

            # Parse end time
            end_parts = end_time.split(':')
            end_minutes = int(end_parts[0])
            end_seconds_parts = end_parts[1].split('.')
            end_seconds = int(end_seconds_parts[0])
            end_ms = int(end_seconds_parts[1]) if len(
                end_seconds_parts) > 1 else 0

            # Calculate total milliseconds
            start_total_ms = (start_minutes * 60 +
                              start_seconds) * 100 + start_ms
            end_total_ms = (end_minutes * 60 + end_seconds) * 100 + end_ms

            duration_ms = end_total_ms - start_total_ms

            # Convert back to MM:SS.ms format
            duration_seconds = duration_ms // 100
            duration_ms_remainder = duration_ms % 100
            duration_minutes = duration_seconds // 60
            duration_seconds_remainder = duration_seconds % 60

            return f"{duration_minutes:02d}:{duration_seconds_remainder:02d}.{duration_ms_remainder:02d}"

        except (ValueError, IndexError):
            return "00:00.00"

    def process_text(self, text, max_sentences_per_group=3, top_k=2, metadata_path="video_metadata.json", has_timestamps=False, similarity_threshold=0.5, distance_threshold=2.0):
        """
        Process text by splitting into sentences, grouping, and finding matching videos

        Args:
            text (str): Input text
            max_sentences_per_group (int): Maximum sentences per group
            top_k (int): Number of videos to return per group
            metadata_path (str): Path to video metadata
            has_timestamps (bool): Whether the text has timestamp format
            similarity_threshold (float): Threshold for grouping adjacent sentences
            distance_threshold (float): Maximum distance threshold for video matching

        Returns:
            list: List of groups with matching videos
        """
        if has_timestamps:
            # Parse timestamped text
            sentences_with_timing = self.parse_timestamp_text(text)
            sentences = [item["sentence"] for item in sentences_with_timing]
        else:
            # Split text into sentences normally
            sentences = self.split_text_into_sentences(text)
            sentences_with_timing = [{"sentence": s, "start_time": None, "end_time": None, "duration": None}
                                     for s in sentences]

        if not sentences:
            return []

        # Group sentences by meaning
        sentence_groups = self.group_sentences_by_meaning(
            sentences, max_sentences_per_group=max_sentences_per_group, similarity_threshold=similarity_threshold
        )

        # Find matching videos for each group and add timing information
        results = []
        for i, group in enumerate(sentence_groups):
            print(f"Finding videos for group {i+1}/{len(sentence_groups)}...")
            matching_videos = self.find_matching_videos_for_group(
                group, top_k=top_k, metadata_path=metadata_path, distance_threshold=distance_threshold
            )

            # Calculate group timing if timestamps are available
            group_timing = self.calculate_group_timing(
                group, sentences_with_timing)

            results.append({
                "group_id": i + 1,
                "sentence_group": group,
                "matching_videos": matching_videos,
                "timing": group_timing
            })

        return results

    def calculate_group_timing(self, group, sentences_with_timing):
        """
        Calculate timing information for a sentence group

        Args:
            group (dict): Sentence group
            sentences_with_timing (list): List of sentences with timing info

        Returns:
            dict: Timing information for the group
        """
        group_sentences = group["sentences"]

        # Find timing info for sentences in this group
        group_timings = []
        for group_sentence in group_sentences:
            for timing_info in sentences_with_timing:
                if timing_info["sentence"] == group_sentence:
                    group_timings.append(timing_info)
                    break

        if not group_timings or not any(t["start_time"] for t in group_timings):
            return {
                "start_time": None,
                "end_time": None,
                "duration": None
            }

        # Get start time from first sentence and end time from last sentence
        valid_timings = [t for t in group_timings if t["start_time"]]
        if not valid_timings:
            return {
                "start_time": None,
                "end_time": None,
                "duration": None
            }

        start_time = valid_timings[0]["start_time"]
        end_time = valid_timings[-1]["end_time"]
        duration = self.calculate_duration(start_time, end_time)

        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        }


def format_results(results):
    """Format results for display"""
    output = []

    output.append("===== SENTENCE GROUPS AND MATCHING VIDEOS =====\n")

    for i, result in enumerate(results):
        group = result["sentence_group"]
        videos = result["matching_videos"]
        timing = result.get("timing", {})

        output.append(f"GROUP {i+1}:")
        output.append("-" * 40)

        # Display timing information if available
        if timing.get("start_time"):
            output.append(
                f"Time: {timing['start_time']} - {timing['end_time']}, Duration: {timing['duration']}")
            output.append("")

        # Display sentences in the group
        output.append("Sentences:")
        for j, sentence in enumerate(group["sentences"]):
            output.append(f"  {j+1}. {sentence}")

        # Display matching videos
        if videos:
            output.append("\nMatching Videos:")
            for j, video in enumerate(videos):
                output.append(
                    f"  {j+1}. Video ID: {video['video_id']} (Distance: {video['distance']:.4f})")
                output.append(f"     URL: {video['url']}")
                output.append(f"     Tags: {video['tags']}")
        else:
            output.append(
                "\nNo matching videos found within distance threshold.")

        output.append("\n" + "=" * 50 + "\n")

    return "\n".join(output)


def format_recommendations(results):
    """Format results in Chinese for recommendations display"""
    output = []

    output.append("推薦插入點分析")
    output.append("================\n")

    for i, result in enumerate(results):
        group = result["sentence_group"]
        videos = result["matching_videos"]
        timing = result.get("timing", {})
        group_id = result.get("group_id", i + 1)

        # Display timing information
        if timing.get("start_time"):
            output.append(
                f"句子群組 {group_id} (時間: {timing['start_time']} - {timing['end_time']}, 時長: {timing['duration']}):")
        else:
            output.append(f"句子群組 {group_id}:")

        output.append("句子內容:")
        for sentence in group["sentences"]:
            output.append(f"- {sentence}")

        if videos:
            output.append("\n推薦影片:")
            for j, video in enumerate(videos):
                distance_score = video['distance']  # Use distance directly
                output.append(
                    f"{j+1}. {video['video_id']} (distance: {distance_score:.4f})")
                if video.get('tags'):
                    output.append(f"   標籤: {video['tags']}")
        else:
            output.append("\n無符合距離閾值的推薦影片")

        if timing.get("duration"):
            output.append(f"\n可用時長: {timing['duration']}")

        output.append("\n---\n")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Find matching videos for groups of semantically similar sentences')
    parser.add_argument('--text', type=str, help='Input text for matching')
    parser.add_argument('--file', type=str,
                        help='Path to file containing text for matching')
    parser.add_argument('--vectors', default='outputs/vectors',
                        help='Directory containing vector data')
    parser.add_argument(
        '--metadata', default='outputs/video_metadata.json', help='Path to video metadata')
    parser.add_argument('--max-group-size', type=int,
                        default=3, help='Maximum sentences per group')
    parser.add_argument('--num-videos', type=int, default=2,
                        help='Number of videos per group')
    parser.add_argument('--output', type=str,
                        help='Output file path (optional)')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Number of top matching videos to return')
    parser.add_argument('--similarity-threshold', type=float, default=0.5,
                        help='Similarity threshold for grouping adjacent sentences (default: 0.5)')
    parser.add_argument('--distance-threshold', type=float, default=2.0,
                        help='Maximum distance threshold for video matching - videos with higher distance will be filtered out (default: 2.0)')
    parser.add_argument('--show-recommendations', action='store_true',
                        help='Show recommendations in Chinese format')

    args = parser.parse_args()

    # Get input text from argument or file
    if args.text:
        input_text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        # If no input provided, ask user interactively
        print(
            "Please enter your text (press Ctrl+D or Ctrl+Z on a new line when finished):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        input_text = '\n'.join(lines)

    # Detect if input has timestamp format
    has_timestamps = bool(
        re.search(r'\d{2}:\d{2}\.\d{2}_\d{2}:\d{2}\.\d{2}_', input_text))
    if has_timestamps:
        print("Detected timestamp format in input text")

    # Initialize matcher and process text
    matcher = SemanticVideoMatcher(args.vectors)
    results = matcher.process_text(
        input_text,
        max_sentences_per_group=args.max_group_size,
        top_k=args.top_k,
        metadata_path=args.metadata,
        has_timestamps=has_timestamps,
        similarity_threshold=args.similarity_threshold,
        distance_threshold=args.distance_threshold
    )

    # Format and display results
    if args.show_recommendations:
        formatted_results = format_recommendations(results)
    else:
        formatted_results = format_results(results)

    print(formatted_results)

    # Save results to file if specified
    if args.output:
        # Save formatted output
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_results)
            print(f"Results saved to {args.output}")

        # Also save enhanced JSON output with timing information
        json_output = os.path.splitext(args.output)[0] + '.json'
        json_data = {
            "sentence_groups": []
        }

        for group_result in results:
            group_data = group_result["sentence_group"]
            videos_data = []
            timing = group_result.get("timing", {})

            for video in group_result["matching_videos"]:
                # Convert distance to similarity score
                similarity_score = 1.0 - video["distance"]

                video_entry = {
                    "video_id": video["video_id"],
                    "similarity_score": round(similarity_score, 4),
                    "distance": round(float(video["distance"]), 4),
                    "tags": video.get("tags", "").split(", ") if video.get("tags") else [],
                    "description": video.get("title", f"Video {video['video_id']}"),
                    "url": video.get("url", ""),
                    "thumbnail": video.get("thumbnail", "")
                }
                videos_data.append(video_entry)

            group_entry = {
                "group_id": group_result.get("group_id", 1),
                "start_time": timing.get("start_time"),
                "end_time": timing.get("end_time"),
                "duration": timing.get("duration"),
                "sentences": group_data["sentences"],
                "recommended_videos": videos_data
            }

            json_data["sentence_groups"].append(group_entry)

        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"Enhanced JSON results saved to {json_output}")

    return results


if __name__ == "__main__":
    main()
