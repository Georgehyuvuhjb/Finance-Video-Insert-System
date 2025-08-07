import json
import faiss
import pickle
import argparse
import os
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

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
            raise FileNotFoundError(f"Embedding metadata not found at {metadata_path}. Run video_vectorizer.py first.")
        
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
    
    def group_sentences_by_meaning(self, sentences, max_sentences_per_group=3, similarity_threshold=0.70):
        """
        Group semantically similar sentences together
        
        Args:
            sentences (list): List of sentences
            max_sentences_per_group (int): Maximum number of sentences per group
            similarity_threshold (float): Threshold for grouping sentences
            
        Returns:
            list: List of sentence groups
        """
        if not sentences:
            return []
        
        print(f"Grouping {len(sentences)} sentences by meaning...")
        
        # Generate sentence embeddings
        sentence_embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        if len(sentences) <= max_sentences_per_group:
            # If there are few sentences, return them as a single group
            return [{'sentences': sentences, 'start_index': 0, 'end_index': len(sentences) - 1}]
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                # Compute cosine similarity
                similarity_matrix[i][j] = np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (
                    np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[j])
                )
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Try to create clustering with compatibility for different scikit-learn versions
        try:
            # Try with distance_threshold first (newer scikit-learn versions)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - similarity_threshold,
                metric='precomputed',
                linkage='average'
            ).fit(distance_matrix)
        except TypeError:
            try:
                # Try without affinity for older versions
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1 - similarity_threshold,
                    metric='precomputed',
                    linkage='average'
                ).fit(distance_matrix)
            except TypeError:
                # Fall back to specifying number of clusters
                print("Warning: Using fixed number of clusters as distance_threshold is not supported")
                n_clusters = max(1, len(sentences) // max_sentences_per_group)
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage='average'
                ).fit(distance_matrix)
        
        # Group sentences by cluster label
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Sort clusters by position in the original text
        sorted_groups = []
        for indices in clusters.values():
            # Sort indices within each cluster
            sorted_indices = sorted(indices)
            
            # Split large clusters to respect max_sentences_per_group
            for i in range(0, len(sorted_indices), max_sentences_per_group):
                chunk = sorted_indices[i:i + max_sentences_per_group]
                sorted_groups.append({
                    'sentences': [sentences[idx] for idx in chunk],
                    'start_index': min(chunk),
                    'end_index': max(chunk)
                })
        
        # Sort groups by their position in the original text
        sorted_groups.sort(key=lambda x: x['start_index'])
        
        print(f"Created {len(sorted_groups)} sentence groups")
        return sorted_groups
    
    def find_matching_videos_for_group(self, sentence_group, top_k=2, metadata_path="video_metadata.json"):
        """
        Find matching videos for a sentence group
        
        Args:
            sentence_group (dict): Dict with 'sentences' key containing sentences
            top_k (int): Number of videos to return
            metadata_path (str): Path to video metadata
            
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
        
        # Format results
        results = []
        for distance, idx in combined:
            idx_str = str(int(idx))
            video_id = self.id_mapping[idx_str]
            
            # Get video details
            video_data = self.video_metadata.get(video_id, {})
            
            # Prepare thumbnail URL
            thumbnail_url = ""
            if "videos" in video_data and "small" in video_data["videos"]:
                thumbnail_url = video_data["videos"]["small"].get("thumbnail", "")
            
            results.append({
                "video_id": video_id,
                "distance": float(distance),  # Keep original distance value (smaller = more similar)
                "title": f"Video {video_id}",
                "tags": video_data.get("tags", "")[:100] + "..." if len(video_data.get("tags", "")) > 100 else video_data.get("tags", ""),
                "url": video_data.get("videos", {}).get("small", {}).get("url", ""),
                "thumbnail": thumbnail_url
            })
        
        return results
    
    def process_text(self, text, max_sentences_per_group=3, top_k=2, metadata_path="video_metadata.json"):
        """
        Process text by splitting into sentences, grouping, and finding matching videos
        
        Args:
            text (str): Input text
            max_sentences_per_group (int): Maximum sentences per group
            top_k (int): Number of videos to return per group
            metadata_path (str): Path to video metadata
            
        Returns:
            list: List of groups with matching videos
        """
        # Split text into sentences
        sentences = self.split_text_into_sentences(text)
        if not sentences:
            return []
        
        # Group sentences by meaning
        sentence_groups = self.group_sentences_by_meaning(
            sentences, max_sentences_per_group=max_sentences_per_group
        )
        
        # Find matching videos for each group
        results = []
        for i, group in enumerate(sentence_groups):
            print(f"Finding videos for group {i+1}/{len(sentence_groups)}...")
            matching_videos = self.find_matching_videos_for_group(
                group, top_k=top_k, metadata_path=metadata_path
            )
            
            results.append({
                "sentence_group": group,
                "matching_videos": matching_videos
            })
        
        return results

def format_results(results):
    """Format results for display"""
    output = []
    
    output.append("===== SENTENCE GROUPS AND MATCHING VIDEOS =====\n")
    
    for i, result in enumerate(results):
        group = result["sentence_group"]
        videos = result["matching_videos"]
        
        output.append(f"GROUP {i+1}:")
        output.append("-" * 40)
        
        # Display sentences in the group
        output.append("Sentences:")
        for j, sentence in enumerate(group["sentences"]):
            output.append(f"  {j+1}. {sentence}")
        
        # Display matching videos
        output.append("\nMatching Videos:")
        for j, video in enumerate(videos):
            output.append(f"  {j+1}. Video ID: {video['video_id']} (Distance: {video['distance']:.4f})")
            output.append(f"     URL: {video['url']}")
            output.append(f"     Tags: {video['tags']}")
        
        output.append("\n" + "=" * 50 + "\n")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='Find matching videos for groups of semantically similar sentences')
    parser.add_argument('--text', type=str, help='Input text for matching')
    parser.add_argument('--file', type=str, help='Path to file containing text for matching')
    parser.add_argument('--vectors', default='outputs/vectors', help='Directory containing vector data')
    parser.add_argument('--metadata', default='outputs/video_metadata.json', help='Path to video metadata')
    parser.add_argument('--max-group-size', type=int, default=3, help='Maximum sentences per group')
    parser.add_argument('--num-videos', type=int, default=2, help='Number of videos per group')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    parser.add_argument('--top-k', type=int, default=2, help='Number of top matching videos to return')

    args = parser.parse_args()
    
    # Get input text from argument or file
    if args.text:
        input_text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        # If no input provided, ask user interactively
        print("Please enter your text (press Ctrl+D or Ctrl+Z on a new line when finished):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        input_text = '\n'.join(lines)
    
    # Initialize matcher and process text
    matcher = SemanticVideoMatcher(args.vectors)
    results = matcher.process_text(
        input_text,
        max_sentences_per_group=args.max_group_size,
        top_k=args.top_k,
        metadata_path=args.metadata
    )
    
    # Format and display results
    formatted_results = format_results(results)
    print(formatted_results)
    
    # Save results to file if specified
    if args.output:
        # Save original formatted output
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_results)
            print(f"Results saved to {args.output}")
        
        # Also save JSON output with same base name but .json extension
        json_output = os.path.splitext(args.output)[0] + '.json'
        with open(json_output, 'w', encoding='utf-8') as f:
            # Convert results to serializable format
            serializable_results = []
            
            for group_result in results:
                group_data = group_result["sentence_group"]
                videos_data = []
                
                for video in group_result["matching_videos"]:
                    # Ensure all values are serializable
                    video_entry = {
                        "video_id": video["video_id"],
                        "distance": float(video["distance"]),
                        "title": video.get("title", f"Video {video['video_id']}"),
                        "tags": video.get("tags", ""),
                        "url": video.get("url", ""),
                        "thumbnail": video.get("thumbnail", "")
                    }
                    videos_data.append(video_entry)
                
                serializable_results.append({
                    "sentence_group": {
                        "sentences": group_data["sentences"],
                        "start_index": group_data.get("start_index", 0),
                        "end_index": group_data.get("end_index", 0)
                    },
                    "matching_videos": videos_data
                })
            
            # Write JSON with UTF-8 encoding and pretty print
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"JSON results saved to {json_output}")
    
    # Return a more structured JSON format if needed
    return results

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()