# video_vectorizer.py

import json
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import argparse


def create_video_vectors(metadata_path, output_dir="outputs/vectors", model_name="paraphrase-multilingual-MiniLM-L12-v2", batch_size=32):
    """
    Pre-compute and save vector representations of videos based on their tags.

    Args:
        metadata_path (str): Path to the video metadata JSON file
        output_dir (str): Directory to save the vectors and index
        model_name (str): Name of the sentence transformer model to use
        batch_size (int): Batch size for computing embeddings

    Returns:
        dict: Mapping of video IDs to their vector indices
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Loading video metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        video_metadata = json.load(f)

    print(f"Found {len(video_metadata)} videos in metadata file.")

    # Prepare video information for vectorization
    video_ids = []
    video_texts = []

    print("Preparing video data for vectorization...")
    for video_id, data in video_metadata.items():
        tags = data.get("tags", "")
        video_ids.append(video_id)
        video_texts.append(tags)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the video IDs and their metadata mapping
    id_mapping = {idx: video_id for idx, video_id in enumerate(video_ids)}
    with open(os.path.join(output_dir, "video_id_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(id_mapping, f, indent=2)

    # Compute embeddings in batches
    print("Computing video embeddings (this may take some time)...")
    start_time = time.time()

    all_embeddings = []
    for i in range(0, len(video_texts), batch_size):
        batch = video_texts[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=True)
        all_embeddings.append(embeddings)
        print(
            f"Processed batch {i//batch_size + 1}/{(len(video_texts)-1)//batch_size + 1}")

    # Concatenate all embeddings
    video_embeddings = np.vstack(all_embeddings)

    end_time = time.time()
    print(
        f"Embedding computation completed in {end_time - start_time:.2f} seconds.")

    # Save the raw embeddings
    print("Saving raw embeddings...")
    with open(os.path.join(output_dir, "video_embeddings.pkl"), 'wb') as f:
        pickle.dump(video_embeddings, f)

    # Create and save the FAISS index
    print("Creating FAISS index...")
    dimension = video_embeddings.shape[1]

    # Create two types of indexes for different needs

    # 1. Flat index for maximum precision
    index_flat = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(video_embeddings)  # Normalize for cosine similarity
    index_flat.add(video_embeddings)

    faiss.write_index(index_flat, os.path.join(
        output_dir, "video_index_flat.faiss"))
    print(
        f"Saved flat index to {os.path.join(output_dir, 'video_index_flat.faiss')}")

    # 2. HNSW index for faster search with slight precision trade-off
    index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node
    # Higher value = more accurate but slower construction
    index_hnsw.hnsw.efConstruction = 200
    index_hnsw.add(video_embeddings)

    faiss.write_index(index_hnsw, os.path.join(
        output_dir, "video_index_hnsw.faiss"))
    print(
        f"Saved HNSW index to {os.path.join(output_dir, 'video_index_hnsw.faiss')}")

    # Save metadata about the embeddings
    embedding_metadata = {
        "model_name": model_name,
        "dimension": dimension,
        "num_videos": len(video_ids),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(output_dir, "embedding_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(embedding_metadata, f, indent=2)

    print(
        f"Vector computation and indexing completed. All files saved to {output_dir}/")
    return id_mapping


def test_vector_search(output_dir="outputs/vectors", query="financial market stock trading", top_k=5):
    """
    Test the created vector index with a sample query

    Args:
        output_dir (str): Directory where vectors and index are saved
        query (str): Test query to search for
        top_k (int): Number of results to return
    """
    print(f"\nTesting vector search with query: '{query}'")

    # Load the model
    with open(os.path.join(output_dir, "embedding_metadata.json"), 'r') as f:
        metadata = json.load(f)

    model = SentenceTransformer(metadata["model_name"])

    # Encode the query
    query_vector = model.encode([query])[0]
    query_vector = query_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # Load the index
    index = faiss.read_index(os.path.join(
        output_dir, "video_index_flat.faiss"))

    # Load the ID mapping
    with open(os.path.join(output_dir, "video_id_mapping.json"), 'r') as f:
        id_mapping = json.load(f)

    # Search
    distances, indices = index.search(query_vector, top_k)

    print(f"\nTop {top_k} results for query '{query}':")
    for i, idx in enumerate(indices[0]):
        video_id = id_mapping[str(idx)]
        similarity = distances[0][i]
        print(f"{i+1}. Video ID: {video_id}, Similarity: {similarity:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Create and save vector representations for videos')
    parser.add_argument('--metadata', default='outputs/video_metadata.json',
                        help='Path to video metadata JSON file')
    parser.add_argument('--output', default='outputs/vectors',
                        help='Output directory for vectors and index')
    parser.add_argument('--model', default='paraphrase-multilingual-MiniLM-L12-v2',
                        help='Name of the sentence transformer model')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for computing embeddings')
    parser.add_argument('--test', action='store_true',
                        help='Run a test query after creating vectors')
    parser.add_argument(
        '--test-query', default='financial market stock trading', help='Query to use for testing')

    args = parser.parse_args()

    id_mapping = create_video_vectors(
        args.metadata, args.output, args.model, args.batch_size)

    if args.test:
        test_vector_search(args.output, args.test_query)


if __name__ == "__main__":
    main()
