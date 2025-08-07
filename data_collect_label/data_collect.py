import os
import time
import json
import requests
import sys
import argparse
import yaml

class PixabayVideoDownloader:
    def __init__(self, api_key, search_term="finance", num_videos=10):
        """
        Initialize the downloader with API key and search parameters.
        The metadata will be saved into a single JSON file.
        """
        self.api_key = api_key
        self.search_term = search_term
        self.num_videos = num_videos
        self.download_dir = "outputs/videos"
        # Define the path for the consolidated metadata database file
        self.metadata_db_path = "outputs/video_metadata.json"
        self.api_url = "https://pixabay.com/api/videos/"
        
        # Create download directory if it doesn't exist
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            print(f"Created directory: {self.download_dir}")
    
    def search_videos(self):
        """
        Search for videos using Pixabay API
        """
        print(f"Searching for '{self.search_term}' videos on Pixabay...")
        
        # Calculate how many pages we need to fetch
        videos_per_page = min(200, self.num_videos)  # Maximum 200 per page as per API docs
        pages_needed = (self.num_videos + videos_per_page - 1) // videos_per_page
        
        all_videos = []
        
        for page in range(1, pages_needed + 1):
            # Prepare API parameters
            params = {
                'key': self.api_key,
                'q': self.search_term,
                'page': page,
                'per_page': videos_per_page,
                'safesearch': 'true',
                'lang': 'en',
                'order': 'popular'
            }
            
            try:
                # Make API request
                response = requests.get(self.api_url, params=params)
                
                remaining = response.headers.get('X-RateLimit-Remaining')
                if remaining:
                    print(f"API calls remaining: {remaining}")
                
                if response.status_code == 200:
                    data = response.json()
                    videos = data.get('hits', [])
                    
                    print(f"Found {len(videos)} videos on page {page}")
                    all_videos.extend(videos)
                    
                    if len(all_videos) >= self.num_videos or len(videos) < videos_per_page:
                        break
                        
                else:
                    print(f"API request failed with status code: {response.status_code}")
                    print(f"Response: {response.text}")
                    break
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error making API request: {str(e)}")
                break
        
        return all_videos[:self.num_videos]

    def load_metadata_database(self):
        """
        Loads the existing metadata database from the JSON file.
        If the file doesn't exist or is invalid, returns an empty dictionary.
        """
        if os.path.exists(self.metadata_db_path):
            try:
                with open(self.metadata_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not read or parse {self.metadata_db_path}. A new file will be created.")
                return {}
        return {}

    def save_metadata_database(self, database):
        """
        Saves the updated metadata database to the JSON file.
        """
        try:
            with open(self.metadata_db_path, 'w', encoding='utf-8') as f:
                json.dump(database, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved updated metadata to {self.metadata_db_path}")
        except IOError as e:
            print(f"Error: Could not write to {self.metadata_db_path}. Reason: {e}")

    def download_videos(self, videos):
        """
        Download videos and update the single metadata JSON file.
        """
        print(f"Starting to download {len(videos)} videos...")
        
        # Load the entire metadata database into memory
        metadata_database = self.load_metadata_database()
        new_videos_added = 0
        
        for i, video_data in enumerate(videos):
            try:
                video_id = video_data.get('id')
                if not video_id:
                    print(f"Skipping video {i+1} due to missing ID.")
                    continue

                # Check if the video already exists in our database
                if str(video_id) in metadata_database:
                    print(f"Video {i+1}/{len(videos)} (ID: {video_id}) already in database. Skipping download.")
                    continue

                # --- Video Download Logic (unchanged) ---
                quality_options = ['large', 'medium', 'small', 'tiny']
                video_url, video_size, selected_quality = None, None, None
                
                for quality in quality_options:
                    if quality in video_data['videos'] and video_data['videos'][quality]['url']:
                        video_url = video_data['videos'][quality]['url']
                        video_size = video_data['videos'][quality]['size']
                        selected_quality = quality
                        break
                
                if not video_url:
                    print(f"No downloadable URL found for video {i+1} (ID: {video_id})")
                    continue
                
                filename = f"{video_id}_{self.search_term}_{selected_quality}.mp4"
                filepath = os.path.join(self.download_dir, filename)
                
                print(f"Downloading video {i+1}/{len(videos)}: {filename} ({video_size / (1024 * 1024):.2f} MB)")
                
                response = requests.get(video_url, stream=True)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Successfully downloaded: {filename}")
                else:
                    print(f"Failed to download video {i+1}. Status code: {response.status_code}")
                    continue # Skip metadata update if download fails

                # --- Metadata Update Logic (MODIFIED) ---
                # Add the new video's metadata to the in-memory database
                metadata_database[str(video_id)] = video_data
                new_videos_added += 1
                
                time.sleep(1)
            
            except Exception as e:
                print(f"An error occurred while processing video {i+1}: {str(e)}")

        # Save the updated database back to the file once, after the loop
        if new_videos_added > 0:
            print(f"Added {new_videos_added} new video(s) to the database.")
            self.save_metadata_database(metadata_database)
        else:
            print("No new videos were added to the database.")
    
    def run(self):
        """
        Run the complete video download process
        """
        print(f"Starting Pixabay API video downloader for '{self.search_term}' videos...")
        videos = self.search_videos()
        
        if videos:
            print(f"Found {len(videos)} videos matching the search term.")
            self.download_videos(videos)
            print(f"\nProcess complete. Videos are in '{self.download_dir}' folder.")
            print(f"All metadata is stored in '{self.metadata_db_path}'.")
        else:
            print(f"No new videos found for search term '{self.search_term}'.")


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        else:
            print(f"Config file not found: {config_path}")
            config = {"pixabay": {"api_key": "YOUR_API_KEY_HERE"}}
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            print(f"A template config file has been created at {config_path}")
            print("Please edit this file to add your Pixabay API key.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download videos from Pixabay and store metadata in a central JSON file.")
    parser.add_argument("search_term", help="Search term for videos (e.g., 'finance+business')")
    parser.add_argument("num_videos", type=int, nargs='?', default=5, 
                        help="Number of videos to download (default: 5)")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to configuration file (default: config.yaml)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = load_config(args.config)
    api_key = config.get('pixabay', {}).get('api_key')
    
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("Please edit the config.yaml file and add your Pixabay API key.")
        sys.exit(1)
    
    downloader = PixabayVideoDownloader(
        api_key=api_key,
        search_term=args.search_term,
        num_videos=args.num_videos
    )
    downloader.run()