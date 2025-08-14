# Automated Video Production System

An automated video production system that can automatically insert relevant video clips into financial reporting videos using AI-driven content matching and video composition techniques.

## Recent Bug Fixes and Improvements

### PyTorch Conversion for HPC Compatibility (Latest - December 2024)

**Issue:** FFmpeg compatibility issues in HPC (High Performance Computing) environments.

**Root Cause:** The manual video insertion system relied on FFmpeg subprocess calls which:
- Failed to execute properly in HPC cluster environments
- Had external dependency issues in containerized deployments
- Lacked GPU acceleration capabilities for video processing
- Created system compatibility barriers for deployment

**Solution:** Complete conversion to PyTorch-based video processing:
- **Pure Python implementation**: Replaced FFmpeg subprocess calls with PyTorch tensor operations
- **GPU acceleration**: Added CUDA support for faster video processing on GPU clusters
- **HPC compatibility**: Ensured compatibility with containerized and cluster environments
- **Batch processing**: Implemented efficient batch processing for large video operations
- **Progress monitoring**: Maintained real-time progress feedback during processing

**Technical Implementation:**
- New `PyTorchVideoProcessor` class with GPU-accelerated frame operations
- OpenCV for video I/O, PyTorch for processing and overlay operations
- Tensor-based frame resizing, positioning, and blending
- Memory-efficient batch processing with configurable batch sizes

**Files Modified:**
- `manual_insert/manual_inserter.py` - Complete rewrite using PyTorch
- `requirements.txt` - Updated with PyTorch dependencies
- `manual_insert/test_pytorch_version.yaml` - Added test configuration

**Usage Impact:** 
- Video processing now works in HPC environments without external dependencies
- GPU acceleration provides significant performance improvements
- Memory-efficient processing handles large 4K videos (up to 180GB estimated memory requirement)
- Automatic chunked processing prevents GPU memory overflow
- Audio insertion temporarily removed (use FFmpeg separately for audio if needed)
- Maintains all existing functionality including center-based coordinates and progress monitoring

**Memory Optimization Features:**
- Conservative GPU memory allocation (60% instead of 80%)
- Automatic batch size adjustment based on available GPU memory  
- Chunked processing for videos requiring >10GB memory
- Real-time memory monitoring and cleanup
- Support for videos up to 4K resolution on 16GB+ GPUs
- **Efficient segment processing**: Only processes modified video segments (99%+ memory savings)
- **Automatic audio preservation**: Maintains original audio track using FFmpeg or moviepy fallback

### JSON Format Compatibility Fix (Previous - August 2025)

**Issue:** Module compatibility issues due to JSON format changes in the semantic matching system.

**Root Cause:** The data-match module updated its JSON output format from direct arrays to a `sentence_groups` wrapper structure, causing compatibility issues in downstream modules:
- video-merge and video-merge-simple modules expecting old format fields
- Field name changes from `matching_videos` to `recommended_videos`
- Nested structure changes affecting video composition pipeline

**Solution:** Implemented comprehensive format detection and compatibility:
- **Automatic format detection**: All modules now detect and handle both old and new JSON formats
- **Field name mapping**: Dynamic mapping between `matching_videos`/`recommended_videos` and related fields
- **Backward compatibility**: Full support for existing JSON files and workflows
- **Enhanced error handling**: Graceful fallbacks and informative error messages

**Files Modified:**
- `data_match/semantic_video_matcher.py` - Fixed duplicate execution and JSON output format
- `video_merge/video_composer.py` - Added JSON format compatibility detection
- `video_merge/video_composer2.py` - Enhanced format detection and field mapping
- `manual_insert/manual_inserter.py` - Added compatibility for show-recommendations feature

**Usage Impact:** All modules now seamlessly work with both old and new JSON formats, ensuring uninterrupted workflows and backward compatibility.

### Manual Video Insert Improvements with YAML Configuration (Latest - August 2025)

**Issue:** Manual video insertion had several usability and functionality problems, plus inflexible configuration management.

**Root Cause:** The manual insertion system had coordinate system inconsistencies, lack of progress feedback, video encoding issues, and required repetitive command-line parameters:
- Position coordinates using top-left corner instead of center point (inconsistent with video-merge)
- No progress indication during processing, making it unclear if the system was working
- Static video frames instead of animated clips due to codec issues
- Poor user experience with long processing times
- **Configuration inflexibility**: Required `--input-video` parameter even when using YAML configs
- **Path restrictions**: Unclear whether only specific folders could be used for video sources

**Solution:** Implemented comprehensive manual insertion improvements with flexible YAML configuration:
- **Center-based coordinates**: Position parameters now specify center point, consistent with video-merge module
- **Progress monitoring**: Real-time progress bars and status updates during video processing
- **Enhanced video encoding**: Proper codec settings to ensure dynamic video clips instead of static frames
- **Better error handling**: Improved feedback and debugging information
- **Self-contained YAML mode**: Configuration files can include input/output paths, eliminating command-line parameters
- **Path freedom**: Videos can be sourced from any accessible path on the system
- **Dual usage modes**: Support both legacy command-line style and modern self-contained configurations

**Files Modified:**
- `manual_insert/manual_inserter.py` - Overhauled coordinate system, added progress monitoring, fixed video encoding, implemented dual configuration modes
- `manual_insert/simple_insert_config.yaml` - Self-contained configuration example
- `manual_insert/complete_config_example.yaml` - Advanced configuration showcase
- `manual_insert/YAML_MODES_GUIDE.md` - Comprehensive usage documentation

**Usage Impact:** 
- **Legacy mode**: `python main.py manual-insert -- --config config.yaml --input-video input.mp4 --output output.mp4`
- **Self-contained mode**: `python main.py manual-insert -- --config self_contained.yaml` (no additional parameters needed)
- **Path flexibility**: Videos can be sourced from `C:/any/path/video.mp4`, `\\network\share\video.mp4`, or any accessible location
- **Simplified workflow**: Configuration files contain all necessary information for repeatable processing

### Small Video Download Fix (Previous - August 2025)

**Issue:** API error when downloading fewer than 3 videos from Pixabay.

**Root Cause:** The Pixabay API requires a minimum value of 3 for the `per_page` parameter, but the system was setting it to match the requested number of videos (e.g., 2), causing:
- `[ERROR 400] "per_page" is out of valid range` error
- Failed downloads for requests of 1-2 videos
- Inconsistent behavior for small batch downloads

**Solution:** Implemented minimum parameter validation:
- **Minimum per_page enforcement**: Always use minimum 3 for `per_page` parameter
- **Proper result limiting**: Download minimum required but return only requested number
- **Better error handling**: Graceful handling of API parameter constraints

**Files Modified:**
- `data_collect_label/data_collect.py` - Fixed `per_page` parameter validation in `search_videos()` method

**Usage Impact:** Small video downloads (1-2 videos) now work correctly without API errors.

### Semantic Matching Algorithm Improvements (Latest - August 2025)

**Issue:** Original semantic clustering grouped non-adjacent sentences across the entire document, affecting narrative flow and coherence.

**Root Cause:** The previous algorithm used `AgglomerativeClustering` to group semantically similar sentences regardless of their position in the text, leading to:
- Non-adjacent sentences being grouped together
- Breaking narrative flow and temporal coherence
- Unwanted "suggested insert time" fields in outputs
- Lack of quality control for video matches

**Solution:** Implemented adjacent-only sentence grouping with quality control:
- **Adjacent sentence grouping**: Only groups consecutive sentences based on similarity
- **Similarity threshold control**: `--similarity-threshold` parameter (default: 0.5) controls grouping sensitivity
- **Distance threshold filtering**: `--distance-threshold` parameter (default: 2.0) filters out poorly matching videos
- **Enhanced output**: Removed suggested insert time fields, added debug similarity scores
- **Quality control**: Groups can have zero matching videos if none meet the distance threshold

**Files Modified:**
- `data_match/semantic_video_matcher.py` - Complete algorithm redesign with adjacent-only grouping
- Removed AgglomerativeClustering dependency, added dual threshold system
- Enhanced parameter passing and error handling throughout the processing chain

**Usage Impact:** Video matching now preserves narrative flow by only grouping adjacent sentences, with improved match quality through distance filtering. The system provides better control over grouping behavior and video match quality.

### Video Tensor Dimension Fix (Previous - August 2025)

**Issue:** Tensor dimension mismatches causing video overlay failures during composition.

**Root Cause:** Overlay videos with different aspect ratios (especially vertical videos) were not properly resized, leading to:
- `RuntimeError: The size of tensor a (2160) must match the size of tensor b (3612)` errors
- Failed video insertions at specific timestamps
- Inconsistent overlay rendering for videos exceeding main video boundaries

**Solution:** Implemented intelligent dimension management:
- **Boundary checking**: Automatic detection of overlay videos exceeding main video dimensions
- **Aspect ratio preservation**: Smart resizing maintaining original video proportions
- **Dynamic size calculation**: Overlay videos automatically fit within available space
- **Error recovery**: Graceful handling of dimension mismatches with detailed logging

**Files Modified:**
- `video_merge/video_composer.py` - Enhanced `overlay_batch()` and `parse_size()` methods
- Added comprehensive size validation and automatic resize logic

**Usage Impact:** Video composition now handles all video aspect ratios correctly, eliminating tensor errors and ensuring all overlay videos are properly inserted.

<!--
### Video Matching Logic Fix (Previous)

**Issue:** Videos were repeating incorrectly during playback due to misaligned mapping between sentence groups and transcript entries.

**Root Cause:** The `match_videos_to_transcript()` function was using simple sequential indexing instead of matching based on actual text content, causing:
- Video segments to play at wrong timestamps  
- Same videos to repeat multiple times in close succession
- Content mismatch between audio and inserted video clips

**Solution:** Implemented content-based matching that:
- Creates proper mapping between transcript text and sentence groups
- Matches videos based on actual sentence content rather than position
- Includes fuzzy matching for edge cases
- Eliminates incorrect video repetition

**Files Modified:**
- `video_merge/video_composer.py` - Fixed PyTorch-based composer
- `video_merge/video_composer2.py` - Fixed MoviePy-based composer

**Usage Impact:** Video composition now correctly aligns video clips with their corresponding audio segments, eliminating unwanted repetition and improving content coherence.

-->

## System Architecture


```
video_code/
├── main.py                    # Main program entry point
├── README.md                  # This file
├── config.py                  # Configuration management tool
├── requirements.txt           # Python dependencies
├── outputs/                  # Unified output directory
├── input/                    # Unified input directory
├── text_to_speech/           # Text-to-Speech module
│   ├── speech_synthesis.py
├── data_collect_label/       # Data collection and labeling module
│   ├── data_collect.py       # Video download from Pixabay
│   ├── data_label.py         # AI-powered video labeling
├── data_match/              # Data matching module
│   ├── semantic_video_matcher.py  # Semantic video matching
│   ├── video_vectorizer.py   # Video vectorization
├── video_merge/             # Video composition module
│   ├── video_composer.py     # PyTorch GPU-accelerated composer
│   └── video_composer2.py    # MoviePy-based simple composer
├── manual_insert/           # Manual video/audio insertion module
│   ├── manual_inserter.py    # Manual insertion control (PyTorch-based)
│   ├── simple_insert_config.yaml        # Basic self-contained configuration
│   ├── video_insert_example.yaml        # Complex insertion examples
│   ├── complete_config_example.yaml     # Advanced feature showcase
│   ├── YAML_MODES_GUIDE.md              # Configuration usage guide
│   └── YAML_USAGE_GUIDE.md              # Detailed YAML documentation
└── auto_caption/            # Auto captioning module (under development)
```
│   └── video_composer2.py    # MoviePy-based simple composer
└── auto_caption/            # Auto captioning module (under development)
```

## Important: Command Line Syntax

**When using parameters with dashes (like `--input`, `--output`), you MUST use the `--` separator:**

✅ **Correct:**
```bash
python main.py tts -- --input script.txt --output outputs/tts/
python main.py data-label -- --device cpu --frames 3
```

❌ **Incorrect:**
```bash
python main.py tts --input script.txt --output outputs/tts/
python main.py data-label --device cpu --frames 3
```

**For positional arguments (no dashes), use directly:**
```bash
python main.py data-collect finance 10  # No -- needed
python main.py list                      # No -- needed
```

## Quick Start

### Installation and Setup

```bash
# Install Dependencies
pip install -r requirements.txt

```

**Azure Speech Service API (for text to speech)**

- **Set environment variables** `SPEECH_KEY` and `ENDPOINT` (Azure Speech Service) 
- Ensure input text files are in UTF-8 encoding

**Configuration Options (in code):**
- `speech_synthesis_voice_name`: Default is 'zh-HK-HiuMaanNeural' (Cantonese)
- `speech_synthesis_output_format`: Default is Riff24Khz16BitMonoPcm
- Input/Output directories can be modified via command line parameters

**Pixabay API (for data collection)**
- **Set Pixabay API key** in `data_collect_label/config.yaml`
- Internet connection for API calls and downloads

**Configuration File Format (`config.yaml`):**
```yaml
pixabay:
  api_key: "your_pixabay_api_key_here"
```

**API Parameters (configurable in code):**
- `safesearch`: 'true' (default)
- `lang`: 'en' (default)
- `order`: 'popular' (default)
- `per_page`: Up to 200 per request
- Video quality preference: ['large', 'medium', 'small', 'tiny']

## Module Usage and Parameters

### Step 1: Text-to-Speech (TTS)

Convert text scripts to speech with word boundary timestamps using Azure Speech Services.


```bash
# Process all .txt files in default input directory (text_to_speech/input_text/)
python main.py tts

# Process specific text file
python main.py tts -- --input script.txt

# Process specific directory
python main.py tts -- --input custom_text_files/

# Specify custom output directory
python main.py tts -- --input script.txt --output outputs/custom_tts/

# Using short parameter names
python main.py tts -- -i script.txt -o outputs/tts/
```

**Module Parameters:**
- `--input, -i`: Input text file or directory (default: processes all .txt files in text_to_speech/input_text/)
- `--output, -o`: Output directory (default: outputs/tts/)

**Input Options:**
- **Single file**: Specify a .txt file path (e.g., `--input script.txt`)
- **Directory**: Specify a directory containing .txt files (e.g., `--input text_files/`)
- **Default**: If not specified, processes all .txt files in `text_to_speech/input_text/`


**Output:**
- WAV audio files in `outputs/tts/{filename}/` or specified output directory
- Timestamped text files with format: `start_time_end_time_sentence`
- Directory structure: `{output_dir}/{script_name}/{script_name}.wav` and `{script_name}.txt`

### Step 2: Data Collection

Download relevant video clips from Pixabay API.

```bash
# Download finance-related videos
python main.py data-collect finance 10

# Download university-related videos  
python main.py data-collect university 5

# Using configuration file
python main.py data-collect -- --config custom_config.yaml business 15
```

**Module Parameters:**
- `search_term` (required): Keywords to search for videos
- `num_videos` (optional): Number of videos to download (default: 5)
- `--config, -c`: Path to configuration file (default: config.yaml)

**Advanced Usage:**
```bash
python main.py data-collect -- "finance+business" 20 --config my_config.yaml
```


**Output:**
- Video files in `data_collect_label/videos/` with format: `{video_id}_{search_term}_{quality}.mp4`
- Metadata stored in `video_metadata.json`

### Step 3: Data Labeling

Use AI models to generate tags and descriptions for videos.

```bash
# Label all videos
python main.py data-label

# Label specific video by ID
python main.py data-label -- --video_id 123456

# Use multiple frames for analysis
python main.py data-label -- --frames 3

# Use smaller model for faster processing
python main.py data-label -- --smaller_model

# Force CPU processing
python main.py data-label -- --device cpu

# Disable debug output
python main.py data-label -- --no_debug
```

**Module Parameters:**
- `--videos_dir`: Directory containing video files (default: "videos")
- `--metadata`: Path to metadata JSON file (default: "video_metadata.json")
- `--model_path`: Path to model or model ID (default: "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B")
- `--device`: Device for inference - "cpu", "cuda", "auto" (default: "auto")
- `--frames`: Number of frames to extract per video (default: 1)
- `--video_id`: Process only specific video ID
- `--smaller_model`: Use lighter model for faster processing
- `--no_debug`: Disable debug output

**Model Options:**
- Default: "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B" (multilingual)
- Smaller: "joaogante/phi-2-vision" (when using --smaller_model)
- Custom: Any Hugging Face model ID

**Advanced Usage:**
```bash
# Process specific video with custom model
python main.py data-label -- --video_id 172888 --model_path "custom/model" --frames 5 --device cuda

# Batch process with smaller model on CPU
python main.py data-label -- --smaller_model --device cpu --frames 2
```

**Prerequisites:**
- Downloaded video files
- GPU (recommended) or sufficient CPU resources
- PyTorch and transformers installed
- Internet connection for model downloads

**Output:**
- Updated `video_metadata.json` with AI-generated tags

### Step 4: Create Video Vector Index

Create vector embeddings for semantic video matching.

```bash
# Create vectors with default settings
python main.py video-vectorize

# Use custom model and batch size
python main.py video-vectorize -- --model paraphrase-multilingual-MiniLM-L12-v2 --batch-size 64

# Custom metadata file and output directory
python main.py video-vectorize -- --metadata custom_metadata.json --output custom_vectors

# Test the created vectors
python main.py video-vectorize -- --test --test-query "financial market analysis"
```

**Module Parameters:**
- `--metadata`: Path to video metadata JSON file (default: "video_metadata.json")
- `--output`: Output directory for vectors and index (default: "vectors")
- `--model`: Sentence transformer model name (default: "paraphrase-multilingual-MiniLM-L12-v2")
- `--batch-size`: Batch size for computing embeddings (default: 32)
- `--test`: Run test query after creating vectors
- `--test-query`: Query to use for testing (default: "financial market stock trading")

**Model Options:**
- `paraphrase-multilingual-MiniLM-L12-v2`: Fast, good quality (default)
- `paraphrase-multilingual-mpnet-base-v2`: Higher quality, slower
- `all-MiniLM-L6-v2`: Fastest, English only
- Any sentence-transformers compatible model

**Advanced Usage:**
```bash
# High-quality embeddings with larger batch
python main.py video-vectorize -- --model paraphrase-multilingual-mpnet-base-v2 --batch-size 16

# Fast processing for large datasets
python main.py video-vectorize -- --model all-MiniLM-L6-v2 --batch-size 128
```

**Output:**
- FAISS indexes: `video_index_flat.faiss` (precise), `video_index_hnsw.faiss` (fast)
- Raw embeddings: `video_embeddings.pkl`
- ID mapping: `video_id_mapping.json`
- Metadata: `embedding_metadata.json`

### Step 5: Data Matching

Find matching videos for text content using semantic search with adjacent sentence grouping.

```bash
# Direct text input
python main.py data-match -- --text "Stock market performance today shows positive trends"

# File input with similarity threshold control
python main.py data-match -- --file input_text/script.txt --similarity-threshold 0.4

# Distance threshold for match quality control
python main.py data-match -- --file script.txt --distance-threshold 1.5 --output results.txt

# Combined thresholds for precise control
python main.py data-match -- --text "finance news" --similarity-threshold 0.6 --distance-threshold 1.8 --max-group-size 5 --num-videos 3
```

**Module Parameters:**
- `--text`: Input text for matching (alternative to --file)
- `--file`: Path to file containing text (alternative to --text)
- `--vectors`: Directory containing vector data (default: "vectors")
- `--metadata`: Path to video metadata (default: "video_metadata.json")
- `--max-group-size`: Maximum sentences per group (default: 3)
- `--num-videos`: Number of videos per group (default: 2)
- `--output`: Output file path (optional)
- `--top-k`: Number of top matching videos to return (default: 2)
- `--similarity-threshold`: Similarity threshold for adjacent sentence grouping (default: 0.5)
- `--distance-threshold`: Maximum distance threshold for video matching (default: 2.0)

**New Grouping Algorithm:**
- **Adjacent-only grouping**: Only groups consecutive sentences based on cosine similarity
- **Similarity threshold**: Controls when adjacent sentences are grouped together (0.0-1.0)
- **Distance filtering**: Videos with distance above threshold are excluded from results
- **Quality control**: Groups may have zero videos if none meet the distance threshold

**Advanced Usage:**
```bash
# Strict grouping with high-quality matches only
python main.py data-match -- --file script.txt --similarity-threshold 0.7 --distance-threshold 1.0

# Permissive grouping with moderate quality filtering
python main.py data-match -- --file script.txt --similarity-threshold 0.3 --distance-threshold 2.5

# Debug mode to see similarity scores
python main.py data-match -- --text "market analysis" --similarity-threshold 0.4 --distance-threshold 2.0
```

**Interactive Mode:**
```bash
# Enter text interactively (press Ctrl+D/Ctrl+Z when finished)
python main.py data-match
```

**Algorithm Benefits:**
- **Preserves narrative flow**: Only adjacent sentences are grouped
- **Quality control**: Distance threshold filters poor matches
- **Tunable sensitivity**: Similarity threshold controls grouping behavior
- **Debug feedback**: Shows similarity scores between adjacent sentences
- **Graceful handling**: Groups with no matching videos are handled properly

**Prerequisites:**
- Created video vector index
- Labeled video database

**Output:**
- Text format: Formatted results with sentence groups and matching videos
- JSON format: Structured data for video composition (automatically created with .json extension)
- No suggested insert time fields (removed for cleaner output)
- Debug information showing similarity scores and filtering decisions

### Step 6: Video Composition

Compose final video by inserting matched video clips at specific timestamps. Choose between two methods:

#### Method 1: PyTorch GPU-Accelerated (Recommended for large projects)

```bash
# Basic video composition with PyTorch
python main.py video-merge -- \
    --json outputs/data_match/matches.json \
    --transcript outputs/tts/script/script.txt \
    --input-video input/video.mp4 \
    --videos-dir outputs/videos \
    --output merge.mp4 \
    --audio outputs/tts/script/script.wav \
    --batch-size 200

# Advanced composition with custom positioning and audio
python main.py video-merge -- \
    --json matches.json \
    --transcript timestamps.txt \
    --input-video source.mp4 \
    --videos-dir videos \
    --output result.mp4 \
    --position "80%,10%" \
    --size "30%" \
    --distance-threshold 1.5 \
    --batch-size 100 \
    --audio outputs/tts/script/script.wav \
    --audio-start 2.0
```
```

**PyTorch Method Parameters:**
- `--json` (required): Path to JSON file with matched videos
- `--transcript` (required): Path to transcript file with timestamps
- `--input-video` (required): Path to main/source video file
- `--videos-dir` (required): Directory containing pre-downloaded video clips
- `--output`: Output video filename (default: "composed_video_pytorch.mp4")
- `--position`: Position to insert video clips (default: "50%,50%")
- `--size`: Size of inserted videos (default: "25%")
- `--distance-threshold`: Maximum distance threshold for video inclusion (default: 2.0)
- `--cpu`: Force CPU processing even if GPU is available
- `--batch-size`: Number of frames to process in each batch (default: 50)
- `--audio`: Path to audio file to add to the final video (e.g., TTS-generated audio)
- `--audio-start`: Start time for audio in seconds (default: 0.0)

#### Method 2: MoviePy Simple Composer (Easy to use, auto-downloads)

```bash
# Simple video composition with automatic downloads
python main.py video-merge-simple -- \
    --json outputs/data_match/matches.json \
    --transcript outputs/tts/script/script.txt \
    --input-video input/video.mp4 \
    --output outputs/merge.mp4

# With custom positioning and audio
python main.py video-merge-simple -- \
    --json matches.json \
    --transcript timestamps.txt \
    --input-video source.mp4 \
    --output result.mp4 \
    --position "bottom-right" \
    --size 0.3 \
    --distance-threshold 1.5 \
    --audio outputs/tts/script/script.wav \
    --debug
```

**MoviePy Method Parameters:**
- `--json` (required): Path to JSON file with matched videos
- `--transcript` (required): Path to transcript file with timestamps
- `--input-video` (required): Path to main/source video file
- `--output`: Output video filename (default: "final_video.mp4")
- `--position`: Position preset ("top-left", "top-right", "bottom-left", "bottom-right", "center")
- `--size`: Size ratio of inserted videos (0.1-1.0, default: 0.25)
- `--distance-threshold`: Maximum distance threshold for video inclusion (default: 2.0)
- `--download-method`: Download method ("wget", "curl", "urllib") - auto-detects best available
- `--debug`: Show detailed debug information
- `--audio`: Path to audio file to add to the final video (e.g., TTS-generated audio)
- `--audio-start`: Start time for audio in seconds (default: 0.0)

#### Choosing the Right Method

**Use PyTorch Method (`video-merge`) when:**
- Processing large or high-resolution videos
- Need maximum performance and GPU acceleration
- Have pre-downloaded video files
- Working with professional video production

**Use MoviePy Method (`video-merge-simple`) when:**
- Quick testing and prototyping
- Smaller video projects
- Want automatic video downloads
- Prefer simplicity over performance

**Position Parameter Format:**
```bash
# Percentage-based (recommended)
--position "50%,50%"     # Center of screen
--position "80%,20%"     # Top-right area
--position "20%,80%"     # Bottom-left area

# Pixel-based
--position "100,50"      # 100px from left, 50px from top

# Mixed format
--position "50%,100"     # Center horizontally, 100px from top
```

**Size Parameter Format:**
```bash
# Percentage of main video width (maintains aspect ratio)
--size "25%"             # 25% of main video width
--size "40%"             # 40% of main video width

# Width-based
--size "320px"           # 320 pixels width
--size "30%w"            # 30% of main video width

# Height-based  
--size "20%h"            # 20% of main video height
--size "240px-h"         # 240 pixels height
```

**Advanced Parameters:**
- `--distance-threshold`: Lower values = more strict matching (0.0-5.0, default: 2.0)
- `--batch-size`: Higher values = faster processing but more memory usage (default: 50)
- `--cpu`: Use when GPU memory is insufficient

**Performance Optimization:**
```bash
# High performance (requires GPU with sufficient memory)
python main.py video-merge -- [...] --batch-size 100

# Memory conservative (for limited GPU memory)
python main.py video-merge -- [...] --batch-size 20

# CPU only (slower but compatible)
python main.py video-merge -- [...] --cpu --batch-size 30
```

**Prerequisites:**
- Main video file
- TTS-generated timestamp file
- Matching results JSON file
- Downloaded video clips directory

**Output:**
- Final composed video file with inserted clips

### Step 7: Auto Caption Generation

Add subtitles to videos automatically using timestamp script files. Choose between two methods:

#### Method 1: OpenCV GPU-Accelerated (Recommended for HPC environments)

```bash
# Basic caption generation with OpenCV
python main.py auto-caption -- \
    --input-video outputs/merge_video.mp4 \
    --script outputs/tts/script/script.txt

# Advanced caption generation with custom styling
python main.py auto-caption -- \
    --input-video outputs/merge_video.mp4 \
    --script outputs/tts/script/script.txt \
    --output outputs/auto_caption/final_video_opencv.mp4 \
    --font-size 28 \
    --font-color white \
    --position bottom \
    --margin-bottom 60 \
    --max-width 85 \
    --batch-size 200 \
    --quality high

# CPU-only processing for compatibility
python main.py auto-caption -- \
    --input-video outputs/merge_video.mp4 \
    --script outputs/tts/script/script.txt \
    --no-gpu \
    --batch-size 100
```

**OpenCV Method Parameters:**
- `--input-video` (required): Path to input video file
- `--script` (required): Path to timestamp script file
- `--output`: Output video file path (default: "outputs/auto_caption/final_video_opencv.mp4")
- `--font-size`: Font size for subtitles (default: 24)
- `--font-color`: Font color - "white", "black", "yellow" (default: "white")
- `--position`: Subtitle position - "bottom", "top", "center" (default: "bottom")
- `--margin-bottom`: Bottom margin in pixels (default: 50)
- `--max-width`: Maximum characters per line (default: 80)
- `--line-spacing`: Line spacing multiplier (default: 1.2)
- `--batch-size`: Frames to process per batch (default: 200)
- `--no-gpu`: Disable GPU acceleration
- `--workers`: Number of worker threads (default: 4)
- `--quality`: Output quality - "normal", "high" (default: "high")

#### Method 2: FFmpeg-based (Alternative for systems with FFmpeg GPU support)

```bash
# Basic caption generation with FFmpeg
python main.py auto-caption-ffmpeg -- \
    --input-video outputs/merge_video.mp4 \
    --script outputs/tts/script/script.txt

# Advanced caption generation with custom styling
python main.py auto-caption-ffmpeg -- \
    --input-video outputs/merge_video.mp4 \
    --script outputs/tts/script/script.txt \
    --output outputs/auto_caption/final_video_ffmpeg.mp4 \
    --font-size 28 \
    --font-color white \
    --position bottom \
    --margin-bottom 60 \
    --max-width 85 \
    --quality high

# CPU-only processing
python main.py auto-caption-ffmpeg -- \
    --input-video outputs/merge_video.mp4 \
    --script outputs/tts/script/script.txt \
    --no-gpu
```

**FFmpeg Method Parameters:**
- `--input-video` (required): Path to input video file
- `--script` (required): Path to timestamp script file
- `--output`: Output video file path (default: "outputs/auto_caption/final_video_ffmpeg.mp4")
- `--font-size`: Font size for subtitles (default: 24)
- `--font-color`: Font color - "white", "black", "yellow" (default: "white")
- `--position`: Subtitle position - "bottom", "top", "center" (default: "bottom")
- `--margin-bottom`: Bottom margin in pixels (default: 50)
- `--max-width`: Maximum characters per line (default: 80)
- `--line-spacing`: Line spacing multiplier (default: 1.2)
- `--quality`: Output quality - "normal", "high" (default: "high")
- `--no-gpu`: Disable GPU acceleration

#### Choosing the Right Method

**Use OpenCV Method (`auto-caption`) when:**
- Working in HPC environments with A800/similar GPUs
- Need fine control over text rendering and positioning
- Want batch processing for large videos
- FFmpeg GPU acceleration is unavailable
- Processing multiple videos in sequence

**Use FFmpeg Method (`auto-caption-ffmpeg`) when:**
- FFmpeg with GPU support is available and working
- Need fastest possible processing speed
- Want professional-grade subtitle rendering
- Working with standard video production workflows

#### Script File Format

The timestamp script file should follow this format:
```
00:00.05_00:10.68_你好, 我係Christina，我哋依家一齊嚟睇下2025年7月28日至2025年8月1日嘅美國股市市場分析同預測。
00:10.68_00:18.00_美股近期表現強勁，標普500同納指屢創歷史新高，反映市場動力十足。
```

Format: `start_time_end_time_subtitle_text`
- Time format: `MM:SS.ms` or `HH:MM:SS.ms`
- Supports Chinese, English, and other Unicode characters
- Automatic text wrapping based on `max-width` setting

#### Performance Optimization for HPC

```bash
# High-performance settings for A800 GPU with 80GB memory
python main.py auto-caption -- \
    --input-video large_video.mp4 \
    --script script.txt \
    --batch-size 500 \
    --workers 8 \
    --font-size 32 \
    --quality high

# Memory-conservative settings
python main.py auto-caption -- \
    --input-video large_video.mp4 \
    --script script.txt \
    --batch-size 100 \
    --workers 4 \
    --no-gpu  # Use if GPU memory is limited
```

**HPC Optimization Tips:**
- Larger `batch-size` (200-500) for high-memory systems
- More `workers` (4-8) for multi-core systems
- Use `--quality high` for best output
- Monitor memory usage during processing

**Prerequisites:**
- Input video file (MP4 format recommended)
- Timestamp script file with proper formatting
- OpenCV with GPU support (for OpenCV method)
- FFmpeg with subtitle support (for FFmpeg method)
- Chinese font files (automatically detected)

**Output:**
- Final video with embedded subtitles
- Supports multiple subtitle lines per timestamp
- Professional-quality text rendering with shadow effects


### Step 8: Manual Video and Audio Insertion

Manually insert video clips and audio overlays into your main video using a flexible configuration system. This module provides complete control over when, where, and how media is inserted, with support for multiple simultaneous insertions and two usage modes.

#### 8.1 Quick Start: Self-Contained YAML Mode (Recommended)

Create a configuration file that includes everything:

```yaml
# my_config.yaml
input_video: "outputs/final.mp4"
output_video: "outputs/result.mp4"

video_inserts:
  - time: "00:05.00"
    videos:
      - source: "C:/any/path/overlay.mp4"  # Any accessible path!
        start: "00:00.00"
        duration: "00:03.00"
        position: "top-right"
        size: "25%"
        loop: false
```

Execute with a single command:
```bash
# Only configuration file needed - no other parameters!
python main.py manual-insert -- --config my_config.yaml
```

#### 8.2 Alternative: Legacy Mode with Command Line Parameters

For backward compatibility or quick single insertions:

```bash
# Legacy mode - requires input/output parameters
python main.py manual-insert -- \
    --config legacy_config.yaml \
    --input-video input.mp4 \
    --output output.mp4

# Direct command line insertion (no config file)
python main.py manual-insert -- \
    --input-video input.mp4 \
    --output output.mp4 \
    --add-video overlay.mp4 \
    --insert-time "00:05.00" \
    --position "top-right" \
    --size "25%"
```

#### 8.3 Path Flexibility

Videos can be sourced from anywhere on your system:

```yaml
video_inserts:
  - time: "00:10.00"
    videos:
      - source: "C:/Videos/chart.mp4"              # Windows absolute path
      - source: "D:/Media/finance_data.mp4"        # Different drive
      - source: "\\server\share\content.mp4"       # Network path
      - source: "../downloads/clip.mp4"            # Relative path
      - source: "outputs/videos/local.mp4"         # Project relative
```

**No restrictions on folder locations!**

#### 8.4 Configuration Examples

**Basic Single Insertion:**
```yaml
input_video: "main_video.mp4"
output_video: "result.mp4"

video_inserts:
  - time: "00:10.00"
    videos:
      - source: "/path/to/overlay.mp4"
        start: "00:02.00"
        duration: "00:05.00"
        position: "center"
        size: "40%"
        loop: false
```

**Multiple Simultaneous Insertions:**
```yaml
input_video: "presentation.mp4"
output_video: "enhanced_presentation.mp4"

video_inserts:
  - time: "00:15.00"
    videos:
      - source: "chart1.mp4"
        position: "top-left"
        size: "30%"
        duration: "00:05.00"
      
      - source: "chart2.mp4" 
        position: "top-right"
        size: "30%"
        duration: "00:05.00"
      
      - source: "logo.mp4"
        position: "bottom-right"
        size: "15%"
        duration: "00:10.00"
```

**Audio and Video Combined:**
```yaml
input_video: "base_video.mp4"
output_video: "final_video.mp4"

video_inserts:
  - time: "00:05.00"
    videos:
      - source: "overlay.mp4"
        position: "top-right"
        size: "25%"
        duration: "00:03.00"

audio_inserts:
  - time: "00:00.00"
    source: "background_music.mp3"
    duration: "00:30.00"
    volume: 0.3
    mix_mode: "overlay"
```

#### 8.5 Display Recommendations

Generate timing recommendations from matched data:

```bash
# Show existing recommendations in readable format
python main.py manual-insert -- --show-recommendations outputs/data_match/matches.json
```

**Example Output:**
```
推薦插入點分析
================

句子群組 1 (時間: 00:00.05 - 00:10.68, 時長: 00:10.63):
推薦影片:
1. 172888_finance_large.mp4 (distance: 0.1521)
   標籤: finance, stock market, analysis
```

```yaml
# manual_insert_config.yaml

# Video insertions - insert multiple videos at specific times
video_inserts:
  # Insert at 5 seconds into main video
  - time: "00:05.00"              # When to insert in main video
    videos:
      # First video clip (top-left corner)
      - source: "data_collect_label/videos/172888_finance_large.mp4"
        start: "00:00.00"          # Start time in source video
        duration: "00:03.00"       # How long to insert
        position: "top-left"       # Position on screen
        size: "25%"               # Size as percentage of main video width
        loop: true                # Loop if source is shorter than duration
      
      # Second video clip (top-right corner, same time)
      - source: "data_collect_label/videos/172891_finance_large.mp4"
        start: "00:02.00"
        duration: "00:03.00"
        position: "top-right"
        size: "25%"
        loop: false

  # Insert at 15 seconds
  - time: "00:15.00"
    videos:
      - source: "data_collect_label/videos/172894_finance_large.mp4"
        start: "00:00.00"
        duration: "00:05.00"
        position: "50%,20%"        # Custom position (50% from left, 20% from top)
        size: "30%"
        loop: true

# Audio insertions - add audio overlays (optional)
audio_inserts:
  # Background music overlay
  - time: "00:02.00"              # When to start audio in main video
    source: "path/to/background_music.mp3"
    start: "00:30.00"             # Start time in source audio
    duration: "00:10.00"          # How long to play
    volume: 0.3                   # Volume level (0.0 to 1.0+)
    mix_mode: "overlay"           # "overlay" (mix) or "replace"
  
  # Sound effect
  - time: "00:05.00"
    source: "path/to/sound_effect.wav"
    start: "00:00.00"
    duration: "00:01.00"
    volume: 0.8
    mix_mode: "overlay"
```

#### 8.3 Command Line Direct Insertion

For quick insertions without creating a configuration file, use command line parameters:

##### Video Insertion via Command Line

```bash
# Insert a single video clip
python main.py manual-insert -- \
    --input-video outputs/video_merge/final_video.mp4 \
    --output outputs/manual_insert/final_video_manual.mp4 \
    --add-video data_collect_label/videos/172888_finance_large.mp4 \
    --insert-time "00:05.00" \
    --clip-start "00:02.00" \
    --clip-duration "00:03.00" \
    --position "top-right" \
    --size "25%" \
    --loop

# Insert video with custom positioning
python main.py manual-insert -- \
    --input-video outputs/video_merge/final_video.mp4 \
    --output outputs/manual_insert/final_video_manual.mp4 \
    --add-video path/to/chart_video.mp4 \
    --insert-time "00:10.00" \
    --clip-start "00:00.00" \
    --clip-duration "00:05.00" \
    --position "50%,20%" \
    --size "30%"

# Quick insertion with defaults (3 seconds at top-right)
python main.py manual-insert -- \
    --input-video main_video.mp4 \
    --output result.mp4 \
    --add-video overlay.mp4 \
    --insert-time "00:15.00"
```

##### Audio Insertion via Command Line

```bash
# Insert background music
python main.py manual-insert -- \
    --input-video outputs/video_merge/final_video.mp4 \
    --output outputs/manual_insert/final_video_manual.mp4 \
    --add-audio path/to/background_music.mp3 \
    --audio-time "00:02.00" \
    --audio-start "00:30.00" \
    --audio-duration "00:10.00" \
    --volume 0.3 \
    --mix-mode overlay

# Insert sound effect with volume boost
python main.py manual-insert -- \
    --input-video main_video.mp4 \
    --output result.mp4 \
    --add-audio sound_effect.wav \
    --audio-time "00:05.00" \
    --audio-duration "00:01.00" \
    --volume 0.8

# Replace original audio temporarily
python main.py manual-insert -- \
    --input-video main_video.mp4 \
    --output result.mp4 \
    --add-audio narration.mp3 \
    --audio-time "00:10.00" \
    --audio-duration "00:08.00" \
    --volume 1.0 \
    --mix-mode replace
```

##### Combined Video and Audio Insertion

```bash
# Insert both video and audio simultaneously
python main.py manual-insert -- \
    --input-video outputs/video_merge/final_video.mp4 \
    --output outputs/manual_insert/final_video_manual.mp4 \
    --add-video data_collect_label/videos/172888_finance_large.mp4 \
    --insert-time "00:05.00" \
    --clip-duration "00:03.00" \
    --position "top-left" \
    --size "25%" \
    --add-audio background_music.mp3 \
    --audio-time "00:05.00" \
    --audio-duration "00:10.00" \
    --volume 0.4
```

**Command Line Parameters:**

**Video Insertion Parameters:**
- `--add-video`: Path to source video file (required for video insertion)
- `--insert-time`: Time to insert in main video - MM:SS.ms format (required for video insertion)
- `--clip-start`: Start time in source video (default: "00:00.00")
- `--clip-duration`: Duration to insert (default: "00:03.00")
- `--position`: Position on screen - presets or coordinates (default: "top-right")
- `--size`: Size as percentage or pixels (default: "25%")
- `--loop`: Loop video if source is shorter than duration (flag)

**Audio Insertion Parameters:**
- `--add-audio`: Path to source audio file (required for audio insertion)
- `--audio-time`: Time to start audio in main video - MM:SS.ms format (required for audio insertion)
- `--audio-start`: Start time in source audio (default: "00:00.00")
- `--audio-duration`: Audio duration to insert (default: "00:05.00")
- `--volume`: Audio volume level, 0.0-1.0+ (default: 0.8)
- `--mix-mode`: Audio mix mode - "overlay" or "replace" (default: "overlay")

**Common Parameters:**
- `--input-video`: Path to main video file (required)
- `--output`: Path for output video file (required)
- `--temp-dir`: Directory for temporary files (optional)

#### 8.4 Execute Manual Insertions with Configuration File

#### 8.4 Execute Manual Insertions with Configuration File

Process your video with the configuration file:

```bash
# Basic manual insertion
python main.py manual-insert -- \
    --config manual_insert_config.yaml \
    --input-video outputs/video_merge/final_video.mp4 \
    --output outputs/manual_insert/final_video_manual.mp4

# With custom temporary directory
python main.py manual-insert -- \
    --config manual_insert_config.yaml \
    --input-video outputs/video_merge/final_video.mp4 \
    --output outputs/manual_insert/final_video_manual.mp4 \
    --temp-dir /tmp/manual_insert
```

#### 8.6 Position and Size Specifications

**Position Specifications (Center-based coordinates):**
- **Presets**: `"top-left"`, `"top-center"`, `"top-right"`, `"center-left"`, `"center"`, `"center-right"`, `"bottom-left"`, `"bottom-center"`, `"bottom-right"`
- **Percentage**: `"50%,25%"` (50% from left edge, 25% from top edge - center point of video)
- **Pixels**: `"100,50"` (center point at 100px from left, 50px from top)

**Important:** Position coordinates specify the CENTER POINT of the inserted video, consistent with the video-merge module.

**Size Specifications:**
- **Percentage**: `"25%"` (25% of main video width, maintains aspect ratio)
- **Explicit**: `"320x240"` (specific pixel dimensions)
- **Decimal**: `"0.25"` (same as "25%")

**Time Format Support:**
- **MM:SS.ms**: `"01:23.45"` (1 minute, 23.45 seconds)
- **HH:MM:SS.ms**: `"01:23:45.67"` (1 hour, 23 minutes, 45.67 seconds)
- **Seconds**: `"83.45"` (83.45 seconds)

#### 8.7 Usage Mode Comparison

| Feature | Self-Contained YAML | Legacy Mode | Command Line |
|---------|-------------------|-------------|--------------|
| **Setup Complexity** | ⭐⭐⭐⭐⭐ Simple | ⭐⭐⭐ Moderate | ⭐⭐ Complex |
| **Parameter Count** | 1 (config file) | 3+ parameters | 5+ parameters |
| **Multiple Insertions** | ✅ Excellent | ✅ Good | ❌ Difficult |
| **Reusability** | ✅ High | ✅ Medium | ❌ Low |
| **Version Control** | ✅ Easy | ✅ Medium | ❌ Hard |
| **Path Flexibility** | ✅ Any path | ✅ Any path | ✅ Any path |

**Recommended Workflows:**

*Quick Testing:*
```bash
# Command line for single insertions
python main.py manual-insert -- --input-video test.mp4 --output result.mp4 --add-video overlay.mp4 --insert-time "00:05.00"
```

*Production Work:*
```yaml
# Self-contained YAML for complex projects
input_video: "production/main.mp4"
output_video: "production/final.mp4"
video_inserts: [...]
audio_inserts: [...]
```

#### 8.8 Advanced Configuration Features

**Processing Settings:**
```yaml
input_video: "large_video.mp4"
output_video: "processed.mp4"

# Performance optimization
processing_settings:
  memory_efficient: true
  use_segment_processing: true  # Only processes modified segments
  batch_size: 5                 # Smaller for 4K videos
  use_gpu: true

video_inserts: [...]
```

**Loop Behavior:**
```yaml
video_inserts:
  - time: "00:30.00"
    videos:
      - source: "short_clip.mp4"    # 2-second clip
        duration: "00:06.00"        # Want 6 seconds
        loop: true                  # Will repeat 3 times
        position: "center"
        size: "40%"
```

**Prerequisites:**
- **Main video file** (MP4 format recommended)
- **Source video/audio files** for insertion
- **YAML configuration file** (for self-contained mode) or command-line parameters
- **Sufficient disk space** for temporary files

**Output:**
- **Final composed video** with all specified insertions
- **Automatic cleanup** of temporary files
- **Preservation** of original video quality
- **Audio preservation** with automatic fallback systems


## Configuration Files

### Environment Variables for Azure Speech Service

```bash
# Windows Command Prompt
set SPEECH_KEY=your_azure_speech_key
set ENDPOINT=your_azure_endpoint

# Windows PowerShell
$env:SPEECH_KEY="your_azure_speech_key"
$env:ENDPOINT="your_azure_endpoint"

# Linux/macOS
export SPEECH_KEY=your_azure_speech_key
export ENDPOINT=your_azure_endpoint
```

### Pixabay API Configuration

Edit `data_collect_label/config.yaml`:

```yaml
pixabay:
  api_key: "your_pixabay_api_key_here"
```

### Advanced Configuration Options

#### TTS Configuration (modify in speech_synthesis.py)
```python
# Voice options
speech_config.speech_synthesis_voice_name = 'zh-HK-HiuMaanNeural'  # Cantonese
# speech_config.speech_synthesis_voice_name = 'zh-CN-XiaoxiaoNeural'  # Mandarin
# speech_config.speech_synthesis_voice_name = 'en-US-AriaNeural'  # English

# Audio format options
speech_config.speech_synthesis_output_format = speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
```

#### Video Processing Configuration
```python
# Batch sizes (modify in respective modules)
TTS_BATCH_SIZE = 32          # data_label.py
VECTOR_BATCH_SIZE = 32       # video_vectorizer.py  
COMPOSE_BATCH_SIZE = 50      # video_composer.py

# Model configurations
VISION_MODEL = "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B"  # data_label.py
SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"     # video_vectorizer.py
```

## Troubleshooting

### Common Issues and Solutions

#### 0. Semantic Matching Quality Issues (Latest Updates)

**Symptoms:**
- Videos matching non-adjacent sentences 
- Poor narrative flow in video composition
- Low-quality or irrelevant video matches
- Suggested insert time fields in output

**Solutions:**
```bash
# Use adjacent-only grouping with quality control
python main.py data-match -- --file script.txt --similarity-threshold 0.6 --distance-threshold 1.5

# Stricter similarity for better grouping
python main.py data-match -- --file script.txt --similarity-threshold 0.7

# Filter out poor video matches
python main.py data-match -- --file script.txt --distance-threshold 1.0

# Debug mode to see similarity scores
python main.py data-match -- --text "your text" --similarity-threshold 0.4
```

**Parameter Tuning:**
- `--similarity-threshold 0.3-0.4`: More permissive grouping
- `--similarity-threshold 0.6-0.8`: Stricter grouping (better coherence)
- `--distance-threshold 1.0-1.5`: High-quality matches only
- `--distance-threshold 2.0-3.0`: More permissive matching

#### 1. Manual Video Insertion Configuration Issues (Latest Updates)

**Symptoms:**
- Confusion about requiring `--input-video` parameter with YAML configurations
- Uncertainty about path restrictions for video sources
- Repetitive command-line parameter entry for complex projects

**Solutions:**
```bash
# Use self-contained YAML mode (recommended)
python main.py manual-insert -- --config my_config.yaml
# No additional parameters needed!

# YAML configuration includes everything:
input_video: "path/to/input.mp4"
output_video: "path/to/output.mp4"
video_inserts: [...]

# Videos can be sourced from ANY accessible path:
# - C:/Videos/clip.mp4
# - \\server\share\video.mp4  
# - ../downloads/media.mp4
# - /home/user/clips/video.mp4
```

**Configuration Modes:**
- **Self-contained YAML**: Configuration file contains input/output paths
- **Legacy mode**: Configuration + command-line parameters (backward compatible)
- **Command-line only**: Direct parameters without configuration file

#### 2. Manual Video Insertion Position and Progress Issues

**Symptoms:**
- Position coordinates behaving differently from video-merge module
- No progress indication during long processing operations  
- Inserted videos showing only static frames instead of animation
- Unclear processing status and completion time

**Solutions:**
```bash
# All position coordinates now use center-based positioning (consistent with video-merge)
python main.py manual-insert -- \
    --input-video main.mp4 \
    --output result.mp4 \
    --add-video overlay.mp4 \
    --insert-time "00:05.00" \
    --position "75%,50%"  # Center point at 75% width, 50% height

# Progress monitoring is now automatic during processing
# Look for real-time progress updates like:
# "🎯 Self-contained YAML mode: Input: ..., Output: ..."
# "Processing video insertions..."
# "Audio preservation: Using FFmpeg method"
```

#### 3. Module Not Found Errors
```bash
# Ensure you're in the correct directory
cd /path/to/work_code

# Check Python path
python -c "import sys; print(sys.path)"

# Install missing packages
pip install -r requirements.txt
```

#### 4. Azure Speech Service Errors
```bash
# Verify environment variables
echo $SPEECH_KEY
echo $ENDPOINT

# Test connection
python -c "import azure.cognitiveservices.speech as speechsdk; print('Azure SDK loaded')"
```

**Common Error Codes:**
- `401 Unauthorized`: Invalid SPEECH_KEY
- `403 Forbidden`: Invalid ENDPOINT or region mismatch
- `429 Too Many Requests`: API quota exceeded

#### 5. Pixabay Download Failures
```bash
# Verify API key in config.yaml
python config.py verify

# Check network connectivity
curl -I https://pixabay.com/api/videos/
```

**API Limits:**
- Free tier: 20,000 requests/month
- Rate limit: 3 requests/minute
- Minimum `per_page`: 3 videos (API requirement)

**Common Errors:**
- `[ERROR 400] "per_page" is out of valid range`: Fixed in latest version for small downloads
- `401 Unauthorized`: Invalid API key
- `429 Too Many Requests`: Rate limit exceeded

**Small Download Workaround (if using older version):**
```bash
# For 1-2 videos, request 3 and the system will limit results
python main.py data-collect university 3  # Downloads 3, you can delete extras
```

#### 6. Video Processing Memory Issues

**Symptoms:**
- "CUDA out of memory" errors
- System freezing during processing
- Slow processing speeds

**Solutions:**
```bash
# Reduce batch size
python main.py video-merge -- [...] --batch-size 20

# Force CPU processing
python main.py data-label -- --device cpu
python main.py video-merge -- [...] --cpu

# Use smaller models
python main.py data-label -- --smaller_model
```

**Memory Optimization:**
- Close other applications
- Use CPU for large videos
- Process videos in smaller batches
- Use lower resolution source videos

#### 7. GPU Memory Insufficient

**Check GPU memory:**
```bash
nvidia-smi
```

**Optimization strategies:**
```bash
# Smaller batch sizes
python main.py data-label -- --batch-size 8

# CPU fallback for memory-intensive operations
python main.py video-merge -- [...] --cpu

# Use smaller AI models
python main.py data-label -- --smaller_model --device cpu
```

#### 8. Model Download Issues

**Symptoms:**
- "Connection timeout" during model loading
- "Repository not found" errors

**Solutions:**
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Bllossom/llama-3.2-Korean-Bllossom-AICA-5B')"

# Use local models
python main.py data-label -- --model_path /path/to/local/model

# Alternative models
python main.py data-label -- --smaller_model  # Uses lighter model
```

## Performance Optimization

### Hardware Optimization
- **SSD Storage**: Significantly improves video I/O performance
- **High RAM**: Allows larger batch sizes and reduces disk swapping
- **CUDA GPU**: Essential for fast AI inference and video processing
- **Fast Internet**: Reduces model download and API call times

### Software Optimization
```bash
# Use appropriate batch sizes for your hardware
python main.py video-merge [...] --batch-size 100  # High-end GPU
python main.py video-merge [...] --batch-size 50   # Mid-range GPU  
python main.py video-merge [...] --batch-size 20   # Low-end GPU/CPU

# Enable GPU acceleration where possible
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU

# Optimize video formats
# Use H.264 encoded MP4 files for best compatibility and performance
```

### Pipeline Optimization
```bash
# Process videos in parallel (separate terminal windows)
python main.py data-collect finance 20 &
python main.py data-collect business 15 &

# Pre-create vector indexes for faster matching
python main.py video-vectorize -- --batch-size 64

# Use appropriate similarity and distance thresholds for quality control
python main.py data-match -- --file script.txt --similarity-threshold 0.6 --distance-threshold 1.5  # Balanced quality
python main.py data-match -- --file script.txt --similarity-threshold 0.4 --distance-threshold 2.5  # More permissive

# Optimize video composition with quality filtering
python main.py video-merge -- [...] --distance-threshold 1.5  # Stricter matching
python main.py video-merge -- [...] --distance-threshold 3.0  # More permissive
```

## Extension and Customization

### Adding Custom AI Models

#### For Video Labeling:
```python
# In data_label.py, modify the model loading section
self.model = MllamaForConditionalGeneration.from_pretrained(
    "your-custom-model-id",
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
```

#### For Text Embeddings:
```python
# In video_vectorizer.py, use any sentence-transformers model
model = SentenceTransformer("your-custom-embedding-model")
```

### Custom Video Sources

Extend `data_collect.py` to support additional video sources:
```python
class CustomVideoDownloader:
    def __init__(self, api_key, source_name):
        self.api_key = api_key
        self.source_name = source_name
    
    def search_videos(self, query, num_videos):
        # Implement custom API calls
        pass
```

### Custom Video Effects

Extend `video_composer.py` for additional effects:
```python
def apply_custom_overlay(self, main_batch, overlay_batch, effect_type):
    if effect_type == "fade_in":
        # Implement fade-in effect
        pass
    elif effect_type == "slide_in":
        # Implement slide-in effect
        pass
```

## API Reference

### Main Module Commands
- `python main.py list` - List all available modules
- `python main.py --setup` - Setup workspace directories
- `python main.py --help-module MODULE` - Get help for specific module

### Configuration Commands
- `python config.py setup` - Interactive configuration wizard
- `python config.py pixabay` - Setup Pixabay API only
- `python config.py azure` - Setup Azure Speech Service only
- `python config.py verify` - Verify all configurations

## License and Usage

This project is intended for educational and research purposes. Please comply with the terms of service of all used APIs and services:

- **Azure Speech Service**: [Microsoft Azure Terms](https://azure.microsoft.com/en-us/support/legal/)
- **Pixabay API**: [Pixabay API Terms](https://pixabay.com/api/docs/)
- **Hugging Face Models**: Check individual model licenses
- **Video Content**: Ensure proper licensing for all video content used

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

### Development Setup
```bash
# Clone repository
git clone [repository-url]
cd work_code

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Additional dev tools

# Run tests
python test_system.py
```

## Support and Resources

### Documentation
- [Azure Speech Service Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/)
- [Pixabay API Documentation](https://pixabay.com/api/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### Community Resources
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

For issues and questions, please check the troubleshooting section above or create an issue in the project repository.
