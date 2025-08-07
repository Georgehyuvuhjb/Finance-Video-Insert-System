# Automated Video Production System

An automated video production system that can automatically insert relevant video clips into financial reporting videos using AI-driven content matching and video composition techniques.

<!--
## Recent Bug Fixes and Improvements

### Video Matching Logic Fix (Latest)

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

Find matching videos for text content using semantic search.

```bash
# Direct text input
python main.py data-match -- --text "Stock market performance today shows positive trends"

# File input
python main.py data-match -- --file input_text/script.txt

# Specify output files
python main.py data-match -- --file script.txt --output results.txt

# Custom parameters
python main.py data-match -- --text "finance news" --max-group-size 5 --num-videos 3 --top-k 5
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

**Grouping Parameters:**
- Sentences are automatically grouped by semantic similarity
- `max_sentences_per_group`: Controls group size (default: 3)
- `similarity_threshold`: Threshold for grouping (default: 0.70, configurable in code)

**Advanced Usage:**
```bash
# Large groups with more video options
python main.py data-match -- --file script.txt --max-group-size 10 --num-videos 5 --top-k 10

# Custom vector directory and metadata
python main.py data-match -- --text "market analysis" --vectors custom_vectors --metadata custom_metadata.json
```

**Interactive Mode:**
```bash
# Enter text interactively (press Ctrl+D/Ctrl+Z when finished)
python main.py data-match
```

**Prerequisites:**
- Created video vector index
- Labeled video database

**Output:**
- Text format: Formatted results with sentence groups and matching videos
- JSON format: Structured data for video composition (automatically created with .json extension)

### Step 6: Video Composition

Compose final video by inserting matched video clips at specific timestamps. Choose between two methods:

#### Method 1: PyTorch GPU-Accelerated (Recommended for large projects)

```bash
# Basic video composition with PyTorch
python main.py video-merge -- \
    --json data_match/output/matches.json \
    --transcript text_to_speech/output/script/script.txt \
    --input-video main_video.mp4 \
    --videos-dir data_collect_label/videos \
    --output final_video.mp4

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
    --json data_match/output/matches.json \
    --transcript text_to_speech/output/script/script.txt \
    --input-video main_video.mp4 \
    --output final_video_simple.mp4

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

#### 1. Module Not Found Errors
```bash
# Ensure you're in the correct directory
cd /path/to/work_code

# Check Python path
python -c "import sys; print(sys.path)"

# Install missing packages
pip install -r requirements.txt
```

#### 2. Azure Speech Service Errors
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

#### 3. Pixabay Download Failures
```bash
# Verify API key in config.yaml
python config.py verify

# Check network connectivity
curl -I https://pixabay.com/api/videos/
```

**API Limits:**
- Free tier: 20,000 requests/month
- Rate limit: 3 requests/minute

#### 4. Video Processing Memory Issues

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

#### 5. GPU Memory Insufficient

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

#### 6. Model Download Issues

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

# Use appropriate distance thresholds
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
