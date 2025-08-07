# Pixabay Video Downloader Tool

## Project Overview

This is a command-line tool for downloading royalty-free video materials from Pixabay. Using Pixabay's official API, you can search and download high-quality videos on various topics, perfect for educational, presentation, or creative projects.

## Features

- Search for videos on Pixabay based on keywords
- Support for multi-keyword search (using + connection)
- Customizable download quantity
- Automatic selection of highest available quality
- Video metadata saving
- YAML configuration file for API key storage

## Installing Dependencies

Before using this tool, you need to install the following Python libraries:

```bash
pip install requests pyyaml
```

## How to Use

### 1. Get a Pixabay API Key

1. Register an account at [Pixabay](https://pixabay.com/accounts/register/)
2. After logging in, visit the [API Documentation Page](https://pixabay.com/api/docs/) to get your API key

### 2. Configure API Key

When you run the program for the first time, it will automatically create a `config.yaml` configuration file template. You need to edit this file and replace `YOUR_API_KEY_HERE` with your actual API key:

```yaml
# Pixabay API Configuration
pixabay:
  api_key: "YOUR_API_KEY_HERE"  # Replace with your API key
```

### 3. Run the Program

Basic usage:

```bash
python data_collect.py keyword download_count
```

For example:

```bash
python data_collect.py finance 5
```

This will search for and download 5 videos related to "finance" to the `finance_videos` folder.

#### Multi-keyword search:

Connect multiple keywords using `+`:

```bash
python data_collect.py finance+business 10
```

This will search for and download 10 videos related to "finance" and "business" to the `finance+business_videos` folder.

#### Using a custom configuration file:

```bash
python data_collect.py finance 5 --config my_config.yaml
```

## Output Results

- Video files: Saved in a folder named after the search term (e.g., `finance_videos`)
- Naming format: `number_keyword_videoID_quality.mp4`
- Metadata: Each video has a corresponding JSON format metadata file

## Notes

1. Pixabay API has a limit of 100 requests per minute
2. Downloaded videos should comply with Pixabay's [Terms of Service](https://pixabay.com/service/terms/)
3. If you use these videos in public projects, it's recommended to credit Pixabay as the source
4. Systematic mass downloading of videos is not allowed


## Author

Georgehyuvuhjb

## Last Updated

July 30, 2025