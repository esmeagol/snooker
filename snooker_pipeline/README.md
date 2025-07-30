# Snooker Training Data Pipeline

A unified pipeline for generating training data for snooker object detection models from YouTube videos.

## Features

- **YouTube Video Downloading**: Download videos from a list of YouTube URLs
- **Video Preprocessing**: Process videos to extract relevant frames with full table view
- **Dataset Splitting**: Automatically split videos into train/val/test sets (70/20/10 by default)
- **Frame Extraction**: Extract frames from videos at specified intervals
- **Modular Design**: Run individual steps or the entire pipeline
- **Configuration File**: Easy configuration via YAML file

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd snooker/snooker_pipeline
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install them manually:
   ```bash
   pip install opencv-python tqdm pyyaml yt-dlp numpy
   ```

3. Install yt-dlp (for YouTube video downloading):
   ```bash
   pip install -U yt-dlp
   ```

## Usage

1. **Create a configuration file** (see `example_config.yaml` for reference)

2. **Prepare a text file** with YouTube URLs (one URL per line)

3. **Run the pipeline**:
   ```bash
   python -m snooker_pipeline.cli --config your_config.yaml
   ```

### Command Line Options

```
usage: cli.py [-h] --config CONFIG [--steps {download,preprocess,split,extract,all} ...]
              [--youtube-urls YOUTUBE_URLS] [--output-dir OUTPUT_DIR]
              [--cookies COOKIES] [-v]

Snooker Training Data Pipeline - Generate training data for object detection models

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the YAML configuration file
  --steps {download,preprocess,split,extract,all}
                        Pipeline steps to run (default: ['all'])
  --youtube-urls YOUTUBE_URLS
                        Path to file containing YouTube URLs (overrides config file)
  --output-dir OUTPUT_DIR
                        Output directory for processed data (overrides config file)
  --cookies COOKIES     Path to cookies file for YouTube (overrides config file)
  -v, --verbose         Enable verbose output
```

### Example

1. Create a config file (`config.yaml`):
   ```yaml
   raw_videos_dir: "data/raw_videos"
   processed_videos_dir: "data/processed_videos"
   output_dir: "data/dataset"
   youtube_urls: "youtube_urls.txt"
   cookies_file: "youtube_cookies.txt"
   target_fps: 30
   min_table_visibility: 0.4
   frame_interval: 10
   train_ratio: 0.7
   val_ratio: 0.2
   test_ratio: 0.1
   ```

2. Create a file with YouTube URLs (`youtube_urls.txt`):
   ```
   https://www.youtube.com/watch?v=EXAMPLE1
   https://www.youtube.com/watch?v=EXAMPLE2
   ```

3. Run the pipeline:
   ```bash
   python -m snooker_pipeline.cli --config config.yaml
   ```

## Pipeline Steps

1. **Download**: Download videos from YouTube
2. **Preprocess**: Process videos to extract relevant frames with full table view
3. **Split**: Split videos into train/val/test sets
4. **Extract**: Extract frames from videos for each split

## Output Structure

```
data/
├── raw_videos/          # Downloaded YouTube videos
├── processed_videos/    # Preprocessed videos
└── dataset/             # Final dataset
    ├── train/           # Training set
    │   ├── images/      # Training images
    │   └── labels/      # Training labels (empty, to be annotated)
    ├── val/             # Validation set
    │   ├── images/      # Validation images
    │   └── labels/      # Validation labels (empty, to be annotated)
    └── test/            # Test set
        ├── images/      # Test images
        └── labels/      # Test labels (empty, to be annotated)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
