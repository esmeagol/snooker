#!/usr/bin/env python3
"""
Command-line interface for the Snooker Training Data Pipeline.
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .pipeline import run_pipeline

def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        sys.exit(1)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Snooker Training Data Pipeline - Generate training data for object detection models"
    )
    
    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['download', 'preprocess', 'split', 'extract', 'all'],
        default=['all'],
        help='Pipeline steps to run (default: all)'
    )
    
    parser.add_argument(
        '--youtube-urls',
        type=str,
        help='Path to file containing YouTube URLs (overrides config file)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for processed data (overrides config file)'
    )
    
    parser.add_argument(
        '--cookies',
        type=str,
        help='Path to cookies file for YouTube (overrides config file)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.youtube_urls:
        config['youtube_urls'] = args.youtube_urls
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.cookies:
        config['cookies_file'] = args.cookies
    
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Normalize steps
    steps = args.steps
    if 'all' in steps:
        steps = ['download', 'preprocess', 'split', 'extract']
    
    # Run the pipeline
    try:
        run_pipeline(config, steps)
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
