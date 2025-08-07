#!/usr/bin/env python3
"""
Automated Video Production System
====================================

A unified command-line interface for the automated video production system.
This system can automatically insert relevant video clips into financial reporting videos.

Available modules:
1. text-to-speech (tts): Convert text to speech with word boundaries
2. data-collect: Download videos from Pixabay API
3. data-label: Label videos using AI vision models
4. data-match: Find matching videos for text content
5. video-merge: Compose final video with PyTorch GPU acceleration
6. video-merge-simple: Compose final video with MoviePy (simple & auto-download)
7. auto-caption: Add captions to videos using OpenCV GPU acceleration
8. auto-caption-ffmpeg: Add captions to videos using FFmpeg

Usage:
    python main.py <module> [arguments...]
    python main.py <module> -- [--options with dashes]

**Important**: When using parameters with dashes (like --input, --output), you must use the `--` separator:

Examples:
    # Using parameters with dashes (use -- separator)
    python main.py tts -- --input script.txt
    python main.py data-label -- --videos_dir videos --device cpu
    python main.py data-match -- --text "Today's stock market performance is excellent"
    python main.py video-merge -- --json outputs/data_match/matches.json --transcript outputs/tts/script/script.txt --input-video main.mp4 --audio outputs/tts/script/script.wav
    python main.py video-merge-simple -- --json outputs/data_match/matches.json --transcript outputs/tts/script/script.txt --input-video main.mp4 --audio outputs/tts/script/script.wav
    python main.py auto-caption -- --input-video outputs/merge_video.mp4 --script outputs/tts/script/script.txt
    python main.py auto-caption-ffmpeg -- --input-video outputs/merge_video.mp4 --script outputs/tts/script/script.txt --font-size 28
    
    # Using positional arguments (no -- separator needed)
    python main.py data-collect finance 10
    python main.py data-collect university 5
"""

import sys
import os
import argparse
import importlib.util
from pathlib import Path

class VideoProductionSystem:
    """Main system class for managing all video production modules"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.modules = {
            'tts': {
                'name': 'Text-to-Speech',
                'path': self.base_dir / 'text_to_speech' / 'speech_synthesis.py',
                'description': 'Convert text to speech with word boundaries'
            },
            'data-collect': {
                'name': 'Data Collection',
                'path': self.base_dir / 'data_collect_label' / 'data_collect.py',
                'description': 'Download videos from Pixabay API'
            },
            'data-label': {
                'name': 'Data Labeling',
                'path': self.base_dir / 'data_collect_label' / 'data_label.py',
                'description': 'Label videos using AI vision models'
            },
            'data-match': {
                'name': 'Data Matching',
                'path': self.base_dir / 'data_match' / 'semantic_video_matcher.py',
                'description': 'Find matching videos for text content'
            },
            'video-vectorize': {
                'name': 'Video Vectorization',
                'path': self.base_dir / 'data_match' / 'video_vectorizer.py',
                'description': 'Create vector embeddings for videos'
            },
            'video-merge': {
                'name': 'Video Merging (PyTorch)',
                'path': self.base_dir / 'video_merge' / 'video_composer.py',
                'description': 'Compose final video with PyTorch GPU acceleration'
            },
            'video-merge-simple': {
                'name': 'Video Merging (MoviePy)',
                'path': self.base_dir / 'video_merge' / 'video_composer2.py',
                'description': 'Compose final video with MoviePy (simple & auto-download)'
            },
            'auto-caption': {
                'name': 'Auto Caption (OpenCV)',
                'path': self.base_dir / 'auto_caption' / 'caption_generator.py',
                'description': 'Add captions to videos using OpenCV GPU acceleration'
            },
            'auto-caption-ffmpeg': {
                'name': 'Auto Caption (FFmpeg)',
                'path': self.base_dir / 'auto_caption' / 'caption_generator_ffmpeg.py',
                'description': 'Add captions to videos using FFmpeg'
            }
        }
    
    def _add_default_output_paths(self, module_name, args):
        """Add default output paths for modules if not specified"""
        # Create outputs directory structure
        outputs_dir = self.base_dir / 'outputs'
        outputs_dir.mkdir(exist_ok=True)
        
        # Module-specific output path handling
        if module_name == 'tts':
            if '--output' not in args:
                output_dir = outputs_dir / 'tts'
                output_dir.mkdir(exist_ok=True)
                args.extend(['--output', str(output_dir)])
                print(f"Using default output: {output_dir}")
        
        elif module_name == 'video-merge':
            if '--output' not in args:
                output_dir = outputs_dir / 'video_merge'
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / 'final_video_pytorch.mp4'
                args.extend(['--output', str(output_file)])
                print(f"Using default output: {output_file}")
        
        elif module_name == 'video-merge-simple':
            if '--output' not in args:
                output_dir = outputs_dir / 'video_merge'
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / 'final_video_moviepy.mp4'
                args.extend(['--output', str(output_file)])
                print(f"Using default output: {output_file}")
        
        elif module_name == 'data-match':
            if '--output' not in args:
                output_dir = outputs_dir / 'data_match'
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / 'matches.txt'
                args.extend(['--output', str(output_file)])
                print(f"Using default output: {output_file}")
        
        elif module_name == 'auto-caption':
            if '--output' not in args:
                output_dir = outputs_dir / 'auto_caption'
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / 'final_video_opencv.mp4'
                args.extend(['--output', str(output_file)])
                print(f"Using default output: {output_file}")
        
        elif module_name == 'auto-caption-ffmpeg':
            if '--output' not in args:
                output_dir = outputs_dir / 'auto_caption'
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / 'final_video_ffmpeg.mp4'
                args.extend(['--output', str(output_file)])
                print(f"Using default output: {output_file}")
        
        return args
    
    def list_modules(self):
        """List all available modules"""
        print("Available modules:")
        print("=" * 50)
        for key, module in self.modules.items():
            status = "✓" if module['path'].exists() else "✗"
            print(f"{status} {key:<15} - {module['description']}")
        print()
    
    def run_module(self, module_name, args):
        """Run a specific module with given arguments"""
        if module_name not in self.modules:
            print(f"Error: Module '{module_name}' not found.")
            self.list_modules()
            return 1
        
        module_info = self.modules[module_name]
        module_path = module_info['path']
        
        if not module_path.exists():
            print(f"Error: Module file not found: {module_path}")
            return 1
        
        # Add default output paths for modules that support it
        args = self._add_default_output_paths(module_name, args)
        
        print(f"Running {module_info['name']}...")
        print(f"Module: {module_path}")
        print(f"Arguments: {' '.join(args)}")
        print("-" * 50)
        
        # Import and run the module
        try:
            # Use subprocess to run the module, which ensures if __name__ == "__main__" works correctly
            import subprocess
            
            cmd = [sys.executable, str(module_path)] + args
            print(f"Executing: {' '.join(cmd)}")
            
            # Run the module as a subprocess
            result = subprocess.run(cmd, cwd=self.base_dir)
            return result.returncode
                
        except Exception as e:
            print(f"Error running module {module_name}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def show_module_help(self, module_name):
        """Show help for a specific module"""
        if module_name not in self.modules:
            print(f"Error: Module '{module_name}' not found.")
            return
        
        module_info = self.modules[module_name]
        print(f"Help for {module_info['name']}:")
        print("=" * 50)
        print(f"Description: {module_info['description']}")
        print(f"Module path: {module_info['path']}")
        print()
        
        # Try to get help from the module
        try:
            self.run_module(module_name, ['--help'])
        except SystemExit:
            pass  # argparse calls sys.exit(), which is expected
    
    def setup_workspace(self):
        """Set up the workspace with necessary directories and files"""
        print("Setting up workspace...")
        
        # Create necessary directories
        directories = [
            # Input directories (within modules)
            'text_to_speech/input_text',
            'data_collect_label',
            'data_match/input_text',
            'video_merge',
            'auto_caption',
            # Unified output directories
            'outputs/tts',
            'outputs/data_collection/videos',
            'outputs/data_match/vectors',
            'outputs/data_match/results',
            'outputs/video_merge',
            'outputs/auto_caption'
        ]
        
        for dir_path in directories:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
        
        # Create sample config file if it doesn't exist
        config_path = self.base_dir / 'data_collect_label' / 'config.yaml'
        if not config_path.exists():
            config_content = """pixabay:
  api_key: "YOUR_API_KEY_HERE"
"""
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            print(f"✓ Created sample config: {config_path}")
        
        print("Workspace setup complete!")

def create_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'module',
        nargs='?',
        help='Module to run (use "list" to see available modules)'
    )
    
    parser.add_argument(
        'module_args',
        nargs='*',
        help='Arguments to pass to the module'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Set up workspace directories and sample files'
    )
    
    parser.add_argument(
        '--help-module',
        metavar='MODULE',
        help='Show help for a specific module'
    )
    
    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    system = VideoProductionSystem()
    
    # Handle setup
    if args.setup:
        system.setup_workspace()
        return 0
    
    # Handle module help
    if args.help_module:
        system.show_module_help(args.help_module)
        return 0
    
    # Handle no module specified or list command
    if not args.module or args.module == 'list':
        print(__doc__)
        system.list_modules()
        return 0
    
    # Run the specified module
    return system.run_module(args.module, args.module_args)

if __name__ == '__main__':
    sys.exit(main())
