#!/usr/bin/env python3
"""
Configuration Management Tool for Automated Video Production System
==================================================================

This tool helps you set up and manage API keys and configuration files.
"""

import os
import sys
import yaml
import json
from pathlib import Path


class ConfigManager:
    """Manage configuration files and API keys"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / 'data_collect_label' / 'config.yaml'
        self.env_file = self.base_dir / '.env'

    def setup_pixabay_config(self):
        """Set up Pixabay API configuration"""
        print("Setting up Pixabay API configuration")
        print("=" * 40)
        print("You can get your API key from: https://pixabay.com/api/docs/")
        print()

        api_key = input("Enter your Pixabay API key: ").strip()

        if not api_key:
            print("ERROR: API key cannot be empty")
            return False

        # Create config directory if it doesn't exist
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new one
        config = {}
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

        # Update config
        config['pixabay'] = {'api_key': api_key}

        # Save config
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✓ Pixabay API key saved to {self.config_file}")
        return True

    def setup_azure_speech_config(self):
        """Set up Azure Speech Service configuration"""
        print("Setting up Azure Speech Service configuration")
        print("=" * 45)
        print("You can get your keys from: https://portal.azure.com")
        print("Navigate to: Cognitive Services > Speech Service")
        print()

        speech_key = input("Enter your Speech Service key: ").strip()
        endpoint = input("Enter your Speech Service endpoint: ").strip()

        if not speech_key or not endpoint:
            print("ERROR: Both speech key and endpoint are required")
            return False

        # Set environment variables for current session
        os.environ['SPEECH_KEY'] = speech_key
        os.environ['ENDPOINT'] = endpoint

        print("✓ Environment variables set for current session")

        # Save to .env file for future sessions
        env_content = f"""# Azure Speech Service Configuration
SPEECH_KEY={speech_key}
ENDPOINT={endpoint}
"""

        with open(self.env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)

        print(f"✓ Configuration saved to {self.env_file}")
        print()
        print("To use these settings in future sessions, run:")
        print("  Windows: config.bat load-env")
        print("  Or manually: set SPEECH_KEY=your_key && set ENDPOINT=your_endpoint")

        return True

    def load_env_variables(self):
        """Load environment variables from .env file"""
        if not self.env_file.exists():
            print("No .env file found. Please run configuration setup first.")
            return False

        print("Loading environment variables from .env file...")

        with open(self.env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    print(f"✓ Set {key}")

        return True

    def verify_configuration(self):
        """Verify that all required configurations are set"""
        print("Verifying configuration...")
        print("=" * 30)

        errors = []

        # Check Pixabay config
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                if config and 'pixabay' in config and 'api_key' in config['pixabay']:
                    api_key = config['pixabay']['api_key']
                    if api_key and api_key != 'YOUR_API_KEY_HERE':
                        print("✓ Pixabay API key configured")
                    else:
                        errors.append(
                            "Pixabay API key not properly configured")
                else:
                    errors.append("Pixabay configuration missing or invalid")
            except Exception as e:
                errors.append(f"Error reading Pixabay config: {e}")
        else:
            errors.append("Pixabay config file not found")

        # Check Azure Speech Service config
        speech_key = os.environ.get('SPEECH_KEY')
        endpoint = os.environ.get('ENDPOINT')

        if speech_key and endpoint:
            print("✓ Azure Speech Service environment variables set")
        else:
            errors.append("Azure Speech Service environment variables not set")

        # Check directories
        required_dirs = [
            'text_to_speech/input_text',
            'text_to_speech/output',
            'data_collect_label/videos',
            'data_match/vectors',
            'data_match/output'
        ]

        for dir_path in required_dirs:
            full_path = self.base_dir / dir_path
            if full_path.exists():
                print(f"✓ Directory exists: {dir_path}")
            else:
                errors.append(f"Directory missing: {dir_path}")

        if errors:
            print("\n❌ Configuration issues found:")
            for error in errors:
                print(f"   - {error}")
            print("\nRun 'python config.py setup' to fix these issues.")
            return False
        else:
            print("\n✅ All configurations are valid!")
            return True

    def interactive_setup(self):
        """Interactive setup wizard"""
        print("Automated Video Production System - Setup Wizard")
        print("=" * 50)
        print()

        # Setup workspace
        print("1. Setting up workspace directories...")
        os.system(f'python "{self.base_dir / "main.py"}" --setup')
        print()

        # Setup Pixabay
        print("2. Setting up Pixabay API...")
        while True:
            if self.setup_pixabay_config():
                break
            retry = input("Would you like to try again? (y/n): ").lower()
            if retry != 'y':
                break
        print()

        # Setup Azure Speech
        print("3. Setting up Azure Speech Service...")
        while True:
            if self.setup_azure_speech_config():
                break
            retry = input("Would you like to try again? (y/n): ").lower()
            if retry != 'y':
                break
        print()

        # Verify configuration
        print("4. Verifying configuration...")
        self.verify_configuration()
        print()

        print("Setup completed! You can now use the system:")
        print("  python main.py list              # List all modules")
        print("  python main.py tts               # Test text-to-speech")
        print("  python main.py data-collect finance 5  # Download videos")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', default='setup', choices=[
        'setup', 'pixabay', 'azure', 'verify', 'load-env'
    ], help='Configuration command to run')

    args = parser.parse_args()

    config_manager = ConfigManager()

    if args.command == 'setup':
        config_manager.interactive_setup()
    elif args.command == 'pixabay':
        config_manager.setup_pixabay_config()
    elif args.command == 'azure':
        config_manager.setup_azure_speech_config()
    elif args.command == 'verify':
        config_manager.verify_configuration()
    elif args.command == 'load-env':
        config_manager.load_env_variables()


if __name__ == '__main__':
    main()
