#!/usr/bin/env python3
"""
Chinese Font Downloader
======================

Downloads Chinese fonts to user's local font directory without admin privileges.
Uses multiple sources for reliability.
"""

import os
import sys
import urllib.request
import shutil
import subprocess
from pathlib import Path
import tempfile
import zipfile
import platform
import time

def download_file(url, destination, timeout=60):
    """Download a file from URL to destination"""
    print(f"Downloading from {url}")
    print(f"Saving to {destination}")
    
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(destination, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            file_size = os.path.getsize(destination)
            print(f"Download complete: {file_size/1024:.1f} KB")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def download_wqy_microhei():
    """Download WenQuanYi Microhei font - good fallback for Chinese text"""
    # Create font directories
    user_font_dir = Path.home() / ".local" / "share" / "fonts"
    user_font_dir.mkdir(parents=True, exist_ok=True)
    
    local_font_dir = Path("./fonts")
    local_font_dir.mkdir(exist_ok=True)
    
    # WenQuanYi Microhei direct download URL
    wqy_url = "https://sourceforge.net/projects/wqy/files/wqy-microhei/0.2.0-beta/wqy-microhei-0.2.0-beta.tar.gz/download"
    
    print("Attempting to download WenQuanYi Microhei font...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download tar.gz file
        tar_path = os.path.join(temp_dir, "wqy-microhei.tar.gz")
        if not download_file(wqy_url, tar_path):
            print("Failed to download WenQuanYi Microhei")
            return False
        
        # Extract tar.gz file
        try:
            print("Extracting files...")
            import tarfile
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=temp_dir)
            
            # Find the font file in extracted directory
            font_file = None
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.ttc') or file.endswith('.ttf'):
                        font_file = os.path.join(root, file)
                        break
                if font_file:
                    break
            
            if not font_file:
                print("Could not find font file in extracted archive")
                return False
            
            # Copy to user font directory
            user_font_path = user_font_dir / os.path.basename(font_file)
            local_font_path = local_font_dir / os.path.basename(font_file)
            
            shutil.copy(font_file, user_font_path)
            shutil.copy(font_file, local_font_path)
            
            print(f"WenQuanYi Microhei installed: {user_font_path}")
            print(f"Local copy: {local_font_path}")
            return True
            
        except Exception as e:
            print(f"Error extracting or installing font: {e}")
            return False

def download_google_noto_sc():
    """Download Google Noto Sans SC (Simplified Chinese)"""
    # Direct link to Noto Sans SC Regular
    url = "https://fonts.google.com/download?family=Noto%20Sans%20SC"
    
    # Create font directories
    user_font_dir = Path.home() / ".local" / "share" / "fonts"
    user_font_dir.mkdir(parents=True, exist_ok=True)
    
    local_font_dir = Path("./fonts")
    local_font_dir.mkdir(exist_ok=True)
    
    print("Attempting to download Noto Sans SC font...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download zip file
        zip_path = os.path.join(temp_dir, "notosanssc.zip")
        if not download_file(url, zip_path):
            print("Failed to download Noto Sans SC")
            return False
        
        try:
            # Extract zip file
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the Regular font file
            font_file = None
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('Regular.otf') or file.endswith('Regular.ttf'):
                        font_file = os.path.join(root, file)
                        break
                if font_file:
                    break
            
            if not font_file:
                print("Could not find font file in extracted archive")
                return False
            
            # Copy to user font directory
            user_font_path = user_font_dir / "NotoSansSC-Regular.otf"
            local_font_path = local_font_dir / "NotoSansSC-Regular.otf"
            
            shutil.copy(font_file, user_font_path)
            shutil.copy(font_file, local_font_path)
            
            print(f"Noto Sans SC installed: {user_font_path}")
            print(f"Local copy: {local_font_path}")
            return True
            
        except Exception as e:
            print(f"Error extracting or installing font: {e}")
            return False

def main():
    """Main function to download and install Chinese fonts"""
    print("Chinese Font Downloader")
    print("======================")
    print(f"User: {os.getenv('USER')}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print("======================")
    
    # Create font directories
    user_font_dir = Path.home() / ".local" / "share" / "fonts"
    user_font_dir.mkdir(parents=True, exist_ok=True)
    
    local_font_dir = Path("./fonts")
    local_font_dir.mkdir(exist_ok=True)
    
    print(f"Font directories created:")
    print(f"  - {user_font_dir}")
    print(f"  - {local_font_dir}")
    
    success = False
    
    # Try downloading WenQuanYi Microhei (more reliable download)
    if download_wqy_microhei():
        success = True
    
    # Try downloading Google Noto Sans SC
    if download_google_noto_sc():
        success = True
    
    # Update font cache if fc-cache is available
    try:
        print("Updating font cache...")
        subprocess.run(['fc-cache', '-f'], check=False)
        print("Font cache updated")
    except:
        print("Could not update font cache (fc-cache not available)")
    
    if success:
        print("\nSuccessfully installed at least one Chinese font!")
        print("\nTo use these fonts in your script, add the following paths:")
        print(f"  {user_font_dir}/wqy-microhei.ttc")
        print("  or")
        print("  ./fonts/wqy-microhei.ttc")
    else:
        print("\nFailed to download any fonts. Please check your internet connection.")
        print("You may need to manually download a Chinese font.")

if __name__ == "__main__":
    main()