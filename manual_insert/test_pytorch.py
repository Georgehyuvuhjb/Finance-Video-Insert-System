#!/usr/bin/env python3
"""
Test script for PyTorch-based manual video inserter
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manual_insert.manual_inserter import PyTorchVideoProcessor

def test_pytorch_processor():
    """Test basic PyTorch video processor functionality"""
    print("Testing PyTorch Video Processor...")
    
    # Initialize processor
    try:
        processor = PyTorchVideoProcessor(use_gpu=False)  # Use CPU for testing
        print("✅ PyTorchVideoProcessor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PyTorchVideoProcessor: {e}")
        return False
    
    # Test tensor operations
    try:
        import torch
        test_tensor = torch.randn(1, 3, 480, 640)  # Sample video frame tensor
        print(f"✅ Created test tensor with shape: {test_tensor.shape}")
        
        # Test resizing
        resized = processor.resize_frames(test_tensor, (240, 320))
        print(f"✅ Resized tensor to shape: {resized.shape}")
        
        # Test size calculation
        width, height = processor.calculate_size("25%", 1920, 1080)
        print(f"✅ Size calculation: 25% of 1920x1080 = {width}x{height}")
        
        # Test position calculation
        x, y = processor.calculate_center_position("center", 1920, 1080, 480, 270)
        print(f"✅ Center position calculation: ({x}, {y})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during tensor operations: {e}")
        return False

def test_main_class():
    """Test main ManualVideoInserter class"""
    print("\nTesting ManualVideoInserter class...")
    
    try:
        from manual_insert.manual_inserter import ManualVideoInserter
        inserter = ManualVideoInserter(use_gpu=False)
        print("✅ ManualVideoInserter initialized successfully")
        
        # Test time parsing
        seconds = inserter.parse_time_to_seconds("01:30.50")
        print(f"✅ Time parsing: 01:30.50 = {seconds} seconds")
        
        time_str = inserter.seconds_to_time_str(90.5)
        print(f"✅ Time formatting: 90.5 seconds = {time_str}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ManualVideoInserter: {e}")
        return False

def main():
    """Run all tests"""
    print("PyTorch Manual Video Inserter Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test 1: PyTorch processor
    success &= test_pytorch_processor()
    
    # Test 2: Main class
    success &= test_main_class()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! PyTorch version is working correctly.")
        print("\nNext steps:")
        print("1. Test with actual video files using the test configuration")
        print("2. Run: python manual_inserter.py --config test_pytorch_version.yaml --input-video ../input/video.mp4 --output test_output.mp4")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()
