#!/usr/bin/env python3
"""
字體和中文渲染診斷工具
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def check_chinese_fonts():
    """檢查可用的中文字體"""
    font_paths = [
        # Linux Chinese fonts (common locations)
        "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",  # Common CJK font
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Google Noto CJK
        "/usr/share/fonts/truetype/arphic/uming.ttc",  # AR PL UMing
        "/usr/share/fonts/truetype/arphic/ukai.ttc",   # AR PL UKai
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # WenQuanYi Micro Hei
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # WenQuanYi Zen Hei
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # Ubuntu/Debian specific paths
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    
    print("=== 字體檢查 ===")
    available_fonts = []
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 24)
                available_fonts.append((font_path, font))
                print(f"✅ 可用: {font_path}")
            except Exception as e:
                print(f"❌ 無法載入: {font_path} - {e}")
        else:
            print(f"❌ 不存在: {font_path}")
    
    return available_fonts

def test_chinese_rendering(font_path, font):
    """測試中文字符渲染"""
    print(f"\n=== 測試字體: {font_path} ===")
    
    # 測試文本
    test_texts = [
        "你好, 我係Christina",
        "美國股市市場分析同預測",
        "標普500同納指屢創歷史新高",
        "English text test 123"
    ]
    
    for text in test_texts:
        try:
            # 創建測試圖像
            img = Image.new('RGB', (800, 100), color='black')
            draw = ImageDraw.Draw(img)
            
            # 渲染文本
            draw.text((10, 10), text, font=font, fill='white')
            
            # 轉換為OpenCV格式
            cv_img = np.array(img)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            
            # 保存測試圖像
            safe_filename = f"test_font_{os.path.basename(font_path)}_{hash(text) % 10000}.png"
            cv2.imwrite(safe_filename, cv_img)
            
            print(f"✅ 成功渲染: '{text}' -> {safe_filename}")
            
        except Exception as e:
            print(f"❌ 渲染失敗: '{text}' - {e}")

def install_fonts_suggestion():
    """提供字體安裝建議"""
    print("\n=== 字體安裝建議 ===")
    print("如果沒有找到中文字體，請嘗試安裝：")
    print("Ubuntu/Debian: sudo apt-get install fonts-noto-cjk fonts-wqy-microhei")
    print("CentOS/RHEL: sudo yum install google-noto-cjk-fonts wqy-microhei-fonts")
    print("或者：sudo dnf install google-noto-cjk-fonts wqy-microhei-fonts")

def main():
    print("中文字體和渲染診斷工具")
    print("=" * 50)
    
    # 檢查可用字體
    available_fonts = check_chinese_fonts()
    
    if not available_fonts:
        print("\n❌ 沒有找到任何可用的字體！")
        install_fonts_suggestion()
        return 1
    
    print(f"\n✅ 找到 {len(available_fonts)} 個可用字體")
    
    # 測試第一個可用字體
    if available_fonts:
        font_path, font = available_fonts[0]
        test_chinese_rendering(font_path, font)
        
        print(f"\n建議使用字體: {font_path}")
    
    # 檢查字體目錄
    print(f"\n=== 系統字體目錄 ===")
    font_dirs = [
        "/usr/share/fonts/",
        "/usr/local/share/fonts/",
        "~/.fonts/",
        "/System/Library/Fonts/"
    ]
    
    for font_dir in font_dirs:
        expanded_dir = os.path.expanduser(font_dir)
        if os.path.exists(expanded_dir):
            print(f"✅ {expanded_dir}")
            # 列出一些字體文件
            try:
                files = os.listdir(expanded_dir)
                font_files = [f for f in files if f.endswith(('.ttf', '.ttc', '.otf'))][:5]
                for font_file in font_files:
                    print(f"   - {font_file}")
                if len(font_files) == 5:
                    print(f"   ... (還有更多)")
            except:
                pass
        else:
            print(f"❌ {expanded_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
