#!/usr/bin/env python3
"""
測試系統字體中文支持情況
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont


def test_font(font_path, text="測試中文字體"):
    """測試字體是否支持中文"""
    try:
        # 加載字體
        font = ImageFont.truetype(font_path, 24)

        # 創建測試圖像
        img = Image.new('RGB', (300, 50), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 嘗試繪製中文文本
        draw.text((10, 10), text, font=font, fill=(0, 0, 0))

        # 獲取文本邊界框
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]

        if text_width > 0:
            print(f"✅ {os.path.basename(font_path)} 支持中文渲染")
            return True
        else:
            print(f"❌ {os.path.basename(font_path)} 不支持中文渲染")
            return False
    except Exception as e:
        print(f"❌ {os.path.basename(font_path)} 測試失敗: {e}")
        return False


def scan_fonts_directory(directory):
    """掃描目錄中的所有字體文件並測試"""
    print(f"\n掃描目錄: {directory}")
    if not os.path.exists(directory):
        print(f"目錄不存在: {directory}")
        return []

    supported_fonts = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                font_path = os.path.join(root, file)
                if test_font(font_path):
                    supported_fonts.append(font_path)

    return supported_fonts


def main():
    """主函數"""
    print("中文字體支持測試")
    print("=" * 50)

    # 常見的字體目錄
    font_dirs = [
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/opentype",
        "/usr/local/share/fonts",
        "~/.fonts",
        "~/.local/share/fonts",
        "/usr/X11R6/lib/X11/fonts",
        "/Library/Fonts",  # macOS
        "/System/Library/Fonts"  # macOS
    ]

    all_supported = []

    for directory in font_dirs:
        expanded_dir = os.path.expanduser(directory)
        supported = scan_fonts_directory(expanded_dir)
        all_supported.extend(supported)

    print("\n結果摘要:")
    print(f"找到 {len(all_supported)} 個支持中文的字體")

    if all_supported:
        print("\n推薦字體:")
        for font in all_supported[:5]:  # 只顯示前5個
            print(f"- {font}")
    else:
        print("\n沒有找到支持中文的字體，建議安裝:")
        print("Ubuntu/Debian: sudo apt-get install fonts-noto-cjk fonts-wqy-microhei")
        print("Fedora/RHEL: sudo dnf install google-noto-sans-cjk-fonts wqy-microhei-fonts")


if __name__ == "__main__":
    main()
