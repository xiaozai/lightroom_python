#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python Lightroom Tool - 命令行接口

用法:
    python cli.py --input input.jpg --param params.json --output output.jpg
    python cli.py -i input.jpg -p params.json -o output.jpg

参数:
    --input, -i      输入图像文件路径 (支持 JPG, PNG, BMP, TIFF)
    --param, -p      参数JSON文件路径
    --output, -o     输出图像文件路径
    --help, -h       显示帮助信息
"""

import argparse
import json
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.editor import ImageEditor


def main():
    parser = argparse.ArgumentParser(
        description='Python Lightroom Tool - 命令行图像编辑工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python cli.py -i photo.jpg -p preset.json -o edited.jpg
  python cli.py --input photo.jpg --param preset.json --output edited.jpg

参数JSON文件格式:
  {
    "exposure": 10,
    "contrast": 15,
    "highlights": -20,
    "shadows": 30,
    "whites": 5,
    "blacks": -10,
    "texture": 20,
    "clarity": 25,
    "dehaze": 10,
    "vibrance": 15,
    "saturation": 10,
    "temp": 5,
    "tint": 3,
    "hsl_hue_red": 5,
    "hsl_sat_red": 10,
    "hsl_lum_red": -5,
    "cg_shadows_hue": 240,
    "cg_shadows_sat": 30,
    "cg_highlights_hue": 60,
    "cg_highlights_sat": 20,
    "cg_blending": 50,
    "cg_balance": 0,
    "sharpen_amount": 50,
    "sharpen_radius": 1.0,
    "sharpen_detail": 25,
    "sharpen_masking": 0,
    "noise_luminance": 10,
    "noise_color": 10,
    "curve_rgb": [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
    "curve_red": [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
    "curve_green": [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
    "curve_blue": [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
    "curve_saturation": 0
  }

说明:
  - curve_*: 色调曲线控制点，每个点是 [x, y] 坐标，范围 0-255
  - curve_rgb: RGB复合曲线
  - curve_red/green/blue: 分通道曲线
  - curve_saturation: 曲线饱和度调整 (-100 到 100)
        '''
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入图像文件路径'
    )

    parser.add_argument(
        '-p', '--param',
        required=True,
        help='参数JSON文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出图像文件路径'
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.param):
        print(f"错误: 参数文件不存在: {args.param}")
        sys.exit(1)

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载参数文件
    try:
        with open(args.param, 'r', encoding='utf-8') as f:
            params = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 参数文件JSON格式错误: {e}")
        sys.exit(1)

    # 创建编辑器并加载图像
    print(f"加载图像: {args.input}")
    editor = ImageEditor()
    try:
        editor.load_image(args.input)
    except Exception as e:
        print(f"错误: 无法加载图像: {e}")
        sys.exit(1)

    # 应用参数
    print(f"应用参数: {args.param}")
    editor.set_params_from_dict(params)

    # 保存输出图像
    print(f"保存图像: {args.output}")
    try:
        editor.save_image(args.output)
        print("完成!")
    except Exception as e:
        print(f"错误: 无法保存图像: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
