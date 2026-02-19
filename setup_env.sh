#!/bin/bash
# 创建conda环境并安装依赖

echo "Creating conda environment 'lightroom'..."
conda env create -f environment.yml

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate lightroom"
echo ""
echo "To run the application:"
echo "  python main.py"
