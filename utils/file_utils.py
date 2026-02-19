"""文件操作工具函数"""
from pathlib import Path


def get_supported_image_filters():
    """获取支持的图像文件过滤器"""
    return "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)"


def get_save_filters():
    """获取保存文件的过滤器"""
    return "JPEG (*.jpg *.jpeg);;PNG (*.png);;BMP (*.bmp)"


def get_file_extension(filter_text: str) -> str:
    """根据过滤器文本获取文件扩展名"""
    if "JPEG" in filter_text or "jpg" in filter_text.lower():
        return ".jpg"
    elif "PNG" in filter_text:
        return ".png"
    elif "BMP" in filter_text:
        return ".bmp"
    return ".jpg"


def is_supported_image(file_path: str) -> bool:
    """检查文件是否为支持的图像格式"""
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    return Path(file_path).suffix.lower() in supported_extensions
