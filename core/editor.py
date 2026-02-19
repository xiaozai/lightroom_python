"""图像编辑核心功能 - 非破坏性编辑"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EditParams:
    """编辑参数数据类"""
    # 基本调整参数
    exposure: float = 0.0       # 曝光: -5.0 到 +5.0
    contrast: float = 0.0       # 对比度: -100 到 +100
    highlights: float = 0.0     # 高光: -100 到 +100
    shadows: float = 0.0        # 阴影: -100 到 +100
    whites: float = 0.0         # 白色色阶: -100 到 +100
    blacks: float = 0.0         # 黑色色阶: -100 到 +100
    texture: float = 0.0        # 纹理: -100 到 +100
    clarity: float = 0.0        # 清晰度: -100 到 +100
    dehaze: float = 0.0         # 去朦胧: -100 到 +100
    vibrance: float = 0.0       # 鲜艳度: -100 到 +100
    saturation: float = 0.0     # 饱和度: -100 到 +100


class ImageEditor:
    """图像编辑器类 - 支持非破坏性编辑"""

    # 预览图像的最大尺寸
    PREVIEW_MAX_SIZE = 800

    def __init__(self):
        self._original_image: Optional[Image.Image] = None
        self._preview_image: Optional[Image.Image] = None  # 缩放后的预览图
        self._params = EditParams()
        self._cached_edited: Optional[Image.Image] = None
        self._cache_valid = False

    def load_image(self, path: str) -> Image.Image:
        """加载图像

        Args:
            path: 图像文件路径

        Returns:
            加载的PIL Image对象
        """
        self._original_image = Image.open(path)
        # 转换为RGB模式（如果是RGBA则保留alpha通道）
        if self._original_image.mode not in ('RGB', 'RGBA'):
            self._original_image = self._original_image.convert('RGB')

        # 创建预览用的缩放图像
        self._create_preview_image()

        self._params = EditParams()  # 重置参数
        self._cache_valid = False
        return self._original_image

    def _create_preview_image(self):
        """创建预览用的缩放图像"""
        if self._original_image is None:
            return

        width, height = self._original_image.size
        max_dim = max(width, height)

        if max_dim > self.PREVIEW_MAX_SIZE:
            scale = self.PREVIEW_MAX_SIZE / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            self._preview_image = self._original_image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )
        else:
            self._preview_image = self._original_image.copy()

    def save_image(self, path: str) -> bool:
        """保存编辑后的图像（全分辨率）

        Args:
            path: 保存路径

        Returns:
            是否保存成功
        """
        if self._original_image is None:
            return False
        try:
            # 保存时使用全分辨率图像
            edited = self._apply_edits_fullres()
            edited.save(path)
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False

    def get_original(self) -> Optional[Image.Image]:
        """获取原始图像"""
        return self._original_image

    def get_edited(self) -> Optional[Image.Image]:
        """获取编辑后的图像（实时计算）"""
        if self._original_image is None:
            return None

        # 如果缓存有效，直接返回
        if self._cache_valid and self._cached_edited is not None:
            return self._cached_edited

        # 应用所有编辑效果
        self._cached_edited = self._apply_edits()
        self._cache_valid = True
        return self._cached_edited

    def reset(self):
        """重置所有参数"""
        self._params = EditParams()
        self._cache_valid = False

    def has_image(self) -> bool:
        """检查是否已加载图像"""
        return self._original_image is not None

    def set_param(self, param_name: str, value: float):
        """设置单个参数值

        Args:
            param_name: 参数名称
            value: 参数值
        """
        if hasattr(self._params, param_name):
            setattr(self._params, param_name, value)
            self._cache_valid = False

    def get_param(self, param_name: str) -> float:
        """获取单个参数值"""
        return getattr(self._params, param_name, 0.0)

    def get_all_params(self) -> EditParams:
        """获取所有参数"""
        return self._params

    def _apply_edits_fullres(self) -> Image.Image:
        """应用所有编辑效果到原始图像（全分辨率，用于保存）"""
        if self._original_image is None:
            return None

        # 转换为numpy数组进行处理
        img_array = np.array(self._original_image).astype(np.float32)
        is_rgba = self._original_image.mode == 'RGBA'

        # 1. 曝光调整 (UI: -100 到 +100, 内部映射到 -5~5)
        if self._params.exposure != 0:
            actual_exposure = self._params.exposure / 20.0
            exposure_factor = np.power(2, actual_exposure)
            img_array[:, :, :3] *= exposure_factor

        # 2. 对比度调整 (-100 到 +100)
        if self._params.contrast != 0:
            contrast_factor = 1 + self._params.contrast / 100.0
            mean = 128
            img_array[:, :, :3] = mean + (img_array[:, :, :3] - mean) * contrast_factor

        # 3. 高光调整 (-100 到 +100)
        if self._params.highlights != 0:
            img_array = self._adjust_highlights(img_array, self._params.highlights)

        # 4. 阴影调整 (-100 到 +100)
        if self._params.shadows != 0:
            img_array = self._adjust_shadows(img_array, self._params.shadows)

        # 5. 白色色阶调整 (-100 到 +100)
        if self._params.whites != 0:
            img_array = self._adjust_whites(img_array, self._params.whites)

        # 6. 黑色色阶调整 (-100 到 +100)
        if self._params.blacks != 0:
            img_array = self._adjust_blacks(img_array, self._params.blacks)

        # 7. 纹理调整 (-100 到 +100)
        if self._params.texture != 0:
            img_array = self._adjust_texture(img_array, self._params.texture, is_rgba)

        # 8. 清晰度调整 (-100 到 +100)
        if self._params.clarity != 0:
            img_array = self._adjust_clarity(img_array, self._params.clarity, is_rgba)

        # 9. 去朦胧调整 (-100 到 +100)
        if self._params.dehaze != 0:
            img_array = self._adjust_dehaze(img_array, self._params.dehaze)

        # 10. 鲜艳度调整 (-100 到 +100)
        if self._params.vibrance != 0:
            img_array = self._adjust_vibrance(img_array, self._params.vibrance)

        # 11. 饱和度调整 (-100 到 +100)
        if self._params.saturation != 0:
            sat_factor = 1 + self._params.saturation / 100.0
            img_array = self._adjust_saturation(img_array, sat_factor)

        # 裁剪值到有效范围
        img_array[:, :, :3] = np.clip(img_array[:, :, :3], 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def _apply_edits(self) -> Image.Image:
        """应用所有编辑效果到原始图像（使用预览图加速）"""
        if self._original_image is None:
            return None

        # 使用预览图像进行处理以加速
        source_image = self._preview_image if self._preview_image else self._original_image

        # 转换为numpy数组进行处理
        img_array = np.array(source_image).astype(np.float32)
        is_rgba = source_image.mode == 'RGBA'

        # 1. 曝光调整 (UI: -100 到 +100, 内部映射到 -5~5)
        if self._params.exposure != 0:
            # 将 -100~100 映射到 -5~5
            actual_exposure = self._params.exposure / 20.0
            exposure_factor = np.power(2, actual_exposure)
            img_array[:, :, :3] *= exposure_factor

        # 2. 对比度调整 (-100 到 +100)
        if self._params.contrast != 0:
            contrast_factor = 1 + self._params.contrast / 100.0
            mean = 128
            img_array[:, :, :3] = mean + (img_array[:, :, :3] - mean) * contrast_factor

        # 3. 高光调整 (-100 到 +100)
        if self._params.highlights != 0:
            img_array = self._adjust_highlights(img_array, self._params.highlights)

        # 4. 阴影调整 (-100 到 +100)
        if self._params.shadows != 0:
            img_array = self._adjust_shadows(img_array, self._params.shadows)

        # 5. 白色色阶调整 (-100 到 +100)
        if self._params.whites != 0:
            img_array = self._adjust_whites(img_array, self._params.whites)

        # 6. 黑色色阶调整 (-100 到 +100)
        if self._params.blacks != 0:
            img_array = self._adjust_blacks(img_array, self._params.blacks)

        # 7. 纹理调整 (-100 到 +100)
        if self._params.texture != 0:
            img_array = self._adjust_texture(img_array, self._params.texture, is_rgba)

        # 8. 清晰度调整 (-100 到 +100)
        if self._params.clarity != 0:
            img_array = self._adjust_clarity(img_array, self._params.clarity, is_rgba)

        # 9. 去朦胧调整 (-100 到 +100)
        if self._params.dehaze != 0:
            img_array = self._adjust_dehaze(img_array, self._params.dehaze)

        # 10. 鲜艳度调整 (-100 到 +100)
        if self._params.vibrance != 0:
            img_array = self._adjust_vibrance(img_array, self._params.vibrance)

        # 11. 饱和度调整 (-100 到 +100)
        if self._params.saturation != 0:
            sat_factor = 1 + self._params.saturation / 100.0
            img_array = self._adjust_saturation(img_array, sat_factor)

        # 裁剪值到有效范围
        img_array[:, :, :3] = np.clip(img_array[:, :, :3], 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def _adjust_highlights(self, img: np.ndarray, value: float) -> np.ndarray:
        """调整高光"""
        # 找出高光区域（亮度 > 128）
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        highlight_mask = np.clip((luminance - 128) / 127, 0, 1)

        # 创建调整因子
        factor = value / 100.0
        adjustment = 1 - factor * highlight_mask

        img[:, :, :3] = img[:, :, :3] * adjustment[:, :, np.newaxis]
        return img

    def _adjust_shadows(self, img: np.ndarray, value: float) -> np.ndarray:
        """调整阴影"""
        # 找出阴影区域（亮度 < 128）
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        shadow_mask = np.clip((128 - luminance) / 128, 0, 1)

        # 创建调整因子
        factor = value / 100.0
        adjustment = 1 + factor * shadow_mask

        img[:, :, :3] = img[:, :, :3] * adjustment[:, :, np.newaxis]
        return img

    def _adjust_whites(self, img: np.ndarray, value: float) -> np.ndarray:
        """调整白色色阶（白场）"""
        factor = value / 100.0
        # 调整最亮区域
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        bright_mask = np.clip((luminance - 200) / 55, 0, 1)

        if factor > 0:
            # 增加白色：向255推进
            adjustment = 255 - (255 - img[:, :, :3]) * (1 - factor * bright_mask[:, :, np.newaxis])
        else:
            # 减少白色：压暗高光
            adjustment = img[:, :, :3] * (1 + factor * bright_mask[:, :, np.newaxis])

        img[:, :, :3] = adjustment
        return img

    def _adjust_blacks(self, img: np.ndarray, value: float) -> np.ndarray:
        """调整黑色色阶（黑场）"""
        factor = value / 100.0
        # 调整最暗区域
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        dark_mask = np.clip((50 - luminance) / 50, 0, 1)

        if factor > 0:
            # 提升黑色：向更亮推进
            adjustment = img[:, :, :3] + factor * dark_mask[:, :, np.newaxis] * 50
        else:
            # 加深黑色：向0推进
            adjustment = img[:, :, :3] * (1 + factor * dark_mask[:, :, np.newaxis])

        img[:, :, :3] = adjustment
        return img

    def _adjust_texture(self, img: np.ndarray, value: float, is_rgba: bool) -> np.ndarray:
        """调整纹理（中等尺度细节）"""
        factor = abs(value) / 100.0

        if factor == 0:
            return img

        # 转换为PIL图像进行滤波
        pil_img = Image.fromarray(img.astype(np.uint8))

        # 使用UnsharpMask进行纹理增强
        if value > 0:
            # 增强纹理
            blurred = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=0))
            result = Image.blend(pil_img, blurred, factor * 0.5)
        else:
            # 平滑纹理
            blurred = pil_img.filter(ImageFilter.MedianFilter(size=3))
            result = Image.blend(pil_img, blurred, factor * 0.5)

        return np.array(result).astype(np.float32)

    def _adjust_clarity(self, img: np.ndarray, value: float, is_rgba: bool) -> np.ndarray:
        """调整清晰度（局部对比度）"""
        factor = value / 100.0

        if factor == 0:
            return img

        # 转换为PIL图像
        pil_img = Image.fromarray(img.astype(np.uint8))

        # 使用更强的UnsharpMask进行清晰度增强
        if value > 0:
            blurred = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=0))
            result = Image.blend(pil_img, blurred, abs(factor) * 0.7)
        else:
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
            result = Image.blend(pil_img, blurred, abs(factor) * 0.5)

        return np.array(result).astype(np.float32)

    def _adjust_dehaze(self, img: np.ndarray, value: float) -> np.ndarray:
        """去朦胧调整"""
        factor = value / 100.0

        if factor == 0:
            return img

        # 计算最小通道值（用于估计雾霾）
        min_channel = np.min(img[:, :, :3], axis=2)

        if factor > 0:
            # 去朦胧：增加对比度和饱和度
            # 基于暗通道先验的简化实现
            transmission = 1 - factor * (min_channel / 255)
            transmission = np.clip(transmission, 0.1, 1)

            for i in range(3):
                img[:, :, i] = (img[:, :, i] - min_channel * factor * 0.5) / transmission
        else:
            # 添加朦胧：降低对比度
            haze_amount = abs(factor) * 50
            img[:, :, :3] = img[:, :, :3] * (1 - abs(factor) * 0.3) + haze_amount

        return img

    def _adjust_vibrance(self, img: np.ndarray, value: float) -> np.ndarray:
        """鲜艳度调整（智能饱和度）"""
        factor = value / 100.0

        if factor == 0:
            return img

        # 计算每个像素的饱和度
        max_val = np.max(img[:, :, :3], axis=2)
        min_val = np.min(img[:, :, :3], axis=2)
        current_sat = (max_val - min_val) / (max_val + 1e-6)

        # 饱和度较低的像素获得更强的调整
        sat_weight = 1 - current_sat

        # 应用调整
        if factor > 0:
            adjustment = factor * sat_weight * 0.5
        else:
            adjustment = factor * current_sat * 0.5

        # 调整饱和度
        mean = (max_val + min_val) / 2
        for i in range(3):
            img[:, :, i] = mean + (img[:, :, i] - mean) * (1 + adjustment)

        return img

    def _adjust_saturation(self, img: np.ndarray, factor: float) -> np.ndarray:
        """饱和度调整"""
        if factor == 1:
            return img

        # 计算亮度
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # 根据亮度调整饱和度
        for i in range(3):
            img[:, :, i] = luminance + (img[:, :, i] - luminance) * factor

        return img
