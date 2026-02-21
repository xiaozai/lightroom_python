"""图像编辑核心功能 - 非破坏性编辑"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import colorsys


@dataclass
class EditParams:
    """编辑参数数据类"""
    # 基本调整参数
    exposure: float = 0.0       # 曝光: -100 到 +100
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

    # 白平衡参数
    temp: float = 0.0           # 色温: -100 到 +100 (蓝-黄)
    tint: float = 0.0           # 色调: -150 到 +150 (绿-洋红)

    # HSL参数 - 8种颜色 (红、橙、黄、绿、青、蓝、紫、洋红)
    # 色相调整
    hsl_hue_red: float = 0.0
    hsl_hue_orange: float = 0.0
    hsl_hue_yellow: float = 0.0
    hsl_hue_green: float = 0.0
    hsl_hue_cyan: float = 0.0
    hsl_hue_blue: float = 0.0
    hsl_hue_purple: float = 0.0
    hsl_hue_magenta: float = 0.0

    # 饱和度调整
    hsl_sat_red: float = 0.0
    hsl_sat_orange: float = 0.0
    hsl_sat_yellow: float = 0.0
    hsl_sat_green: float = 0.0
    hsl_sat_cyan: float = 0.0
    hsl_sat_blue: float = 0.0
    hsl_sat_purple: float = 0.0
    hsl_sat_magenta: float = 0.0

    # 明度调整
    hsl_lum_red: float = 0.0
    hsl_lum_orange: float = 0.0
    hsl_lum_yellow: float = 0.0
    hsl_lum_green: float = 0.0
    hsl_lum_cyan: float = 0.0
    hsl_lum_blue: float = 0.0
    hsl_lum_purple: float = 0.0
    hsl_lum_magenta: float = 0.0

    # 颜色分级参数
    # 阴影
    cg_shadows_hue: float = 0.0     # 0-360
    cg_shadows_sat: float = 0.0     # 0-100
    # 中间调
    cg_midtones_hue: float = 0.0
    cg_midtones_sat: float = 0.0
    # 高光
    cg_highlights_hue: float = 0.0
    cg_highlights_sat: float = 0.0
    # 混合与平衡
    cg_blending: float = 50.0       # 0-100
    cg_balance: float = 0.0         # -100 到 +100

    # 细节面板参数 - 锐化
    sharpen_amount: float = 0.0     # 0-150
    sharpen_radius: float = 1.0     # 0.5-3.0
    sharpen_detail: float = 25.0    # 0-100
    sharpen_masking: float = 0.0    # 0-100

    # 细节面板参数 - 降噪
    noise_luminance: float = 0.0    # 0-100
    noise_color: float = 0.0        # 0-100

    # 色调曲线参数
    # RGB复合曲线 (控制点列表，每个点是(x,y)元组，x和y范围0-255)
    curve_rgb: list = field(default_factory=lambda: [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)])
    # 分通道曲线
    curve_red: list = field(default_factory=lambda: [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)])
    curve_green: list = field(default_factory=lambda: [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)])
    curve_blue: list = field(default_factory=lambda: [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)])
    # 曲线饱和度调整
    curve_saturation: float = 0.0  # -100 到 +100


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

    def set_curve_param(self, channel: str, points: list):
        """设置曲线参数

        Args:
            channel: 通道名称 ('rgb', 'red', 'green', 'blue')
            points: 控制点列表 [(x1, y1), (x2, y2), ...]
        """
        param_name = f'curve_{channel}'
        if hasattr(self._params, param_name):
            setattr(self._params, param_name, points)
            self._cache_valid = False

    def get_curve_param(self, channel: str) -> list:
        """获取曲线参数"""
        param_name = f'curve_{channel}'
        return getattr(self._params, param_name, [])

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

        # 12. 白平衡调整
        img_array = self._adjust_white_balance(img_array)

        # 13. HSL调整
        img_array = self._adjust_hsl(img_array)

        # 14. 颜色分级
        img_array = self._adjust_color_grading(img_array)

        # 15. 锐化调整
        img_array = self._adjust_sharpening(img_array, is_rgba)

        # 16. 降噪调整
        img_array = self._adjust_noise_reduction(img_array)

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

        # 12. 白平衡调整
        img_array = self._adjust_white_balance(img_array)

        # 13. HSL调整
        img_array = self._adjust_hsl(img_array)

        # 14. 颜色分级
        img_array = self._adjust_color_grading(img_array)

        # 15. 锐化调整
        img_array = self._adjust_sharpening(img_array, is_rgba)

        # 16. 降噪调整
        img_array = self._adjust_noise_reduction(img_array)

        # 17. 色调曲线调整
        img_array = self._adjust_tone_curve(img_array)

        # 裁剪值到有效范围
        img_array[:, :, :3] = np.clip(img_array[:, :, :3], 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))

    def _adjust_highlights(self, img: np.ndarray, value: float) -> np.ndarray:
        """调整高光 - 参考 monetGPT 的平滑遮罩实现"""
        if value == 0:
            return img

        # 计算亮度
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        L = luminance / 255.0  # 归一化到 [0, 1]

        # 使用 smoothstep 创建平滑的高光遮罩 (0.35 到 1.0)
        # 平滑过渡，避免硬边缘
        def smoothstep(x, edge0, edge1):
            t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        # 高光遮罩：影响亮度在 0.35 到 1.0 范围的区域
        highlight_mask = smoothstep(L, 0.35, 1.0)

        factor = value / 100.0 * 0.3  # 缩放因子

        # 计算调整后的亮度
        L_adjusted = L + factor * highlight_mask * 0.5

        # 避免除零
        eps = 1e-8
        ratio = L_adjusted / (L + eps)

        # 应用到RGB通道
        img[:, :, 0] = img[:, :, 0] * ratio
        img[:, :, 1] = img[:, :, 1] * ratio
        img[:, :, 2] = img[:, :, 2] * ratio

        return img

    def _adjust_shadows(self, img: np.ndarray, value: float) -> np.ndarray:
        """调整阴影 - 参考 monetGPT 的平滑遮罩实现"""
        if value == 0:
            return img

        # 计算亮度
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        L = luminance / 255.0  # 归一化到 [0, 1]

        # 使用 smoothstep 创建平滑的阴影遮罩
        def smoothstep(x, edge0, edge1):
            t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        # 创建遮罩：0 在暗部，1 在亮部
        # 反转以获得阴影遮罩
        shadow_mask = 1.0 - smoothstep(L, 0.0, 0.65)

        factor = value / 100.0 * 0.1  # 缩放因子

        # 计算调整后的亮度
        L_adjusted = L + factor * shadow_mask * 0.5

        # 避免除零
        eps = 1e-8
        ratio = L_adjusted / (L + eps)

        # 应用到RGB通道
        img[:, :, 0] = img[:, :, 0] * ratio
        img[:, :, 1] = img[:, :, 1] * ratio
        img[:, :, 2] = img[:, :, 2] * ratio

        return img

    def _adjust_whites(self, img: np.ndarray, value: float) -> np.ndarray:
        """调整白色色阶（白场）- 参考 monetGPT 实现"""
        if value == 0:
            return img

        factor = value / 100.0 * 0.3

        # 计算亮度
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        L = luminance / 255.0

        # 使用平滑遮罩 (0.65 到 1.0)
        def smoothstep(x, edge0, edge1):
            t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        whites_mask = smoothstep(L, 0.65, 1.0)

        # 调整亮度
        L_adjusted = L + factor * whites_mask * 0.5

        eps = 1e-8
        ratio = L_adjusted / (L + eps)

        img[:, :, 0] = img[:, :, 0] * ratio
        img[:, :, 1] = img[:, :, 1] * ratio
        img[:, :, 2] = img[:, :, 2] * ratio

        return img

    def _adjust_blacks(self, img: np.ndarray, value: float) -> np.ndarray:
        """调整黑色色阶（黑场）- 参考 monetGPT 实现"""
        if value == 0:
            return img

        factor = value / 100.0

        # 计算亮度
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        L = luminance / 255.0

        # 使用平滑遮罩 (0.0 到 0.3)
        def smoothstep(x, edge0, edge1):
            t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        # 黑色遮罩：亮度越低，影响越大
        blacks_mask = 1.0 - smoothstep(L, 0.0, 0.3)

        # 调整
        if factor > 0:
            # 提升黑色
            L_adjusted = L + factor * blacks_mask * 0.15
        else:
            # 加深黑色
            L_adjusted = L + factor * blacks_mask * 0.1

        eps = 1e-8
        ratio = L_adjusted / (L + eps)

        img[:, :, 0] = img[:, :, 0] * ratio
        img[:, :, 1] = img[:, :, 1] * ratio
        img[:, :, 2] = img[:, :, 2] * ratio

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
        """鲜艳度调整（智能饱和度）- 参考 monetGPT 实现"""
        if value == 0:
            return img

        factor = value / 100.0

        # 转换为HSV空间
        rgb_img = Image.fromarray(img.astype(np.uint8))
        hsv_img = rgb_img.convert('HSV')
        hsv_array = np.array(hsv_img).astype(np.float32)

        # 分离 H, S, V 通道
        H = hsv_array[:, :, 0].astype(np.float32)
        S = hsv_array[:, :, 1].astype(np.float32)
        V = hsv_array[:, :, 2].astype(np.float32)

        # 归一化到 [0, 1]
        S_norm = S / 255.0
        V_norm = V / 255.0
        H_norm = H / 179.0  # HSV H 通道是 0-179

        # 计算高斯权重（饱和度和明度中心化）
        mean = 0.45
        sigma_s = 0.2
        sigma_v = 0.2

        w_s = np.exp(-((S_norm - mean) ** 2) / (2 * sigma_s ** 2))
        w_v = np.exp(-((V_norm - mean) ** 2) / (2 * sigma_v ** 2))
        weight = w_s * w_v

        # 肤色保护：降低肤色区域的鲜艳度调整
        # 典型肤色 hue ~10-50 (在 0-179 范围内约为 0.06-0.28)
        mean_h_skin = 0.08
        sigma_h = 0.05
        w_h = 1 - np.exp(-((H_norm - mean_h_skin) ** 2) / (2 * sigma_h ** 2))

        # 组合所有权重
        weight_total = weight * w_h

        # 应用鲜艳度调整
        if factor >= 0:
            # 增加鲜艳度：低饱和度区域获得更多调整
            S_new = S_norm + factor * (1 - S_norm) * weight_total
        else:
            # 减少鲜艳度：按比例减少
            S_new = S_norm * (1 + factor * weight_total)

        S_new = np.clip(S_new, 0, 1) * 255.0

        # 合并通道
        hsv_array[:, :, 1] = S_new.astype(np.uint8)

        # 转换回RGB
        result_img = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV').convert('RGB')

        return np.array(result_img).astype(np.float32)

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

    def _adjust_white_balance(self, img: np.ndarray) -> np.ndarray:
        """调整白平衡（色温和色调）"""
        temp = self._params.temp
        tint = self._params.tint

        if temp == 0 and tint == 0:
            return img

        # 色调（Tint）调整 - 更精确的绿色/洋红平衡
        if tint != 0:
            intensity = tint / 150.0

            # 放大效果
            intensity = intensity * 1.4

            # 使用RGB调整模拟Lab的调整（更精确的色调控制）
            if intensity >= 0:
                # 偏洋红：增强红和蓝通道，减弱绿通道
                r_factor = 1.0 + intensity * 0.02
                g_factor = 1.0 - intensity * 0.03
                b_factor = 1.0 + intensity * 0.01
            else:
                # 偏绿（荧光绿）：增强绿通道
                abs_i = abs(intensity)
                r_factor = 1.0 - abs_i * 0.02
                g_factor = 1.0 + abs_i * 0.05
                b_factor = 1.0 - abs_i * 0.03

            img[:, :, 0] *= r_factor
            img[:, :, 1] *= g_factor
            img[:, :, 2] *= b_factor

            # 可选的颜色平衡微调
            if tint < 0:
                # 绿色调时稍提亮
                img[:, :3] += abs(tint / 150) * 3
            elif tint > 0:
                # 洋红色调的明亮度微调
                img[:, :3] -= tint / 150 * 2

        # 色温（Temperature）调整
        if temp != 0:
            factor = temp / 100.0
            # 蓝色通道（冷）vs 红色通道（暖）
            if factor > 0:
                # 变暖：增加红色，减少蓝色
                img[:, :, 0] = img[:, :, 0] * (1 + factor * 0.08)
                img[:, :, 2] = img[:, :, 2] * (1 - factor * 0.06)
            else:
                # 变冷：减少红色，增加蓝色
                abs_factor = abs(factor)
                img[:, :, 0] = img[:, :, 0] * (1 - abs_factor * 0.06)
                img[:, :, 2] = img[:, :, 2] * (1 + abs_factor * 0.08)

        return img

    def _adjust_hsl(self, img: np.ndarray) -> np.ndarray:
        """HSL调整 - 对8种颜色分别调整色相、饱和度、明度（向量化优化版本）"""
        # HSL调整参数
        hsl_params = {
            'red': (self._params.hsl_hue_red, self._params.hsl_sat_red, self._params.hsl_lum_red),
            'orange': (self._params.hsl_hue_orange, self._params.hsl_sat_orange, self._params.hsl_lum_orange),
            'yellow': (self._params.hsl_hue_yellow, self._params.hsl_sat_yellow, self._params.hsl_lum_yellow),
            'green': (self._params.hsl_hue_green, self._params.hsl_sat_green, self._params.hsl_lum_green),
            'cyan': (self._params.hsl_hue_cyan, self._params.hsl_sat_cyan, self._params.hsl_lum_cyan),
            'blue': (self._params.hsl_hue_blue, self._params.hsl_sat_blue, self._params.hsl_lum_blue),
            'purple': (self._params.hsl_hue_purple, self._params.hsl_sat_purple, self._params.hsl_lum_purple),
            'magenta': (self._params.hsl_hue_magenta, self._params.hsl_sat_magenta, self._params.hsl_lum_magenta)
        }

        # 检查是否有任何HSL参数被调整
        has_hsl_adjustment = any(
            hue != 0 or sat != 0 or lum != 0
            for hue, sat, lum in hsl_params.values()
        )

        if not has_hsl_adjustment:
            return img

        # 向量化RGB到HSV转换
        rgb = img[:, :, :3] / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val

        # 计算色相
        h = np.zeros_like(max_val)
        # 红色最大
        mask = (max_val == r) & (diff > 0)
        h[mask] = 60 * (((g[mask] - b[mask]) / diff[mask]) % 6)
        # 绿色最大
        mask = (max_val == g) & (diff > 0)
        h[mask] = 60 * (((b[mask] - r[mask]) / diff[mask]) + 2)
        # 蓝色最大
        mask = (max_val == b) & (diff > 0)
        h[mask] = 60 * (((r[mask] - g[mask]) / diff[mask]) + 4)

        # 饱和度
        s = np.where(max_val == 0, 0, diff / max_val)
        # 明度
        v = max_val

        # 定义颜色范围并应用调整
        color_ranges = [
            ('red', [(0, 15), (345, 360)]),
            ('orange', [(15, 45)]),
            ('yellow', [(45, 75)]),
            ('green', [(75, 150)]),
            ('cyan', [(150, 195)]),
            ('blue', [(195, 255)]),
            ('purple', [(255, 285)]),
            ('magenta', [(285, 345)])
        ]

        for color_name, ranges in color_ranges:
            hue_adj, sat_adj, lum_adj = hsl_params[color_name]
            if hue_adj == 0 and sat_adj == 0 and lum_adj == 0:
                continue

            # 创建颜色蒙版
            mask = np.zeros_like(h, dtype=bool)
            for start, end in ranges:
                if start < end:
                    mask |= (h >= start) & (h < end)
                else:
                    mask |= (h >= start) | (h < end)

            if not np.any(mask):
                continue

            # 色相调整
            if hue_adj != 0:
                h[mask] = (h[mask] + hue_adj * 0.3) % 360

            # 饱和度调整
            if sat_adj != 0:
                s[mask] = np.clip(s[mask] * (1 + sat_adj / 100.0), 0, 1)

            # 明度调整
            if lum_adj != 0:
                v[mask] = np.clip(v[mask] * (1 + lum_adj / 200.0), 0, 1)

        # 向量化HSV到RGB转换
        h_norm = h / 60.0
        c = v * s
        x = c * (1 - np.abs(h_norm % 2 - 1))
        m = v - c

        r_new = np.zeros_like(h)
        g_new = np.zeros_like(h)
        b_new = np.zeros_like(h)

        # 0-60度
        mask = (h_norm >= 0) & (h_norm < 1)
        r_new[mask], g_new[mask], b_new[mask] = c[mask], x[mask], 0
        # 60-120度
        mask = (h_norm >= 1) & (h_norm < 2)
        r_new[mask], g_new[mask], b_new[mask] = x[mask], c[mask], 0
        # 120-180度
        mask = (h_norm >= 2) & (h_norm < 3)
        r_new[mask], g_new[mask], b_new[mask] = 0, c[mask], x[mask]
        # 180-240度
        mask = (h_norm >= 3) & (h_norm < 4)
        r_new[mask], g_new[mask], b_new[mask] = 0, x[mask], c[mask]
        # 240-300度
        mask = (h_norm >= 4) & (h_norm < 5)
        r_new[mask], g_new[mask], b_new[mask] = x[mask], 0, c[mask]
        # 300-360度
        mask = (h_norm >= 5) & (h_norm < 6)
        r_new[mask], g_new[mask], b_new[mask] = c[mask], 0, x[mask]

        img[:, :, 0] = (r_new + m) * 255
        img[:, :, 1] = (g_new + m) * 255
        img[:, :, 2] = (b_new + m) * 255

        return img

    def _adjust_color_grading(self, img: np.ndarray) -> np.ndarray:
        """颜色分级 - 为阴影、中间调、高光添加颜色"""
        # 获取参数
        shadows_hue = self._params.cg_shadows_hue
        shadows_sat = self._params.cg_shadows_sat
        midtones_hue = self._params.cg_midtones_hue
        midtones_sat = self._params.cg_midtones_sat
        highlights_hue = self._params.cg_highlights_hue
        highlights_sat = self._params.cg_highlights_sat
        blending = self._params.cg_blending / 100.0
        balance = self._params.cg_balance / 100.0

        # 检查是否有颜色分级调整
        if (shadows_sat == 0 and midtones_sat == 0 and highlights_sat == 0):
            return img

        # 计算亮度
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        luminance_norm = luminance / 255.0

        # 平衡参数影响中间点的位置
        # balance = 0: 中间点在 0.5
        # balance = 100: 中间点在 0.0 (更多区域被视为高光)
        # balance = -100: 中间点在 1.0 (更多区域被视为阴影)
        mid_point = 0.5 - balance * 0.4  # 范围: 0.1 ~ 0.9

        # 创建区域蒙版 - 使用S曲线实现平滑过渡
        # 混合参数控制过渡的锐度
        transition_width = 0.3 + blending * 0.4  # 范围: 0.3 ~ 0.7

        # 阴影区域：亮度低于中间点的部分
        shadows_mask = 1.0 / (1.0 + np.exp((luminance_norm - mid_point) / transition_width * 4))
        # 高光区域：亮度高于中间点的部分
        highlights_mask = 1.0 / (1.0 + np.exp((mid_point - luminance_norm) / transition_width * 4))
        # 中间调：剩余部分
        midtones_mask = 1.0 - shadows_mask - highlights_mask
        midtones_mask = np.clip(midtones_mask, 0, 1)

        # HSV转RGB函数
        def hsv_to_rgb_array(hue, sat, val=1.0):
            h = hue / 360.0
            c = val * sat
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = val - c

            if h < 1/6:
                return np.array([c + m, x + m, m])
            elif h < 2/6:
                return np.array([x + m, c + m, m])
            elif h < 3/6:
                return np.array([m, c + m, x + m])
            elif h < 4/6:
                return np.array([m, x + m, c + m])
            elif h < 5/6:
                return np.array([x + m, m, c + m])
            else:
                return np.array([c + m, m, x + m])

        # 应用阴影颜色 - 增强效果强度
        if shadows_sat > 0:
            shadow_color = hsv_to_rgb_array(shadows_hue, shadows_sat / 100.0) * 255
            # 增强混合系数 - 使用更高的混合强度
            mix_strength = shadows_sat / 100.0 * 0.9  # 最大混合90%
            for i in range(3):
                img[:, :, i] = img[:, :, i] * (1 - shadows_mask * mix_strength) + \
                               shadow_color[i] * shadows_mask * mix_strength

        # 应用中间调颜色
        if midtones_sat > 0:
            midtone_color = hsv_to_rgb_array(midtones_hue, midtones_sat / 100.0) * 255
            mix_strength = midtones_sat / 100.0 * 0.7
            for i in range(3):
                img[:, :, i] = img[:, :, i] * (1 - midtones_mask * mix_strength) + \
                               midtone_color[i] * midtones_mask * mix_strength

        # 应用高光颜色 - 增强效果强度
        if highlights_sat > 0:
            highlight_color = hsv_to_rgb_array(highlights_hue, highlights_sat / 100.0) * 255
            mix_strength = highlights_sat / 100.0 * 0.9
            for i in range(3):
                img[:, :, i] = img[:, :, i] * (1 - highlights_mask * mix_strength) + \
                               highlight_color[i] * highlights_mask * mix_strength

        return img

    def _adjust_sharpening(self, img: np.ndarray, is_rgba: bool) -> np.ndarray:
        """锐化调整"""
        amount = self._params.sharpen_amount
        radius = self._params.sharpen_radius
        detail = self._params.sharpen_detail
        masking = self._params.sharpen_masking

        if amount == 0:
            return img

        # 转换为PIL图像
        pil_img = Image.fromarray(img.astype(np.uint8))

        # 使用UnsharpMask进行锐化
        # radius 必须是整数，至少为1
        radius_int = max(1, int(round(radius)))

        # percent 控制锐化强度，值越大锐化越强
        # 将 0-150 映射到 50-300 的范围
        percent_int = max(50, int(50 + amount * 1.67))

        # threshold 控制锐化的阈值，值越小锐化越明显
        # detail 参数：值越高锐化更多细节（threshold 越低）
        # 将 detail 0-100 映射到 threshold 20-0
        threshold_int = max(0, int(20 * (1 - detail / 100.0)))

        sharpened = pil_img.filter(
            ImageFilter.UnsharpMask(radius=radius_int, percent=percent_int, threshold=threshold_int)
        )

        # 应用蒙版：只对边缘进行锐化
        if masking > 0:
            # 计算边缘蒙版 - 使用简单的拉普拉斯卷积核
            gray = np.array(pil_img.convert('L')).astype(np.float32)

            # 手动实现拉普拉斯算子 (3x3 卷积核)
            laplacian_kernel = np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ])

            # 使用卷积计算边缘
            from numpy.lib.stride_tricks import as_strided
            h, w = gray.shape
            kh, kw = laplacian_kernel.shape

            # Pad the image
            padded = np.pad(gray, ((1, 1), (1, 1)), mode='edge')

            # Create view for convolution
            shape = (h, w, kh, kw)
            strides = padded.strides * 2
            windows = as_strided(padded, shape=shape, strides=strides)

            # Apply convolution
            edges = np.abs(np.sum(windows * laplacian_kernel, axis=(2, 3)))
            edge_mask = np.clip(edges / (edges.max() + 1e-6), 0, 1)

            # 根据masking参数调整蒙版
            edge_mask = np.power(edge_mask, 2 - masking / 50.0)
            edge_mask = np.stack([edge_mask] * 3, axis=2)

            result_array = np.array(pil_img) * (1 - edge_mask * amount / 150.0) + \
                          np.array(sharpened) * edge_mask * amount / 150.0
            return result_array.astype(np.float32)
        else:
            # 直接混合 - 确保有可见的效果
            factor = min(1.0, amount / 100.0)  # 将混合因子映射到0-1范围
            result = Image.blend(pil_img, sharpened, factor)
            return np.array(result).astype(np.float32)

    def _adjust_noise_reduction(self, img: np.ndarray) -> np.ndarray:
        """降噪调整"""
        lum_noise = self._params.noise_luminance
        color_noise = self._params.noise_color

        if lum_noise == 0 and color_noise == 0:
            return img

        # 明度降噪：使用高斯模糊
        if lum_noise > 0:
            pil_img = Image.fromarray(img.astype(np.uint8))
            # 降噪强度映射到模糊半径
            radius = lum_noise / 25.0
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            factor = lum_noise / 100.0
            result = Image.blend(pil_img, blurred, factor)
            img = np.array(result).astype(np.float32)

        # 颜色降噪：在颜色通道上应用更强的模糊
        if color_noise > 0:
            # 转换到Lab颜色空间进行颜色降噪
            # 简化处理：直接对颜色差异进行平滑
            mean_color = np.mean(img[:, :, :3], axis=2, keepdims=True)
            color_diff = img[:, :, :3] - mean_color
            # 减少颜色差异
            factor = 1 - color_noise / 100.0 * 0.5
            img[:, :, :3] = mean_color + color_diff * factor

        return img

    def _build_curve_lut(self, points: list, size: int = 256) -> np.ndarray:
        """构建曲线查找表（LUT）使用三次样条插值

        Args:
            points: 控制点列表 [(x1, y1), (x2, y2), ...]，x和y都在0-255范围内
            size: LUT大小，默认256

        Returns:
            查找表数组，长度为size
        """
        # 确保控制点按x排序
        points = sorted(points, key=lambda p: p[0])

        # 提取x和y坐标
        x_vals = np.array([p[0] for p in points], dtype=np.float32)
        y_vals = np.array([p[1] for p in points], dtype=np.float32)

        # 创建输入数组（0-255）
        lut_input = np.arange(size, dtype=np.float32)

        # 使用三次样条插值
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x_vals, y_vals, bc_type='natural', extrapolate=True)

        # 计算LUT
        lut = cs(lut_input)

        # 裁剪到有效范围
        lut = np.clip(lut, 0, 255)

        return lut.astype(np.float32)

    def _adjust_tone_curve(self, img: np.ndarray) -> np.ndarray:
        """色调曲线调整"""
        # 检查是否有曲线调整
        has_curve_edits = (
            self._params.curve_saturation != 0 or
            any(p != q for p, q in self._params.curve_rgb for p, q in [(0, 0), (255, 255)])
        )

        # 检查RGB曲线是否不是默认直线
        if len(self._params.curve_rgb) >= 2:
            # 检查是否有任何控制点偏移
            for x, y in self._params.curve_rgb:
                if x != y:
                    has_curve_edits = True
                    break

        # 检查分通道曲线
        def _is_default_curve(points):
            # 默认直线: (0,0), (64,64), (128,128), (192,192), (255,255)
            default = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
            for point, default_point in zip(sorted(points, key=lambda p: p[0]), default):
                if point[0] != default_point[0] or point[1] != default_point[1]:
                    return False
            return True

        if not has_curve_edits:
            if not _is_default_curve(self._params.curve_red):
                has_curve_edits = True
            elif not _is_default_curve(self._params.curve_green):
                has_curve_edits = True
            elif not _is_default_curve(self._params.curve_blue):
                has_curve_edits = True

        if not has_curve_edits:
            return img

        # 构建RGB复合曲线LUT
        rgb_lut = self._build_curve_lut(self._params.curve_rgb)

        # 构建分通道曲线LUT
        red_lut = self._build_curve_lut(self._params.curve_red)
        green_lut = self._build_curve_lut(self._params.curve_green)
        blue_lut = self._build_curve_lut(self._params.curve_blue)

        # 应用曲线到图像
        # 首先应用RGB复合曲线
        # 将像素值裁剪到0-255范围，避免索引越界
        img_int = np.clip(img[:, :, :3], 0, 255).astype(np.int32)
        img[:, :, 0] = rgb_lut[img_int[:, :, 0]]
        img[:, :, 1] = rgb_lut[img_int[:, :, 1]]
        img[:, :, 2] = rgb_lut[img_int[:, :, 2]]

        # 然后叠加分通道曲线
        img_int = np.clip(img[:, :, :3], 0, 255).astype(np.int32)
        img[:, :, 0] = red_lut[img_int[:, :, 0]]
        img[:, :, 1] = green_lut[img_int[:, :, 1]]
        img[:, :, 2] = blue_lut[img_int[:, :, 2]]

        # 应用曲线饱和度调整
        if self._params.curve_saturation != 0:
            sat_factor = 1 + self._params.curve_saturation / 100.0
            # 计算亮度
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            # 调整饱和度
            for i in range(3):
                img[:, :, i] = luminance + (img[:, :, i] - luminance) * sat_factor

        return img

    def get_params_dict(self) -> dict:
        """获取所有参数作为字典（用于导出）"""
        from dataclasses import asdict
        return asdict(self._params)

    def set_params_from_dict(self, params: dict):
        """从字典设置参数（用于导入）"""
        for key, value in params.items():
            if hasattr(self._params, key):
                # 处理曲线参数：将列表的列表转换为元组的列表
                if key.startswith('curve_') and key != 'curve_saturation':
                    if isinstance(value, list):
                        # 将 [[x, y], [x, y], ...] 转换为 [(x, y), (x, y), ...]
                        value = [tuple(point) if isinstance(point, list) else point for point in value]
                setattr(self._params, key, value)
        self._cache_valid = False
