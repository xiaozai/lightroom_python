"""主窗口UI"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QPushButton, QFileDialog,
    QScrollArea, QFrame, QGroupBox, QSplitter, QMessageBox,
    QScrollArea, QComboBox, QTabWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize, pyqtSignal

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.editor import ImageEditor
from utils.file_utils import (
    get_supported_image_filters,
    get_save_filters,
    get_file_extension
)


class EditSlider(QWidget):
    """编辑滑块控件"""

    # 自定义信号：参数改变时发出
    valueChanged = pyqtSignal(str, float)

    def __init__(self, name: str, param_key: str, min_val: float = -100, max_val: float = 100,
                 default_val: float = 0, decimals: int = 0, parent=None):
        super().__init__(parent)
        self.name = name
        self.param_key = param_key
        self.decimals = decimals
        self._multiplier = 10 ** decimals
        self._setup_ui(min_val, max_val, default_val)

    def _setup_ui(self, min_val: float, max_val: float, default_val: float):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标签
        self.label = QLabel(f"{self.name}:")
        self.label.setFixedWidth(70)
        layout.addWidget(self.label)

        # 滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val * self._multiplier))
        self.slider.setMaximum(int(max_val * self._multiplier))
        self.slider.setValue(int(default_val * self._multiplier))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        # 数值显示
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(default_val)
        self.spinbox.setDecimals(self.decimals)
        self.spinbox.setFixedWidth(70)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        layout.addWidget(self.spinbox)

    def _on_slider_changed(self, value: int):
        real_value = value / self._multiplier
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(real_value)
        self.spinbox.blockSignals(False)
        self._notify_edit(real_value)

    def _on_spinbox_changed(self, value: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self._multiplier))
        self.slider.blockSignals(False)
        self._notify_edit(value)

    def _notify_edit(self, value: float):
        sign = "+" if value >= 0 else ""
        print(f"[编辑] {self.name}: {sign}{value:.1f}")
        # 发出信号，通知主窗口参数改变
        self.valueChanged.emit(self.param_key, value)

    def reset(self):
        """重置滑块值"""
        self.slider.setValue(0)

    def value(self) -> float:
        """获取当前值"""
        return self.spinbox.value()

    def set_value(self, value: float):
        """设置值"""
        self.spinbox.setValue(value)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.editor = ImageEditor()
        self._updating_image = False
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Python Lightroom Tool")
        self.setMinimumSize(1200, 800)

        # 菜单栏
        self._create_menu_bar()

        # 工具栏
        self._create_toolbar()

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧图像区域
        self._create_image_area(splitter)

        # 右侧编辑面板
        self._create_edit_panel(splitter)

        # 设置分割器比例
        splitter.setSizes([900, 300])

        # 状态栏
        self.statusBar().showMessage("就绪 - 请打开一张图片开始编辑")

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        open_action = file_menu.addAction("打开(&O)")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)

        save_action = file_menu.addAction("保存(&S)")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)

        file_menu.addSeparator()

        # 导入预设
        import_action = file_menu.addAction("导入预设(&I)")
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self.import_preset)

        # 导出预设
        export_action = file_menu.addAction("导出预设(&E)")
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_preset)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("退出(&X)")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

    def _create_toolbar(self):
        toolbar = self.addToolBar("工具栏")
        toolbar.setMovable(False)

        open_btn = QPushButton("打开")
        open_btn.clicked.connect(self.open_image)
        toolbar.addWidget(open_btn)

        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_image)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        import_btn = QPushButton("导入预设")
        import_btn.clicked.connect(self.import_preset)
        toolbar.addWidget(import_btn)

        export_btn = QPushButton("导出预设")
        export_btn.clicked.connect(self.export_preset)
        toolbar.addWidget(export_btn)

        toolbar.addSeparator()

        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.reset_edits)
        toolbar.addWidget(reset_btn)

    def _create_image_area(self, parent):
        """创建图像显示区域"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setStyleSheet("background-color: #2a2a2a;")

        self.image_label = QLabel("请打开一张图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #888; font-size: 16px;")
        self.image_label.setMinimumSize(800, 600)

        scroll_area.setWidget(self.image_label)
        parent.addWidget(scroll_area)

    def _create_edit_panel(self, parent):
        """创建编辑面板"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(320)
        scroll_area.setMaximumWidth(450)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # 标题
        title = QLabel("编辑面板")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # 创建滑块字典
        self.sliders = {}

        # 基本调整组
        basic_group = QGroupBox("基本调整")
        basic_layout = QVBoxLayout(basic_group)

        # 曝光
        slider = EditSlider("曝光", "exposure", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["exposure"] = slider
        basic_layout.addWidget(slider)

        # 对比度
        slider = EditSlider("对比度", "contrast", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["contrast"] = slider
        basic_layout.addWidget(slider)

        # 高光
        slider = EditSlider("高光", "highlights", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["highlights"] = slider
        basic_layout.addWidget(slider)

        # 阴影
        slider = EditSlider("阴影", "shadows", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["shadows"] = slider
        basic_layout.addWidget(slider)

        # 白色色阶
        slider = EditSlider("白色色阶", "whites", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["whites"] = slider
        basic_layout.addWidget(slider)

        # 黑色色阶
        slider = EditSlider("黑色色阶", "blacks", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["blacks"] = slider
        basic_layout.addWidget(slider)

        layout.addWidget(basic_group)

        # 效果调整组
        effects_group = QGroupBox("效果调整")
        effects_layout = QVBoxLayout(effects_group)

        # 纹理
        slider = EditSlider("纹理", "texture", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["texture"] = slider
        effects_layout.addWidget(slider)

        # 清晰度
        slider = EditSlider("清晰度", "clarity", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["clarity"] = slider
        effects_layout.addWidget(slider)

        # 去朦胧
        slider = EditSlider("去朦胧", "dehaze", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["dehaze"] = slider
        effects_layout.addWidget(slider)

        layout.addWidget(effects_group)

        # ==================== 颜色面板 ====================
        color_group = QGroupBox("颜色面板")
        color_main_layout = QVBoxLayout(color_group)

        # 白平衡
        wb_label = QLabel("白平衡")
        wb_label.setStyleSheet("font-weight: bold;")
        color_main_layout.addWidget(wb_label)

        # 色温
        slider = EditSlider("色温", "temp", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["temp"] = slider
        color_main_layout.addWidget(slider)

        # 色调
        slider = EditSlider("色调", "tint", -150, 150)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["tint"] = slider
        color_main_layout.addWidget(slider)

        # 鲜艳度
        slider = EditSlider("鲜艳度", "vibrance", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["vibrance"] = slider
        color_main_layout.addWidget(slider)

        # 饱和度
        slider = EditSlider("饱和度", "saturation", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["saturation"] = slider
        color_main_layout.addWidget(slider)

        # HSL 颜色混合器
        hsl_label = QLabel("HSL 颜色混合器")
        hsl_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        color_main_layout.addWidget(hsl_label)

        # 颜色选择下拉框
        self.hsl_color_combo = QComboBox()
        self.hsl_color_combo.addItems(["红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "洋红"])
        self.hsl_color_combo.currentIndexChanged.connect(self._on_hsl_color_changed)
        color_main_layout.addWidget(self.hsl_color_combo)

        # HSL 滑块
        self.hsl_sliders = {}
        colors = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]

        slider = EditSlider("色相", "hsl_hue_red", -100, 100)
        slider.valueChanged.connect(self._on_hsl_param_changed)
        self.hsl_sliders["hue"] = slider
        color_main_layout.addWidget(slider)

        slider = EditSlider("饱和度", "hsl_sat_red", -100, 100)
        slider.valueChanged.connect(self._on_hsl_param_changed)
        self.hsl_sliders["sat"] = slider
        color_main_layout.addWidget(slider)

        slider = EditSlider("明度", "hsl_lum_red", -100, 100)
        slider.valueChanged.connect(self._on_hsl_param_changed)
        self.hsl_sliders["lum"] = slider
        color_main_layout.addWidget(slider)

        self.hsl_colors = colors
        self._update_hsl_sliders()

        # 颜色分级
        cg_label = QLabel("颜色分级")
        cg_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        color_main_layout.addWidget(cg_label)

        # 阴影
        cg_shadows_label = QLabel("阴影")
        cg_shadows_label.setStyleSheet("color: #888;")
        color_main_layout.addWidget(cg_shadows_label)

        slider = EditSlider("色相", "cg_shadows_hue", 0, 360)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["cg_shadows_hue"] = slider
        color_main_layout.addWidget(slider)

        slider = EditSlider("饱和度", "cg_shadows_sat", 0, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["cg_shadows_sat"] = slider
        color_main_layout.addWidget(slider)

        # 高光
        cg_highlights_label = QLabel("高光")
        cg_highlights_label.setStyleSheet("color: #888;")
        color_main_layout.addWidget(cg_highlights_label)

        slider = EditSlider("色相", "cg_highlights_hue", 0, 360)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["cg_highlights_hue"] = slider
        color_main_layout.addWidget(slider)

        slider = EditSlider("饱和度", "cg_highlights_sat", 0, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["cg_highlights_sat"] = slider
        color_main_layout.addWidget(slider)

        # 混合与平衡
        slider = EditSlider("混合", "cg_blending", 0, 100, default_val=50)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["cg_blending"] = slider
        color_main_layout.addWidget(slider)

        slider = EditSlider("平衡", "cg_balance", -100, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["cg_balance"] = slider
        color_main_layout.addWidget(slider)

        layout.addWidget(color_group)

        # ==================== 细节面板 ====================
        detail_group = QGroupBox("细节面板")
        detail_layout = QVBoxLayout(detail_group)

        # 锐化
        sharpen_label = QLabel("锐化")
        sharpen_label.setStyleSheet("font-weight: bold;")
        detail_layout.addWidget(sharpen_label)

        slider = EditSlider("数量", "sharpen_amount", 0, 150)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["sharpen_amount"] = slider
        detail_layout.addWidget(slider)

        slider = EditSlider("半径", "sharpen_radius", 0.5, 3.0, default_val=1.0, decimals=1)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["sharpen_radius"] = slider
        detail_layout.addWidget(slider)

        slider = EditSlider("细节", "sharpen_detail", 0, 100, default_val=25)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["sharpen_detail"] = slider
        detail_layout.addWidget(slider)

        slider = EditSlider("遮罩", "sharpen_masking", 0, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["sharpen_masking"] = slider
        detail_layout.addWidget(slider)

        # 降噪
        noise_label = QLabel("降噪")
        noise_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        detail_layout.addWidget(noise_label)

        slider = EditSlider("明度", "noise_luminance", 0, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["noise_luminance"] = slider
        detail_layout.addWidget(slider)

        slider = EditSlider("颜色", "noise_color", 0, 100)
        slider.valueChanged.connect(self._on_param_changed)
        self.sliders["noise_color"] = slider
        detail_layout.addWidget(slider)

        layout.addWidget(detail_group)

        # 重置按钮
        reset_btn = QPushButton("重置所有调整")
        reset_btn.clicked.connect(self.reset_edits)
        layout.addWidget(reset_btn)

        layout.addStretch()

        scroll_area.setWidget(panel)
        parent.addWidget(scroll_area)

    def _on_param_changed(self, param_name: str, value: float):
        """参数变化时更新图像"""
        if self._updating_image:
            return

        self.editor.set_param(param_name, value)
        self._update_display()

    def _on_hsl_color_changed(self, index: int):
        """HSL颜色选择改变时更新滑块"""
        self._update_hsl_sliders()

    def _on_hsl_param_changed(self, param_name: str, value: float):
        """HSL参数变化时处理"""
        if self._updating_image:
            return

        color_idx = self.hsl_color_combo.currentIndex()
        color = self.hsl_colors[color_idx]

        # 确定是hue、sat还是lum
        if "hue" in param_name:
            actual_param = f"hsl_hue_{color}"
        elif "sat" in param_name:
            actual_param = f"hsl_sat_{color}"
        else:
            actual_param = f"hsl_lum_{color}"

        self.editor.set_param(actual_param, value)
        self._update_display()

    def _update_hsl_sliders(self):
        """更新HSL滑块显示当前选中颜色的值"""
        color_idx = self.hsl_color_combo.currentIndex()
        color = self.hsl_colors[color_idx]

        self._updating_image = True
        try:
            # 获取当前颜色的参数值
            hue_val = self.editor.get_param(f"hsl_hue_{color}")
            sat_val = self.editor.get_param(f"hsl_sat_{color}")
            lum_val = self.editor.get_param(f"hsl_lum_{color}")

            # 更新滑块
            self.hsl_sliders["hue"].blockSignals(True)
            self.hsl_sliders["hue"].set_value(hue_val)
            self.hsl_sliders["hue"].blockSignals(False)

            self.hsl_sliders["sat"].blockSignals(True)
            self.hsl_sliders["sat"].set_value(sat_val)
            self.hsl_sliders["sat"].blockSignals(False)

            self.hsl_sliders["lum"].blockSignals(True)
            self.hsl_sliders["lum"].set_value(lum_val)
            self.hsl_sliders["lum"].blockSignals(False)
        finally:
            self._updating_image = False

    def _update_display(self):
        """更新图像显示"""
        if not self.editor.has_image():
            return

        self._updating_image = True
        try:
            self._display_image()
        finally:
            self._updating_image = False

    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            get_supported_image_filters()
        )

        if file_path:
            try:
                self.editor.load_image(file_path)
                self._display_image()
                self.statusBar().showMessage(f"已加载: {file_path}")
                self._reset_sliders()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载图片: {e}")

    def _display_image(self):
        """显示图像"""
        if self.editor.has_image():
            image = self.editor.get_edited()
            if image is None:
                image = self.editor.get_original()

            # 确保图像是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # PIL Image 转 QPixmap
            # 注意：需要指定 bytesPerLine 以避免图像偏移问题
            width, height = image.size
            bytes_per_line = 3 * width

            qimage = QImage(image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimage)

            # 缩放以适应显示区域
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def save_image(self):
        """保存图像"""
        if not self.editor.has_image():
            QMessageBox.warning(self, "警告", "没有可保存的图片")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            "",
            get_save_filters()
        )

        if file_path:
            # 确保文件扩展名正确
            ext = get_file_extension(selected_filter)
            if not file_path.lower().endswith(ext):
                file_path += ext

            if self.editor.save_image(file_path):
                self.statusBar().showMessage(f"已保存: {file_path}")
                QMessageBox.information(self, "成功", "图片保存成功!")
            else:
                QMessageBox.critical(self, "错误", "保存失败")

    def _reset_sliders(self):
        """重置所有滑块到默认值"""
        for slider in self.sliders.values():
            slider.blockSignals(True)
            slider.reset()
            slider.blockSignals(False)

    def reset_edits(self):
        """重置所有编辑"""
        self.editor.reset()
        self._reset_sliders()
        self._update_hsl_sliders()
        self._display_image()
        self.statusBar().showMessage("已重置所有调整")

    def import_preset(self):
        """导入预设文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入预设",
            "",
            "JSON文件 (*.json);;所有文件 (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)

                self.editor.set_params_from_dict(params)
                self._update_sliders_from_params()
                self._display_image()
                self.statusBar().showMessage(f"已导入预设: {file_path}")
                QMessageBox.information(self, "成功", "预设导入成功!")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入预设失败: {e}")

    def export_preset(self):
        """导出预设文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出预设",
            "",
            "JSON文件 (*.json)"
        )

        if file_path:
            if not file_path.endswith('.json'):
                file_path += '.json'

            try:
                params = self.editor.get_params_dict()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=2, ensure_ascii=False)

                self.statusBar().showMessage(f"已导出预设: {file_path}")
                QMessageBox.information(self, "成功", "预设导出成功!")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出预设失败: {e}")

    def _update_sliders_from_params(self):
        """从编辑器参数更新所有滑块"""
        params = self.editor.get_all_params()

        self._updating_image = True
        try:
            # 更新普通滑块
            for key, slider in self.sliders.items():
                if hasattr(params, key):
                    value = getattr(params, key)
                    slider.blockSignals(True)
                    slider.set_value(value)
                    slider.blockSignals(False)

            # 更新HSL滑块
            self._update_hsl_sliders()
        finally:
            self._updating_image = False

    def resizeEvent(self, event):
        """窗口大小改变时重新显示图像"""
        super().resizeEvent(event)
        if self.editor.has_image():
            self._display_image()
