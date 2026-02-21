"""色调曲线编辑控件 - 点曲线（自由拖动）"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QMouseEvent, QWheelEvent, QPixmap, QPolygonF
)
import numpy as np


class CurveWidget(QWidget):
    """色调曲线编辑控件 - 点曲线（自由拖动）

    支持的功能：
    - 显示图像直方图作为背景参考
    - 可自由拖动的曲线控制点
    - 使用样条插值平滑连接控制点
    - 4种通道：RGB、Red、Green、Blue
    - 双击添加控制点，双击端点删除（除了固定端点）
    - 鼠标拖动控制点实时更新
    """

    valueChanged = pyqtSignal(str, list)  # channel, points [(x,y), ...]

    # 固定点半径
    POINT_RADIUS = 6
    # 悬停点半径
    HOVER_RADIUS = 8
    # 画布内边距
    MARGIN = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_channel = "rgb"

        # 各通道的控制点
        self._curves = {
            "rgb": [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)],
            "red": [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)],
            "green": [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)],
            "blue": [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]
        }

        # 交互状态
        self._dragging_point_idx = -1
        self._hovered_point_idx = -1
        self._curve_color = QColor(200, 200, 200)

        # 直方图数据 (缓存)
        self._histogram = None

        # 设置大小策略
        self.setMinimumSize(280, 280)
        self.setSizePolicy(self.sizePolicy().Preferred, self.sizePolicy().Fixed)

    def set_histogram(self, histogram: np.ndarray):
        """设置直方图数据

        Args:
            histogram: 形状为 (256, 3) 的数组，对应R、G、B通道的直方图
        """
        self._histogram = histogram
        self.update()

    def set_channel(self, channel: str):
        """设置当前编辑的通道

        Args:
            channel: 通道名称 ('rgb', 'red', 'green', 'blue')
        """
        self._current_channel = channel

        # 根据通道设置曲线颜色
        if channel == "rgb":
            self._curve_color = QColor(200, 200, 200)
        elif channel == "red":
            self._curve_color = QColor(255, 100, 100)
        elif channel == "green":
            self._curve_color = QColor(100, 255, 100)
        elif channel == "blue":
            self._curve_color = QColor(100, 100, 255)

        self.update()

    def set_curve(self, channel: str, points: list):
        """设置指定通道的控制点

        Args:
            channel: 通道名称
            points: 控制点列表
        """
        if channel in self._curves:
            self._curves[channel] = points
            if channel == self._current_channel:
                self.update()

    def get_curve(self, channel: str) -> list:
        """获取指定通道的控制点"""
        return self._curves.get(channel, [])

    def reset_curve(self, channel: str = None):
        """重置曲线

        Args:
            channel: 要重置的通道，如果为None则重置所有通道
        """
        default = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]

        if channel is None:
            for ch in self._curves:
                self._curves[ch] = list(default)
        elif channel in self._curves:
            self._curves[channel] = list(default)

        self.update()

    def _get_canvas_rect(self) -> QRectF:
        """获取画布区域（除去边距）"""
        return QRectF(
            self.MARGIN, self.MARGIN,
            self.width() - 2 * self.MARGIN,
            self.height() - 2 * self.MARGIN
        )

    def _value_to_coords(self, x: int, y: int) -> QPointF:
        """将值坐标转换为屏幕坐标

        Args:
            x, y: 0-255 的值坐标

        Returns:
            屏幕坐标
        """
        canvas = self._get_canvas_rect()
        sx = canvas.left() + (x / 255.0) * canvas.width()
        # Y轴反转：0在底部，255在顶部
        sy = canvas.bottom() - (y / 255.0) * canvas.height()
        return QPointF(sx, sy)

    def _coords_to_value(self, sx: float, sy: float) -> tuple:
        """将屏幕坐标转换为值坐标

        Args:
            sx, sy: 屏幕坐标

        Returns:
            (x, y) 0-255 的值坐标
        """
        canvas = self._get_canvas_rect()
        x = ((sx - canvas.left()) / canvas.width()) * 255
        y = ((canvas.bottom() - sy) / canvas.height()) * 255
        return int(round(np.clip(x, 0, 255))), int(round(np.clip(y, 0, 255)))

    def _build_cubic_spline(self, points: list, num_samples: int = 256) -> list:
        """使用三次样条插值构建平滑曲线

        Args:
            points: 控制点列表
            num_samples: 采样点数

        Returns:
            插值点列表
        """
        if len(points) < 2:
            return points

        # 按x排序控制点
        sorted_points = sorted(points, key=lambda p: p[0])
        x_vals = [float(p[0]) for p in sorted_points]
        y_vals = [float(p[1]) for p in sorted_points]

        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(x_vals, y_vals, bc_type='natural', extrapolate=True)
            x_sample = np.linspace(0, 255, num_samples)
            y_sample = cs(x_sample)
            y_sample = np.clip(y_sample, 0, 255)
            return list(zip(x_sample, y_sample))
        except ImportError:
            # 如果没有scipy，使用简单的线性插值
            return [(x, 255 - x + y) for x, y in sorted_points]

    def paintEvent(self, event):
        """绘制控件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        canvas = self._get_canvas_rect()

        # 1. 绘制方格背景（透明度参考）
        self._draw_grid(painter, canvas)

        # 2. 绘制直方图（如果有）
        if self._histogram is not None:
            self._draw_histogram(painter, canvas)

        # 3. 绘制对角线（参考线）
        pen = QPen(QColor(80, 80, 80), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(
            self._value_to_coords(0, 0),
            self._value_to_coords(255, 255)
        )

        # 4. 绘制曲线
        current_points = self._curves.get(self._current_channel, [])
        if len(current_points) >= 2:
            self._draw_curve(painter, current_points)

        # 5. 绘制控制点
        self._draw_control_points(painter, current_points)

    def _draw_grid(self, painter: QPainter, canvas: QRectF):
        """绘制方格背景（透明度参考）"""
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        # 绘制4x4灰色棋盘格
        cell_w = canvas.width() / 4
        cell_h = canvas.height() / 4

        for row in range(4):
            for col in range(4):
                if (row + col) % 2 == 0:
                    x = canvas.left() + col * cell_w
                    y = canvas.top() + row * cell_h
                    painter.fillRect(QRectF(x, y, cell_w, cell_h),
                                    QColor(50, 50, 50))

        # 绘制边框
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(canvas)

    def _draw_histogram(self, painter: QPainter, canvas: QRectF):
        """绘制直方图背景"""
        if self._histogram is None or len(self._histogram) == 0:
            return

        max_count = np.max(self._histogram)
        if max_count == 0:
            max_count = 1

        bar_w = float(canvas.width() / 256)

        # 绘制RGB直方图叠加
        for i in range(min(256, len(self._histogram))):
            # histogram[i] 可能是单个值或数组
            hist_val = self._histogram[i]
            if isinstance(hist_val, np.ndarray):
                # 如果是数组，取平均值或最大值
                hist_val = float(np.max(hist_val))
            else:
                hist_val = float(hist_val)

            h = float((hist_val / max_count) * canvas.height() * 0.7)
            x = float(canvas.left() + i * bar_w)
            y = float(canvas.bottom() - h)

            # 使用灰色绘制直方图
            color = QColor(80, 80, 80, 150)
            painter.fillRect(QRectF(x, y, bar_w, max(0.0, h)), color)

    def _draw_curve(self, painter: QPainter, points: list):
        """绘制曲线"""
        # 获取插值点
        spline_points = self._build_cubic_spline(points)

        if len(spline_points) < 2:
            return

        # 创建路径
        path = QPainterPath()
        start = self._value_to_coords(spline_points[0][0], spline_points[0][1])
        path.moveTo(start)

        for x, y in spline_points[1:]:
            point = self._value_to_coords(x, y)
            path.lineTo(point)

        # 绘制曲线
        pen = QPen(self._curve_color, 2)
        painter.setPen(pen)
        painter.drawPath(path)

    def _draw_control_points(self, painter: QPainter, points: list):
        """绘制控制点"""
        for i, (x, y) in enumerate(points):
            center = self._value_to_coords(x, y)

            is_fixed = (i == 0 or i == len(points) - 1)  # 端点固定
            is_hovered = (i == self._hovered_point_idx)
            is_dragging = (i == self._dragging_point_idx)

            radius = self.HOVER_RADIUS if (is_hovered or is_dragging) else self.POINT_RADIUS

            if is_fixed:
                # 固定点：灰色
                color = QColor(150, 150, 150)
            else:
                # 可动点：白色
                color = QColor(255, 255, 255)

            # 绘制填充
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(self._curve_color, 2))
            painter.drawEllipse(center, radius, radius)

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            current_points = self._curves.get(self._current_channel, [])

            # 检查是否点击到某个控制点
            for i, (x, y) in enumerate(current_points):
                center = self._value_to_coords(x, y)
                dist = np.sqrt((event.x() - center.x()) ** 2 +
                              (event.y() - center.y()) ** 2)

                if dist <= self.HOVER_RADIUS:
                    # 端点不能移动（除了可以调整Y值）
                    if i == 0:
                        # 起始端点：只能调整Y值
                        if event.button() == Qt.LeftButton:
                            # 允许调整Y
                            self._dragging_point_idx = i
                    elif i == len(current_points) - 1:
                        # 终点端点：只能调整Y值
                        if event.button() == Qt.LeftButton:
                            self._dragging_point_idx = i
                    else:
                        # 中间点可以自由移动
                        self._dragging_point_idx = i
                    break

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if self._dragging_point_idx >= 0:
            # 正在拖动
            current_points = self._curves.get(self._current_channel, [])
            idx = self._dragging_point_idx

            if idx < len(current_points):
                # 计算新坐标
                new_x, new_y = self._coords_to_value(event.x(), event.y())

                # 限制端点的X坐标
                if idx == 0:
                    new_x = 0
                elif idx == len(current_points) - 1:
                    new_x = 255
                else:
                    # 确保点不会越过相邻点
                    if idx > 0 and new_x <= current_points[idx - 1][0]:
                        new_x = current_points[idx - 1][0] + 1
                    if idx < len(current_points) - 1 and new_x >= current_points[idx + 1][0]:
                        new_x = current_points[idx + 1][0] - 1

                # 更新控制点
                current_points[idx] = (new_x, new_y)

                # 发出信号
                self._emit_curve_changed()
                self.update()
        else:
            # 更新悬停状态
            current_points = self._curves.get(self._current_channel, [])
            self._hovered_point_idx = -1

            for i, (x, y) in enumerate(current_points):
                center = self._value_to_coords(x, y)
                dist = np.sqrt((event.x() - center.x()) ** 2 +
                              (event.y() - center.y()) ** 2)

                if dist <= self.HOVER_RADIUS:
                    self._hovered_point_idx = i
                    break

            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self._dragging_point_idx = -1
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """双击事件：添加或删除控制点"""
        if event.button() == Qt.LeftButton:
            current_points = self._curves.get(self._current_channel, [])

            # 检查是否双击了某个控制点
            for i, (x, y) in enumerate(current_points):
                center = self._value_to_coords(x, y)
                dist = np.sqrt((event.x() - center.x()) ** 2 +
                              (event.y() - center.y()) ** 2)

                if dist <= self.HOVER_RADIUS:
                    # 双击了控制点：如果是端点则不能删除
                    if i != 0 and i != len(current_points) - 1:
                        # 删除控制点
                        current_points.pop(i)
                        self._emit_curve_changed()
                        self.update()
                    return

            # 双击了空白处：添加新控制点
            new_x, new_y = self._coords_to_value(event.x(), event.y())

            # 找到插入位置
            insert_idx = 0
            for i, (x, y) in enumerate(current_points):
                if new_x < x:
                    insert_idx = i
                    break
                insert_idx = i + 1

            # 插入新点
            current_points.insert(insert_idx, (new_x, new_y))
            self._emit_curve_changed()
            self.update()

    def _emit_curve_changed(self):
        """发出曲线改变信号"""
        points = self._curves.get(self._current_channel, [])
        self.valueChanged.emit(self._current_channel, points)


class CurvePanel(QWidget):
    """色调曲线面板 - 包含通道选择器、曲线控件和饱和度滑块"""

    valueChanged = pyqtSignal(str, object)  # param_name, value (可以是 float 或 list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # 通道选择器
        channel_layout = QHBoxLayout()

        channel_label = QLabel("通道:")
        channel_label.setStyleSheet("color: #666;")
        channel_layout.addWidget(channel_label)

        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["RGB", "红 (R)", "绿 (G)", "蓝 (B)"])
        self.channel_combo.setMaximumWidth(150)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)

        self.channel_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: #333;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox:hover {
                border: 1px solid #999;
            }
        """)

        channel_layout.addWidget(self.channel_combo)
        channel_layout.addStretch()
        layout.addLayout(channel_layout)

        # 曲线控件
        self.curve_widget = CurveWidget()
        self.curve_widget.valueChanged.connect(self._on_curve_changed)
        layout.addWidget(self.curve_widget)

    def set_histogram(self, histogram: np.ndarray):
        """设置直方图"""
        self.curve_widget.set_histogram(histogram)

    def set_curve(self, channel: str, points: list):
        """设置曲线"""
        self.curve_widget.set_curve(channel, points)

    def get_curve(self, channel: str) -> list:
        """获取曲线"""
        return self.curve_widget.get_curve(channel)

    def reset_curves(self):
        """重置所有曲线"""
        self.curve_widget.reset_curve()

    def _on_channel_changed(self, index: int):
        """通道改变"""
        channels = ["rgb", "red", "green", "blue"]
        self.curve_widget.set_channel(channels[index])

    def _on_curve_changed(self, channel: str, points: list):
        """曲线改变"""
        # 通过特殊信号格式传递曲线数据
        # 使用 'curve_<channel>' 作为参数名
        param_name = f'curve_{channel}'
        self.valueChanged.emit(param_name, points)