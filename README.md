# Python Lightroom Tool

一个基于 Python 的 Lightroom 风格图像编辑工具，支持非破坏性编辑。

## 功能特性

- 非破坏性编辑：永远不修改原始文件，通过参数栈实时计算效果
- 支持 JPG、PNG、BMP、TIFF 等常见图像格式
- 实时预览编辑效果

## 已实现功能模块

### 基本调整面板 (Basic Panel) ✅
| 功能 | 参数范围 | 说明 |
|------|----------|------|
| 曝光 | -100 ~ +100 | 调整整体亮度，影响中间调 |
| 对比度 | -100 ~ +100 | 调整画面明暗反差 |
| 高光 | -100 ~ +100 | 调整画面中最亮的部分 |
| 阴影 | -100 ~ +100 | 调整画面中最暗的部分 |
| 白色色阶 | -100 ~ +100 | 调整图像最亮端的极限值 |
| 黑色色阶 | -100 ~ +100 | 调整图像最暗端的极限值 |

### 效果调整面板 (Effects Panel) ✅
| 功能 | 参数范围 | 说明 |
|------|----------|------|
| 纹理 | -100 ~ +100 | 增强/平滑中等尺度细节 |
| 清晰度 | -100 ~ +100 | 通过局部对比度增强深度和质感 |
| 去朦胧 | -100 ~ +100 | 移除或添加画面中的雾霾感 |

### 颜色调整面板 (Color Panel) ✅
| 功能 | 参数范围 | 说明 |
|------|----------|------|
| 鲜艳度 | -100 ~ +100 | 智能调整低饱和度颜色 |
| 饱和度 | -100 ~ +100 | 统一调整所有颜色饱和度 |

### 其他已实现功能 ✅
- 图像加载与显示（支持缩放预览）
- 编辑后图像保存（全分辨率输出）
- 参数重置功能
- 快捷键支持

## 环境要求

- Python 3.10+
- Conda（推荐）

## 安装步骤

### 1. 创建 Conda 环境

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate lightroom
```

或者手动创建：

```bash
conda create -n lightroom python=3.10
conda activate lightroom
conda install -c conda-forge pyqt pillow
```

### 2. 运行程序

```bash
cd lightroom_tool
python main.py
```

或者在 Windows 上双击 `run.bat`

## 使用说明

### 打开图像
- 点击菜单 `文件 -> 打开` 或工具栏 `打开` 按钮
- 支持格式：JPG, JPEG, PNG, BMP, TIF, TIFF

### 编辑图像
- 在右侧编辑面板调整各项参数
- 参数调整会实时反映在图像预览中
- 所有编辑均为非破坏性，可随时重置

### 保存图像
- 点击菜单 `文件 -> 保存` 或工具栏 `保存` 按钮
- 选择保存路径和格式

### 重置编辑
- 点击工具栏 `重置` 按钮或编辑面板 `重置所有调整` 按钮
- 所有参数恢复为默认值

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+O | 打开图像 |
| Ctrl+S | 保存图像 |
| Ctrl+Q | 退出程序 |

## 项目结构

```
lightroom_tool/
├── main.py              # 应用入口
├── environment.yml      # Conda 环境配置
├── requirements.txt     # pip 依赖
├── run.bat              # Windows 启动脚本
├── setup_env.bat        # 环境安装脚本（Windows）
├── setup_env.sh         # 环境安装脚本（Linux/Mac）
├── ui/
│   ├── __init__.py
│   └── main_window.py   # 主窗口 UI
├── core/
│   ├── __init__.py
│   └── editor.py        # 图像编辑核心功能
└── utils/
    ├── __init__.py
    └── file_utils.py    # 文件操作工具
```

## 技术栈

- **GUI 框架**: PyQt5
- **图像处理**: Pillow (PIL)
