# 商品图片自动分类工具

基于 **OpenAI CLIP** 图像特征提取 + **KMeans** 无监督聚类，自动将商品图片按视觉相似度分组。

## 功能特性

- 图形界面选择文件夹，无需命令行
- CLIP 模型自动提取图片视觉语义特征
- KMeans 自动聚类，支持手动设置分组数或自动推断
- 输出 `group_01 / group_02 / …` 分类文件夹，图片原地复制
- 自动跳过损坏图片，异常信息打印到控制台
- 支持 PyInstaller 打包为 Windows 独立 EXE

## 支持的图片格式

JPG / JPEG / PNG / BMP / GIF / WEBP / TIFF

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **GPU 加速（可选）**：若有 NVIDIA 显卡，建议安装 CUDA 版 PyTorch，速度可提升 5-10 倍。
> 访问 https://pytorch.org/get-started/locally/ 获取对应安装命令。

### 2. 运行

```bash
python image_classifier.py
```

### 3. 使用步骤

1. 点击「浏览…」选择装有商品图片的文件夹
2. 设置分组数量（或勾选「自动推断」）
3. 点击「▶ 开始分类」
4. 等待完成后，在原文件夹中查看 `group_01`、`group_02` 等子文件夹

## 打包为 Windows EXE

```bat
build_exe.bat
```

打包完成后 EXE 位于 `dist/商品图片分类工具.exe`，可直接分发给团队成员使用（无需安装 Python 环境）。

> **首次运行 EXE 时**，程序会自动从 HuggingFace 下载 CLIP 模型（约 600 MB）并缓存到  
> `C:\Users\<用户名>\.cache\huggingface\`，后续运行无需重复下载。

## 输出目录结构示例

```
商品图片文件夹/
├── img001.jpg          ← 原始图片保留不动
├── img002.jpg
├── ...
├── group_01/           ← 分类结果
│   ├── img003.jpg
│   └── img009.jpg
├── group_02/
│   ├── img001.jpg
│   └── img007.jpg
└── group_03/
    └── ...
```

## 分组数量建议

| 图片总数 | 建议分组数 |
|---------|-----------|
| 50 张   | 5–8       |
| 100 张  | 8–15      |
| 300 张  | 15–30     |
| 500 张  | 20–50     |

或直接勾选「自动推断」，程序将使用 `图片总数 ÷ 10` 作为分组数。
