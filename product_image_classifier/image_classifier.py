#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品图片自动分类工具
基于 OpenAI CLIP 特征提取 + KMeans 无监督聚类
支持 PyInstaller 打包为 Windows 独立 EXE
"""

# freeze_support 必须在所有 import 之前调用，以支持 Windows 多进程 + PyInstaller
import multiprocessing
multiprocessing.freeze_support()

import os
import sys
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ────────────────────────────────────────────
# 常量
# ────────────────────────────────────────────

SUPPORTED_EXT: set = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
CLIP_MODEL_ID: str = "openai/clip-vit-base-patch32"


# ────────────────────────────────────────────
# 特征提取
# ────────────────────────────────────────────

class CLIPExtractor:
    """封装 CLIP 图像特征提取，线程安全（单线程调用）。"""

    def __init__(self) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: Optional[CLIPModel] = None
        self._proc: Optional[CLIPProcessor] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, on_status: Optional[Callable[[str], None]] = None) -> None:
        if self.loaded:
            return
        if on_status:
            on_status("正在加载 CLIP 模型（首次运行需下载约 600 MB）…")
        self._model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        self._proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self._model.to(self.device).eval()

    def extract(
        self,
        paths: List[str],
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        逐张提取 CLIP 图像特征。
        损坏 / 无法读取的图片自动跳过并将异常打印到控制台。

        进度回调范围：0 % → 50 %（后半段留给文件复制）。
        返回：(特征矩阵, 对应的有效路径列表)
        """
        feats: List[np.ndarray] = []
        valid: List[str] = []
        total = len(paths)

        for idx, p in enumerate(paths):
            if on_status:
                on_status(f"提取特征 [{idx + 1}/{total}]：{Path(p).name}")
            if on_progress:
                on_progress((idx + 1) / total * 50)

            try:
                img = Image.open(p).convert("RGB")
                inputs = self._proc(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    vec = self._model.get_image_features(**inputs)
                feats.append(vec.cpu().numpy().ravel())
                valid.append(p)
            except Exception as exc:
                print(f"[SKIP] 无法处理图片：{p}\n       原因：{exc}", flush=True)

        if not feats:
            return np.empty((0,)), valid

        return np.vstack(feats), valid


# ────────────────────────────────────────────
# 聚类
# ────────────────────────────────────────────

def kmeans_cluster(features: np.ndarray, n_clusters: int) -> np.ndarray:
    """对 L2 归一化特征执行 KMeans，返回每张图片的类别标签。"""
    normed = normalize(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    return km.fit_predict(normed)


# ────────────────────────────────────────────
# 文件操作
# ────────────────────────────────────────────

def copy_to_groups(
    root_dir: str,
    paths: List[str],
    labels: np.ndarray,
    n_clusters: int,
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> None:
    """
    在 root_dir 下创建 group_01 … group_NN 文件夹，
    并将每张图片复制到对应分组中。
    进度回调范围：50 % → 100 %。
    """
    root = Path(root_dir)

    # 预先建好所有分组文件夹
    for i in range(n_clusters):
        (root / f"group_{i + 1:02d}").mkdir(exist_ok=True)

    total = len(paths)
    for i, (src, lbl) in enumerate(zip(paths, labels)):
        dst_dir = root / f"group_{int(lbl) + 1:02d}"
        dst = dst_dir / Path(src).name

        # 目标文件已存在时加序号后缀，避免覆盖
        if dst.exists():
            stem, sfx = Path(src).stem, Path(src).suffix
            dst = dst_dir / f"{stem}_{i}{sfx}"

        shutil.copy2(src, dst)

        if on_progress:
            on_progress(50.0 + (i + 1) / total * 50.0)

        if on_log and (i + 1) % 50 == 0:
            on_log(f"  已复制 {i + 1}/{total} 张…")


# ────────────────────────────────────────────
# GUI
# ────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.extractor = CLIPExtractor()
        self._init_window()
        self._build_ui()

    # ── 窗口 ────────────────────────────────

    def _init_window(self) -> None:
        self.root.title("商品图片自动分类工具  v1.0")
        W, H = 720, 590
        self.root.geometry(f"{W}x{H}")
        self.root.resizable(True, True)
        self.root.minsize(620, 520)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"+{(sw - W) // 2}+{(sh - H) // 2}")

    # ── UI 构建 ──────────────────────────────

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TLabelframe.Label", font=("微软雅黑", 9, "bold"))
        style.configure("Run.TButton", font=("微软雅黑", 10, "bold"), padding=(10, 6))

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(
            outer, text="商品图片自动分类工具",
            font=("微软雅黑", 16, "bold")
        ).pack(pady=(0, 14))

        # ── 文件夹 ─────────────────────────
        frm_dir = ttk.LabelFrame(outer, text=" 图片文件夹 ", padding=10)
        frm_dir.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.var_folder = tk.StringVar()
        ttk.Entry(frm_dir, textvariable=self.var_folder,
                  font=("微软雅黑", 9)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(frm_dir, text="浏览…",
                   command=self._browse).pack(side=tk.RIGHT)

        # ── 聚类设置 ────────────────────────
        frm_cfg = ttk.LabelFrame(outer, text=" 聚类设置 ", padding=10)
        frm_cfg.pack(fill=tk.X, padx=4, pady=(0, 8))

        row = ttk.Frame(frm_cfg)
        row.pack(fill=tk.X)

        ttk.Label(row, text="分组数量：").pack(side=tk.LEFT)
        self.var_k = tk.IntVar(value=10)
        self.spin_k = ttk.Spinbox(
            row, from_=2, to=99, textvariable=self.var_k, width=7)
        self.spin_k.pack(side=tk.LEFT, padx=(6, 0))

        self.var_auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row,
            text="自动推断（图片总数 ÷ 10，至少 2 组）",
            variable=self.var_auto,
            command=self._toggle_auto,
        ).pack(side=tk.LEFT, padx=(16, 0))

        ttk.Label(frm_cfg,
                  text="提示：分组数越少，每组图片越多；分组数越多，分类越细。",
                  foreground="gray", font=("微软雅黑", 8)).pack(
            anchor=tk.W, pady=(6, 0))

        # ── 进度 ────────────────────────────
        frm_prog = ttk.LabelFrame(outer, text=" 进度 ", padding=10)
        frm_prog.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.var_prog = tk.DoubleVar()
        ttk.Progressbar(frm_prog, variable=self.var_prog,
                        maximum=100).pack(fill=tk.X, pady=(0, 5))
        self.var_status = tk.StringVar(
            value="就绪 — 请选择文件夹后点击「开始分类」")
        ttk.Label(frm_prog, textvariable=self.var_status,
                  font=("微软雅黑", 9)).pack(anchor=tk.W)

        # ── 日志 ────────────────────────────
        frm_log = ttk.LabelFrame(outer, text=" 运行日志 ", padding=10)
        frm_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 10))

        self.log_box = tk.Text(
            frm_log, height=8, state=tk.DISABLED,
            font=("Consolas", 9),
            bg="#1e1e2e", fg="#cdd6f4",
            insertbackground="white", relief=tk.FLAT,
        )
        sb = ttk.Scrollbar(frm_log, orient=tk.VERTICAL,
                           command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ── 按钮 ────────────────────────────
        self.btn_start = ttk.Button(
            outer, text="▶  开始分类",
            style="Run.TButton",
            command=self._on_start,
            width=18,
        )
        self.btn_start.pack()

    # ── 事件 ────────────────────────────────

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="选择包含商品图片的文件夹")
        if path:
            self.var_folder.set(path)

    def _toggle_auto(self) -> None:
        state = tk.DISABLED if self.var_auto.get() else tk.NORMAL
        self.spin_k.configure(state=state)

    def _on_start(self) -> None:
        folder = self.var_folder.get().strip()
        if not folder:
            messagebox.showwarning("提示", "请先选择图片文件夹！")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("错误", f"路径无效：{folder}")
            return
        self.btn_start.configure(state=tk.DISABLED)
        self.var_prog.set(0)
        threading.Thread(target=self._run, args=(folder,), daemon=True).start()

    # ── 线程安全工具 ─────────────────────────

    def _log(self, msg: str) -> None:
        """线程安全地向日志框追加文本，并同步打印到控制台。"""
        def _append() -> None:
            self.log_box.configure(state=tk.NORMAL)
            self.log_box.insert(tk.END, msg + "\n")
            self.log_box.see(tk.END)
            self.log_box.configure(state=tk.DISABLED)
        self.root.after(0, _append)
        print(msg, flush=True)

    def _set_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.var_status.set(msg))

    def _set_progress(self, val: float) -> None:
        self.root.after(0, lambda: self.var_prog.set(val))

    # ── 主流程（后台线程）────────────────────

    def _run(self, folder: str) -> None:
        try:
            # ① 扫描图片
            self._set_status("扫描图片文件…")
            paths: List[str] = sorted(
                str(p) for p in Path(folder).iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
            )

            if not paths:
                self.root.after(0, lambda: messagebox.showwarning(
                    "提示",
                    "所选文件夹中未找到图片文件！\n"
                    "支持格式：JPG / JPEG / PNG / BMP / GIF / WEBP / TIFF",
                ))
                return

            self._log(f"找到 {len(paths)} 张图片")

            # ② 确定分组数
            if self.var_auto.get():
                k = max(2, len(paths) // 10)
                self._log(f"自动推断分组数：{k}")
            else:
                k = max(2, int(self.var_k.get()))

            k = min(k, len(paths))
            self._log(f"目标分组数：{k}")

            # ③ 加载 CLIP 模型
            if not self.extractor.loaded:
                self._log("加载 CLIP 模型…（首次运行需从 HuggingFace 下载约 600 MB，请耐心等待）")
            self.extractor.load(on_status=self._set_status)
            device_label = "GPU (CUDA)" if self.extractor.device == "cuda" else "CPU"
            self._log(f"CLIP 模型就绪，推理设备：{device_label}")
            self._set_progress(5)

            # ④ 提取图片特征
            self._log("开始提取图片视觉特征…")
            feats, valid_paths = self.extractor.extract(
                paths,
                on_status=self._set_status,
                on_progress=self._set_progress,
            )

            skipped = len(paths) - len(valid_paths)
            if skipped:
                self._log(f"⚠  跳过损坏 / 无法读取的图片 {skipped} 张（详见控制台）")
            self._log(f"特征提取完成：有效 {len(valid_paths)} 张")

            if len(valid_paths) == 0:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", "没有可处理的有效图片！"))
                return

            # 若有效图片少于预设分组数，自动缩减
            k = min(k, len(valid_paths))

            # ⑤ KMeans 聚类
            self._set_status("KMeans 聚类分析中…")
            self._log(f"执行 KMeans 聚类（k={k}）…")
            labels = kmeans_cluster(feats, k)
            self._set_progress(52)

            # 打印各组图片数量
            unique, counts = np.unique(labels, return_counts=True)
            dist = "  ".join(
                f"group_{int(l) + 1:02d}:{c}" for l, c in zip(unique, counts)
            )
            self._log(f"各组图片数：{dist}")

            # ⑥ 复制图片到分组文件夹
            self._set_status("复制图片到分类文件夹…")
            self._log("开始复制图片文件…")
            copy_to_groups(
                folder, valid_paths, labels, k,
                on_progress=self._set_progress,
                on_log=self._log,
            )

            # ⑦ 完成
            self._set_progress(100)
            self._set_status(f"✅ 完成！{len(valid_paths)} 张图片 → {k} 个分组")
            self._log("─" * 52)
            self._log("✅ 分类完成！")
            self._log(f"   图片总数  ：{len(paths)} 张")
            self._log(f"   成功处理  ：{len(valid_paths)} 张")
            self._log(f"   跳过（损坏）：{skipped} 张")
            self._log(f"   创建分组  ：{k} 组")
            self._log(f"   输出目录  ：{folder}")

            self.root.after(0, lambda: messagebox.showinfo(
                "分类完成",
                f"处理完成！\n\n"
                f"图片总数：{len(paths)} 张\n"
                f"成功处理：{len(valid_paths)} 张\n"
                f"跳过（损坏）：{skipped} 张\n"
                f"创建分组：{k} 组\n\n"
                f"分类结果已保存至所选文件夹内的 group_XX 子文件夹",
            ))

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self._log(f"❌ 发生错误：\n{tb}")
            msg = str(exc)
            self.root.after(0, lambda: messagebox.showerror(
                "错误", f"处理失败：\n{msg}"))

        finally:
            self.root.after(0, lambda: self.btn_start.configure(state=tk.NORMAL))


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
