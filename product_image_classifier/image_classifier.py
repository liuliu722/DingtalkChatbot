#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品图片自动分类工具  v2.0（GPT-4o 智能命名版）
CLIP 特征提取 + KMeans 聚类 + GPT-4o 自动推断类目名称
"""

# freeze_support 必须在所有 import 之前，支持 Windows PyInstaller 多进程
import multiprocessing
multiprocessing.freeze_support()

import os
import re
import sys
import base64
import shutil
import threading
import tkinter as tk
from io import BytesIO
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from openai import OpenAI

# ────────────────────────────────────────────
# 常量
# ────────────────────────────────────────────

SUPPORTED_EXT: set = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}
CLIP_MODEL_ID: str = "openai/clip-vit-base-patch32"
GPT_MODEL: str = "gpt-4o"

# Windows 文件名非法字符
_INVALID_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


# ────────────────────────────────────────────
# CLIP 特征提取
# ────────────────────────────────────────────

class CLIPExtractor:
    """封装 CLIP 图像特征提取，逐张处理，损坏图片自动跳过。"""

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
        提取特征向量。
        进度范围：0 % → 40 %（后续步骤占剩余 60 %）。
        """
        feats: List[np.ndarray] = []
        valid: List[str] = []
        total = len(paths)

        for idx, p in enumerate(paths):
            if on_status:
                on_status(f"提取特征 [{idx + 1}/{total}]：{Path(p).name}")
            if on_progress:
                on_progress((idx + 1) / total * 40.0)

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

def kmeans_cluster(
    features: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """L2 归一化后 KMeans，返回 (labels, cluster_centers)。"""
    normed = normalize(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(normed)
    return labels, km.cluster_centers_


def get_representative_images(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    paths: List[str],
) -> Dict[int, str]:
    """每个簇中，选取距离聚类中心最近的图片作为代表图。"""
    normed = normalize(features)
    reps: Dict[int, str] = {}
    for cid in range(len(centers)):
        mask = labels == cid
        if not mask.any():
            continue
        cluster_feats = normed[mask]
        cluster_paths = [p for p, m in zip(paths, mask) if m]
        dists = np.linalg.norm(cluster_feats - centers[cid], axis=1)
        reps[cid] = cluster_paths[int(np.argmin(dists))]
    return reps


# ────────────────────────────────────────────
# GPT-4o 类目命名
# ────────────────────────────────────────────

def _image_to_b64(path: str) -> Tuple[str, str]:
    """
    将图片编码为 base64，返回 (data, mime_type)。
    BMP / TIFF 等 OpenAI API 不支持的格式自动转换为 JPEG。
    """
    suffix = Path(path).suffix.lower().lstrip(".")
    openai_supported = {"jpeg", "jpg", "png", "gif", "webp"}

    img = Image.open(path).convert("RGB")

    if suffix not in openai_supported:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return data, mime


def name_clusters_with_gpt(
    rep_images: Dict[int, str],
    api_key: str,
    on_status: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> Dict[int, str]:
    """
    调用 GPT-4o 为每个簇命名。
    任何单次 API 调用失败时自动降级为 group_XX，不中断流程。
    进度范围：42 % → 70 %。
    """
    client = OpenAI(api_key=api_key)
    names: Dict[int, str] = {}
    total = len(rep_images)

    for idx, (cid, img_path) in enumerate(sorted(rep_images.items())):
        if on_status:
            on_status(f"GPT-4o 识别类目 [{idx + 1}/{total}]…")
        if on_progress:
            on_progress(42.0 + (idx + 1) / total * 28.0)

        try:
            b64, mime = _image_to_b64(img_path)
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": "low",   # low 档位费用最低
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "这是一张商品图片。"
                                    "请用 2～6 个中文字概括这类商品的名称（例如：运动鞋、连衣裙、手提包）。"
                                    "只输出商品名称，不要标点符号，不要解释。"
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=20,
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            name = _sanitize_name(raw) or f"group_{cid + 1:02d}"
        except Exception as exc:
            name = f"group_{cid + 1:02d}"
            print(f"[GPT 命名失败] 簇 {cid + 1}: {exc}", flush=True)

        names[cid] = name
        if on_log:
            on_log(f"  簇 {cid + 1:02d} → {name}  （代表图：{Path(img_path).name}）")

    return _deduplicate_names(names)


def _sanitize_name(name: str) -> str:
    """去除 Windows 文件名非法字符，限制长度。"""
    name = _INVALID_CHARS.sub("", name).strip(". ")
    return name[:40]


def _deduplicate_names(names: Dict[int, str]) -> Dict[int, str]:
    """若多个簇识别为同一名称，追加 _2 / _3 … 后缀。"""
    count: Dict[str, int] = {}
    result: Dict[int, str] = {}
    for cid, name in sorted(names.items()):
        if name not in count:
            count[name] = 1
            result[cid] = name
        else:
            count[name] += 1
            result[cid] = f"{name}_{count[name]}"
    return result


# ────────────────────────────────────────────
# 文件复制
# ────────────────────────────────────────────

def copy_to_groups(
    root_dir: str,
    paths: List[str],
    labels: np.ndarray,
    folder_names: Dict[int, str],
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> None:
    """
    按 folder_names 在 root_dir 下建立子文件夹并复制图片。
    进度范围：70 % → 100 %。
    """
    root = Path(root_dir)
    for name in folder_names.values():
        (root / name).mkdir(exist_ok=True)

    total = len(paths)
    for i, (src, lbl) in enumerate(zip(paths, labels)):
        fname = folder_names.get(int(lbl), f"group_{int(lbl) + 1:02d}")
        dst_dir = root / fname
        dst = dst_dir / Path(src).name
        # 同名文件追加序号，避免覆盖
        if dst.exists():
            stem, sfx = Path(src).stem, Path(src).suffix
            dst = dst_dir / f"{stem}_{i}{sfx}"
        shutil.copy2(src, dst)

        if on_progress:
            on_progress(70.0 + (i + 1) / total * 30.0)
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
        self.root.title("商品图片自动分类工具  v2.0  (GPT-4o 智能命名)")
        W, H = 740, 650
        self.root.geometry(f"{W}x{H}")
        self.root.resizable(True, True)
        self.root.minsize(640, 580)
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

        ttk.Label(outer, text="商品图片自动分类工具",
                  font=("微软雅黑", 16, "bold")).pack(pady=(0, 14))

        # ── 文件夹 ─────────────────────────
        frm_dir = ttk.LabelFrame(outer, text=" 图片文件夹 ", padding=10)
        frm_dir.pack(fill=tk.X, padx=4, pady=(0, 8))
        self.var_folder = tk.StringVar()
        ttk.Entry(frm_dir, textvariable=self.var_folder,
                  font=("微软雅黑", 9)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(frm_dir, text="浏览…", command=self._browse).pack(side=tk.RIGHT)

        # ── API Key ────────────────────────
        frm_api = ttk.LabelFrame(
            outer, text=" OpenAI API Key（GPT-4o 自动识别类目名称）", padding=10)
        frm_api.pack(fill=tk.X, padx=4, pady=(0, 8))

        key_row = ttk.Frame(frm_api)
        key_row.pack(fill=tk.X)
        self.var_key = tk.StringVar()
        self.entry_key = ttk.Entry(key_row, textvariable=self.var_key,
                                   font=("Consolas", 9), show="*")
        self.entry_key.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.var_show = tk.BooleanVar(value=False)
        ttk.Checkbutton(key_row, text="显示", variable=self.var_show,
                        command=self._toggle_key_visibility).pack(side=tk.RIGHT)

        ttk.Label(frm_api,
                  text="留空则跳过 GPT-4o 命名，使用 group_01 / group_02… 编号",
                  foreground="gray", font=("微软雅黑", 8)).pack(anchor=tk.W, pady=(5, 0))

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
            row, text="自动推断（图片总数 ÷ 10，至少 2 组）",
            variable=self.var_auto,
            command=self._toggle_auto,
        ).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(frm_cfg,
                  text="建议：100 张图片设 10 组；300 张设 20 组；500 张设 30～50 组",
                  foreground="gray", font=("微软雅黑", 8)).pack(anchor=tk.W, pady=(6, 0))

        # ── 进度 ────────────────────────────
        frm_prog = ttk.LabelFrame(outer, text=" 进度 ", padding=10)
        frm_prog.pack(fill=tk.X, padx=4, pady=(0, 8))
        self.var_prog = tk.DoubleVar()
        ttk.Progressbar(frm_prog, variable=self.var_prog,
                        maximum=100).pack(fill=tk.X, pady=(0, 5))
        self.var_status = tk.StringVar(value="就绪 — 请选择文件夹后点击「开始分类」")
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
        sb = ttk.Scrollbar(frm_log, orient=tk.VERTICAL, command=self.log_box.yview)
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
        self.spin_k.configure(
            state=tk.DISABLED if self.var_auto.get() else tk.NORMAL)

    def _toggle_key_visibility(self) -> None:
        self.entry_key.configure(show="" if self.var_show.get() else "*")

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
                    "支持格式：JPG / PNG / BMP / GIF / WEBP / TIFF",
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
                self._log("加载 CLIP 模型（首次运行需下载约 600 MB，请耐心等待）…")
            self.extractor.load(on_status=self._set_status)
            device_label = "GPU (CUDA)" if self.extractor.device == "cuda" else "CPU"
            self._log(f"CLIP 模型就绪，推理设备：{device_label}")
            self._set_progress(2)

            # ④ 提取图片特征（进度 0→40%）
            self._log("提取图片视觉特征…")
            feats, valid_paths = self.extractor.extract(
                paths, on_status=self._set_status, on_progress=self._set_progress)

            skipped = len(paths) - len(valid_paths)
            if skipped:
                self._log(f"⚠  跳过损坏 / 无法读取图片 {skipped} 张（详见控制台）")
            self._log(f"特征提取完成：有效 {len(valid_paths)} 张")

            if not valid_paths:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", "没有可处理的有效图片！"))
                return
            k = min(k, len(valid_paths))

            # ⑤ KMeans 聚类（进度 40→42%）
            self._set_status("KMeans 聚类分析中…")
            self._log(f"KMeans 聚类（k={k}）…")
            labels, centers = kmeans_cluster(feats, k)
            unique, counts = np.unique(labels, return_counts=True)
            self._log("各簇图片数：" + "  ".join(
                f"{int(l)+1:02d}:{c}" for l, c in zip(unique, counts)))
            self._set_progress(42)

            # ⑥ 找各簇代表图
            rep_images = get_representative_images(feats, labels, centers, valid_paths)

            # ⑦ GPT-4o 命名（进度 42→70%）或 fallback
            api_key = self.var_key.get().strip()
            if api_key:
                self._log("调用 GPT-4o 识别各簇类目名称…")
                folder_names = name_clusters_with_gpt(
                    rep_images, api_key,
                    on_status=self._set_status,
                    on_progress=self._set_progress,
                    on_log=self._log,
                )
                self._log("类目识别完成")
            else:
                self._log("未填写 API Key → 使用默认编号命名（group_XX）")
                folder_names = {cid: f"group_{cid + 1:02d}" for cid in range(k)}
                self._set_progress(70)

            # ⑧ 复制图片（进度 70→100%）
            self._set_status("复制图片到分类文件夹…")
            self._log("复制图片文件…")
            copy_to_groups(
                folder, valid_paths, labels, folder_names,
                on_progress=self._set_progress,
                on_log=self._log,
            )

            # ⑨ 完成
            self._set_progress(100)
            self._set_status(f"✅ 完成！{len(valid_paths)} 张 → {k} 个分组")
            names_str = "\n".join(
                f"  • {n}" for n in sorted(folder_names.values()))
            self._log("─" * 52)
            self._log("✅ 分类完成！")
            self._log(f"   处理：{len(valid_paths)} 张 / 跳过：{skipped} 张 / 分组：{k} 个")
            self._log(f"   分组名称：\n{names_str}")

            self.root.after(0, lambda: messagebox.showinfo(
                "分类完成",
                f"处理完成！\n\n"
                f"成功处理：{len(valid_paths)} 张\n"
                f"跳过（损坏）：{skipped} 张\n"
                f"创建分组：{k} 个\n\n"
                f"分组名称：\n{names_str}\n\n"
                f"结果已保存至所选文件夹",
            ))

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self._log(f"❌ 错误：\n{tb}")
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
