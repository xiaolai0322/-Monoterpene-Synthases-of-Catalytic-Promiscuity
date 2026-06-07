import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import warnings

# 引入多重定位器，用于精准控制刻度线
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心参数与路径设置
# ==========================================
DATA_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析\3D_GAT_Mapped_Scores"
OUTPUT_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "Fig_GAT_7Rows_Minimalist_Continuous_Shaded_Fixed.png")

COLOR_DICT = {
    "Linear": "#d73027", "Monocyclic": "#4575b4", "Bicyclic": "#1a9850",
    "Linear+Monocyclic": "#762a83", "Linear+Bicyclic": "#f46d43",
    "Monocyclic+Bicyclic": "#00ced1", "Linear+Monocyclic+Bicyclic": "#8c510a"
}

FILE_MAPPING = {
    "3D_GAT_Mapped_2_Linear_Only.csv": "Linear",
    "3D_GAT_Mapped_3_Monocyclic_Only.csv": "Monocyclic",
    "3D_GAT_Mapped_1_Bicyclic_Only.csv": "Bicyclic",
    "3D_GAT_Mapped_6_Linear_Monocyclic.csv": "Linear+Monocyclic",
    "3D_GAT_Mapped_4_Bicyclic_Linear.csv": "Linear+Bicyclic",
    "3D_GAT_Mapped_5_Bicyclic_Monocyclic.csv": "Monocyclic+Bicyclic",
    "3D_GAT_Mapped_7_All_Three.csv": "Linear+Monocyclic+Bicyclic"
}

PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# ==========================================
# 🚨 α-螺旋的区间 (整合为带标签的格式)
# ==========================================
ALPHA_HELICES = [
    ("α1", 82,135),
    ("α2", 167, 195),
    ("α3", 221, 246),
    ("α4", 361, 379),
    ("α5", 401, 426),
    ("α6", 476, 515),
    ("α7", 548, 603),
    ("α8", 621, 653),
    ("α9", 689, 702)
]

# ==========================================
# 2. 初始化 7行1列 画布
# ==========================================
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(14, 16), dpi=300, sharex=True)

print("🌊 正在生成 7 行极致精简版 GAT 图谱 (已修复重叠斑块，连续阴影纯净版)...")

for idx, (filename, class_name) in enumerate(FILE_MAPPING.items()):
    ax = axes[idx]
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        ax.axis('off')
        continue

    df = pd.read_csv(file_path)
    x_col = 'Reference_Position' if 'Reference_Position' in df.columns else 'MSA_Column'
    df = df.sort_values(by=x_col)
    x_positions = df[x_col].values
    current_color = COLOR_DICT[class_name]

    indiv_cols = [c for c in df.columns if c.endswith('_GAT_ATT')]
    if 'Mean_GAT_Attention' in df.columns:
        mean_raw = df['Mean_GAT_Attention'].values.copy()
    else:
        mean_raw = df[indiv_cols].mean(axis=1).values.copy()

    valid_ratio = df[indiv_cols].notna().sum(axis=1) / len(indiv_cols)
    mean_raw[valid_ratio < 0.15] = np.nan

    # --- 数据处理 ---
    mean_filled = np.nan_to_num(mean_raw, nan=0.0)
    mean_clipped = np.clip(mean_filled, 0, 10)
    smooth_base = gaussian_filter1d(mean_clipped, sigma=6.0)
    sharp_peaks = gaussian_filter1d(mean_clipped, sigma=0.5)

    p_ref = np.percentile(mean_filled, 99)
    peaks, _ = find_peaks(sharp_peaks, distance=30, prominence=p_ref * 0.08)
    if len(peaks) > 5: peaks = peaks[np.argsort(sharp_peaks[peaks])[-5:]]
    mask = np.zeros_like(sharp_peaks)
    for p in peaks:
        left, right = max(0, p - 12), min(len(mask), p + 12)
        mask[left:right] = 1.0
    mask = gaussian_filter1d(mask, sigma=3.0)
    mean_final = mask * sharp_peaks + (1 - mask) * smooth_base
    mean_final[np.isnan(mean_raw)] = np.nan

    # --- 绘图 ---
    for col in indiv_cols:
        y_raw = df[col].values.copy()
        y_raw[valid_ratio < 0.15] = np.nan
        y_f = np.nan_to_num(y_raw, nan=0.0)
        y_s_base = gaussian_filter1d(y_f, sigma=6.0)
        y_s_sharp = gaussian_filter1d(y_f, sigma=1.0)
        y_final = mask * y_s_sharp + (1 - mask) * y_s_base
        y_final[np.isnan(y_raw)] = np.nan
        ax.plot(x_positions, y_final, color=current_color, alpha=0.25, linewidth=0.8, zorder=2)

    ax.plot(x_positions, mean_final, color=current_color, linewidth=1.5, zorder=3)

    # ================= 核心修改区域：无斑块连续实色阴影 =================
    for (h_label, h_start, h_end) in ALPHA_HELICES:
        h_width = h_end - h_start

        # 【关键修复】：
        # 使用不透明的浅灰色 (#EAEAEA, alpha=1.0) 代替半透明。
        # 每幅图的阴影只负责覆盖自己（0 到 1.0）以及下方的空白缝隙（-0.65 到 0）。
        # 因为颜色不透明，下方的缝隙拼接处绝对不会变深，完美杜绝"补丁感"。
        if idx == len(FILE_MAPPING) - 1:
            y_min, y_max = 0.0, 1.0  # 最后一行只需覆盖自己
        else:
            y_min, y_max = -0.65, 1.0  # 覆盖自己 + 下方的 hspace (0.6 留点余量防白线)

        ax.axvspan(h_start, h_end, facecolor='#EAEAEA', alpha=1.0, zorder=0, clip_on=False, ymin=y_min, ymax=y_max)

        # 在 C图 (idx == 2) 和 D图之间标注 α 标签
        if idx == 2:
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(
                h_start + h_width / 2, -0.3, h_label,
                transform=trans,
                ha='center', va='center',
                fontsize=14, fontweight='bold', color='#333333', zorder=5, clip_on=False
            )
    # ========================================================================

    # --- 格式化 ---
    ax.text(0.01, 0.95, PANEL_LABELS[idx], transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    ax.text(0.99, 0.95, class_name, transform=ax.transAxes, fontsize=16, fontweight='bold', color=current_color,
            va='top', ha='right')

    # Y轴设置
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 2, 4])
    ax.set_yticklabels(['0', '2', '4'], fontsize=12, fontweight='bold')

    # X轴刻度
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(20))

    # 控制边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    # 控制刻度线与标签
    if idx == len(FILE_MAPPING) - 1:
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        for label in ax.get_xticklabels():
            label.set_fontsize(12)
            label.set_fontweight('bold')
    else:
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=False)

# ==========================================
# 3. 全局标签与布局调整
# ==========================================
fig.text(0.02, 0.5, "GAT Attention Score", va='center', rotation='vertical', fontsize=18, fontweight='bold')

plt.tight_layout(rect=[0.08, 0.05, 1, 1])
plt.subplots_adjust(hspace=0.6)

plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
print(f"✅ 绘制成功！图片已生成:\n{os.path.abspath(OUTPUT_IMAGE)}")

try:
    os.startfile(OUTPUT_IMAGE)
except:
    pass