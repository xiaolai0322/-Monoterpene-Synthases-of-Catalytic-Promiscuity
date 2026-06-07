import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心参数与路径设置
# ==========================================
# 您的 CSV 数据存放目录
DATA_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\SHAP_Results"
# 输出目录
OUTPUT_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "Fig_RF_SHAP_7Rows_DynamicBaseline.png")

COLOR_DICT = {
    "Linear": "#d73027", "Monocyclic": "#4575b4", "Bicyclic": "#1a9850",
    "Linear+Monocyclic": "#762a83", "Linear+Bicyclic": "#f46d43",
    "Monocyclic+Bicyclic": "#00ced1", "Linear+Monocyclic+Bicyclic": "#8c510a"
}

FILE_MAPPING = {
    "1D_SHAP_Group_2_Linear_Only.csv": "Linear",
    "1D_SHAP_Group_3_Monocyclic_Only.csv": "Monocyclic",
    "1D_SHAP_Group_1_Bicyclic_Only.csv": "Bicyclic",
    "1D_SHAP_Group_6_Linear_Monocyclic.csv": "Linear+Monocyclic",
    "1D_SHAP_Group_4_Bicyclic_Linear.csv": "Linear+Bicyclic",
    "1D_SHAP_Group_5_Bicyclic_Monocyclic.csv": "Monocyclic+Bicyclic",
    "1D_SHAP_Group_7_All_Three.csv": "Linear+Monocyclic+Bicyclic"
}

PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# ==========================================
# 🎯 核心配置：虚线基准线随心调节
# ==========================================
# 您可以在这里随意修改：
# - 填写 'mean'   : 自动计算各类的平均值作为虚线
# - 填写 'median' : 自动计算各类的中位数作为虚线
# - 填写 具体数字 : (例如 6, 4.5, 3)，将虚线强制固定在 Y 轴的特定刻度上
BASELINE_SETTING =2.3

# ==========================================
# 2. 初始化 7行1列 纵向画布
# ==========================================
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(14, 16), dpi=300, sharex=True)

print(f"🌊 正在生成最终版 RF-SHAP 波浪图谱 (当前虚线基准为: {BASELINE_SETTING})...")

# 用于存储图例标签
legend_label = ""

for idx, (filename, class_name) in enumerate(FILE_MAPPING.items()):
    ax = axes[idx]
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        ax.text(0.5, 0.5, f"Missing: {filename}", transform=ax.transAxes, ha='center')
        continue

    # 读取并智能识别列名
    df = pd.read_csv(file_path)
    x_col = 'MSA_Column' if 'MSA_Column' in df.columns else 'Reference_Position'
    df = df.sort_values(by=x_col)

    x_positions = df[x_col].values
    current_color = COLOR_DICT[class_name]

    # 提取个体 SHAP 列与均值列
    indiv_cols = [c for c in df.columns if c.endswith('_SHAP') and not c.startswith('Mean')]
    if 'Mean_Abs_SHAP' in df.columns:
        mean_raw = df['Mean_Abs_SHAP'].values.copy()
    else:
        mean_raw = df[indiv_cols].mean(axis=1).values.copy()

    # 孤岛/罕见插入断层修复：有效序列不到 15% 强行打断
    valid_ratio = df[indiv_cols].notna().sum(axis=1) / len(indiv_cols)
    mean_raw[valid_ratio < 0.15] = np.nan

    # 🎯【全新升级：动态识别基准线数值】
    if str(BASELINE_SETTING).lower() == 'mean':
        dynamic_baseline = np.nanmean(mean_raw)
        legend_label = 'Importance Threshold (Mean)'
    elif str(BASELINE_SETTING).lower() == 'median':
        dynamic_baseline = np.nanmedian(mean_raw)
        legend_label = 'Importance Threshold (Median)'
    else:
        # 如果填的是数字，强行将其转为 10 的 -3 次方量级
        try:
            val = float(BASELINE_SETTING)
            dynamic_baseline = val * 1e-3
            legend_label = f'Importance Threshold ({val})'
        except ValueError:
            # 容错机制
            dynamic_baseline = 0.002
            legend_label = 'Importance Threshold (2.0)'

    # --- 双分辨率掩码融合平滑 ---
    mean_filled = np.nan_to_num(mean_raw, nan=0.0)

    # 截断极值防爆表
    p_max = 0.006
    mean_clipped = np.clip(mean_filled, 0, p_max)

    smooth_base = gaussian_filter1d(mean_clipped, sigma=6.0)
    sharp_peaks = gaussian_filter1d(mean_clipped, sigma=0.5)

    # 寻峰并构建掩码
    p_ref = np.percentile(mean_filled, 99)
    peaks, _ = find_peaks(sharp_peaks, distance=30, prominence=p_ref * 0.08)

    # 限制最多显示 5 个最高峰
    if len(peaks) > 5:
        peaks = peaks[np.argsort(sharp_peaks[peaks])[-5:]]

    mask = np.zeros_like(sharp_peaks)
    for p in peaks:
        left, right = max(0, p - 12), min(len(mask), p + 12)
        mask[left:right] = 1.0
    mask = gaussian_filter1d(mask, sigma=3.0)

    # 融合生成主曲线
    mean_final = mask * sharp_peaks + (1 - mask) * smooth_base
    mean_final[np.isnan(mean_raw)] = np.nan

    # --- 绘图 ---
    for col in indiv_cols:
        y_raw = df[col].values.copy()
        y_raw[valid_ratio < 0.15] = np.nan
        y_f = np.nan_to_num(y_raw, nan=0.0)
        y_clipped = np.clip(y_f, 0, p_max)

        y_base = gaussian_filter1d(y_clipped, sigma=6.0)
        y_sharp = gaussian_filter1d(y_clipped, sigma=1.0)
        y_final = mask * y_sharp + (1 - mask) * y_base
        y_final[np.isnan(y_raw)] = np.nan

        ax.plot(x_positions, y_final, color=current_color, alpha=0.25, linewidth=0.8, zorder=1)

    # 绘制均值主线 (去除阴影)
    ax.plot(x_positions, mean_final, color=current_color, linewidth=1.5, zorder=3)

    # 🎯【绘制动态虚线】
    ax.axhline(y=dynamic_baseline, color='black', linestyle='--', linewidth=1.5, zorder=2, alpha=0.8)

    # --- 图表格式化 ---
    ax.text(0.01, 0.95, PANEL_LABELS[idx], transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='top', ha='left')

    ax.text(0.99, 0.95, class_name, transform=ax.transAxes,
            fontsize=16, fontweight='bold', color=current_color, va='top', ha='right')

    # 强制锁定 Y 轴范围为 0 到 0.006
    ax.set_ylim(0, 0.006)
    ax.set_yticks([0, 0.002, 0.004, 0.006])
    ax.set_yticklabels(['0', '2', '4', '6'])

    ax.set_xlim(left=x_positions.min(), right=x_positions.max())

    # X 轴次要刻度（无文字竖杠）
    ax.xaxis.set_minor_locator(MultipleLocator(20))
    ax.tick_params(axis='x', which='minor', bottom=True, length=4, color='gray', direction='out')

    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==========================================
# 3. 全局图例、标题与收尾
# ==========================================
# 统一 Y 轴标签
fig.text(0.04, 0.5, r"RF-SHAP Score ($\times 10^{-3}$)", va='center', rotation='vertical', fontsize=18, fontweight='bold')

# 最底部 X 轴标签
axes[-1].set_xlabel("Alignment Position (Structural Coordinates)", fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0.05, 0.08, 1, 1])
plt.subplots_adjust(hspace=0.2)

# 图例构造，自动显示设定的标签名字
legend_elements = [
    Line2D([0], [0], color='#444444', alpha=0.4, lw=1.5, label='Individual Enzyme SHAP Trajectory'),
    Line2D([0], [0], color='#444444', alpha=0.9, lw=2.5, label='Consensus Feature Importance Profile'),
    Line2D([0], [0], color='black', linestyle='--', lw=1.8, label=legend_label)
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, fontsize=14, frameon=False)

# 保存与自动打开
plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
print(f"✅ 绘制成功！7行动态基线版图片已保存在绝对路径:\n{os.path.abspath(OUTPUT_IMAGE)}")

try:
    os.startfile(OUTPUT_IMAGE)
except:
    pass