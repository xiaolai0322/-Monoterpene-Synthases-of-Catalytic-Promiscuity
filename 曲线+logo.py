import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import logomaker
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心参数与路径设置
# ==========================================
DATA_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\SHAP_Results"
OUTPUT_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "Fig_Combined_SHAP_Trajectory_and_Logos.png")

REF_MAPPING_CSV = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析\3D_GAT_Mapped_Scores\3D_GAT_Mapped_2_Linear_Only.csv"

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

CLASS_ORDER = [
    'Linear', 'Monocyclic', 'Bicyclic',
    'Linear+Monocyclic', 'Linear+Bicyclic',
    'Monocyclic+Bicyclic', 'Linear+Monocyclic+Bicyclic'
]

CLASS_TO_FILE = {v: k for k, v in FILE_MAPPING.items()}
PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

HOTSPOTS_CONFIG = {
    "Linear": [(23, 33), (115, 126), (405, 415), (575, 595), (665, 680)],
    "Monocyclic": [(21, 37), (52, 65), (320, 335), (520, 540), (686, 697)],
    "Bicyclic": [(20, 40), (70, 90), (115, 140), (320, 340), (440, 460)],
    "Linear+Monocyclic": [(20, 40), (70, 90), (130, 140), (350, 370), (530, 550)],
    "Linear+Bicyclic": [(20, 35), (70, 90), (120, 140), (540, 570), (665, 680)],
    "Monocyclic+Bicyclic": [(115, 137), (310, 330), (405, 415), (530, 550), (630, 650)],
    "Linear+Monocyclic+Bicyclic": [(20, 38), (170, 183), (320, 340), (580, 595), (665, 680)]
}

BASELINE_SETTING = 2.3

# ==========================================
# 2. 读取坐标映射与 Logo 计算函数
# ==========================================
print("🗺️ 正在读取基准坐标映射关系...")
ref_df = pd.read_csv(REF_MAPPING_CSV)
x_col_ref = 'Reference_Position' if 'Reference_Position' in ref_df.columns else 'MSA_Column'
msa_to_real_x = dict(zip(ref_df['MSA_Column'], ref_df[x_col_ref]))


def calculate_shap_logo_matrix(df):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_cols = [c for c in df.columns if str(c).endswith('_AA') and not c.startswith('Mean')]
    shap_cols = [c.replace('_AA', '_SHAP') for c in aa_cols]

    matrix = {aa: np.zeros(len(df)) for aa in aa_list}
    valid_ratios = np.zeros(len(df))

    for i in range(len(df)):
        row = df.iloc[i]
        valid_count = sum([1 for c in aa_cols if pd.notna(row[c]) and row[c] != '-'])
        valid_ratios[i] = valid_count / len(aa_cols) if len(aa_cols) > 0 else 0

        aa_shap_sum = {aa: 0.0 for aa in aa_list}
        for aa_col, shap_col in zip(aa_cols, shap_cols):
            aa = row[aa_col]
            shap_val = row[shap_col]
            if pd.notna(aa) and aa in aa_list and pd.notna(shap_val):
                aa_shap_sum[aa] += abs(float(shap_val))

        num_seqs = len(aa_cols)
        for aa in aa_list:
            matrix[aa][i] = aa_shap_sum[aa] / num_seqs

    return pd.DataFrame(matrix, index=df['MSA_Column'].values), valid_ratios


# ==========================================
# 3. 初始化联合图表排版 (GridSpec)
# ==========================================
max_logo_cols = 5
num_classes = len(CLASS_ORDER)

# 高度比例分配：保持较大的 hspace 给虚线留出空间
height_ratios = []
for _ in range(num_classes):
    height_ratios.extend([1.8, 1.0])

fig = plt.figure(figsize=(16, 26), dpi=300)
gs = gridspec.GridSpec(num_classes * 2, max_logo_cols, height_ratios=height_ratios, hspace=0.65, wspace=0.3)

print(f"🌊 开始生成 轨迹曲线+峰值Logo 联合排版大图...")

global_legend_label = ""
p_max = 0.006

for i, class_name in enumerate(CLASS_ORDER):
    filename = CLASS_TO_FILE.get(class_name)
    file_path = os.path.join(DATA_DIR, filename)
    current_color = COLOR_DICT[class_name]

    row_line = 2 * i
    row_logo = 2 * i + 1

    # ===== A. 绘制上方的轨迹曲线图 =====
    ax_line = fig.add_subplot(gs[row_line, :])

    if not os.path.exists(file_path):
        ax_line.text(0.5, 0.5, f"Missing: {filename}", transform=ax_line.transAxes, ha='center')
        continue

    df = pd.read_csv(file_path)
    if 'MSA_Column' not in df.columns:
        df['MSA_Column'] = df.index + 1

    df_line = df.copy()
    df_line['Real_Pos'] = df_line['MSA_Column'].map(msa_to_real_x)
    df_line = df_line.dropna(subset=['Real_Pos']).sort_values(by='Real_Pos')
    x_positions = df_line['Real_Pos'].values

    indiv_cols = [c for c in df_line.columns if c.endswith('_SHAP') and not c.startswith('Mean')]
    if 'Mean_Abs_SHAP' in df_line.columns:
        mean_raw = df_line['Mean_Abs_SHAP'].values.copy()
    else:
        mean_raw = df_line[indiv_cols].mean(axis=1).values.copy()

    valid_ratio_line = df_line[indiv_cols].notna().sum(axis=1) / len(indiv_cols)
    mean_raw[valid_ratio_line < 0.15] = np.nan

    if str(BASELINE_SETTING).lower() == 'mean':
        dynamic_baseline = np.nanmean(mean_raw)
        global_legend_label = 'Importance Threshold (Mean)'
    elif str(BASELINE_SETTING).lower() == 'median':
        dynamic_baseline = np.nanmedian(mean_raw)
        global_legend_label = 'Importance Threshold (Median)'
    else:
        try:
            val = float(BASELINE_SETTING)
            dynamic_baseline = val * 1e-3
            global_legend_label = f'Importance Threshold ({val})'
        except ValueError:
            dynamic_baseline = 0.002
            global_legend_label = 'Importance Threshold (2.0)'

    mean_filled = np.nan_to_num(mean_raw, nan=0.0)
    mean_clipped = np.clip(mean_filled, 0, p_max)

    smooth_base = gaussian_filter1d(mean_clipped, sigma=6.0)
    sharp_peaks = gaussian_filter1d(mean_clipped, sigma=0.5)

    p_ref = np.percentile(mean_filled, 99)
    peaks, _ = find_peaks(sharp_peaks, distance=30, prominence=p_ref * 0.08)
    if len(peaks) > 5:
        peaks = peaks[np.argsort(sharp_peaks[peaks])[-5:]]

    mask = np.zeros_like(sharp_peaks)
    for p in peaks:
        left, right = max(0, p - 12), min(len(mask), p + 12)
        mask[left:right] = 1.0
    mask = gaussian_filter1d(mask, sigma=3.0)

    mean_final = mask * sharp_peaks + (1 - mask) * smooth_base
    mean_final[np.isnan(mean_raw)] = np.nan

    for col in indiv_cols:
        y_raw = df_line[col].values.copy()
        y_raw[valid_ratio_line < 0.15] = np.nan
        y_f = np.nan_to_num(y_raw, nan=0.0)
        y_clipped = np.clip(y_f, 0, p_max)

        y_base = gaussian_filter1d(y_clipped, sigma=6.0)
        y_sharp = gaussian_filter1d(y_clipped, sigma=1.0)
        y_final = mask * y_sharp + (1 - mask) * y_base
        y_final[np.isnan(y_raw)] = np.nan

        ax_line.plot(x_positions, y_final, color=current_color, alpha=0.25, linewidth=0.8, zorder=1)

    ax_line.plot(x_positions, mean_final, color=current_color, linewidth=1.5, zorder=3)
    ax_line.axhline(y=dynamic_baseline, color='black', linestyle='--', linewidth=1.5, zorder=2, alpha=0.8)

    ax_line.text(0.01, 0.95, PANEL_LABELS[i], transform=ax_line.transAxes,
                 fontsize=24, fontweight='bold', va='top', ha='left')
    ax_line.text(0.99, 0.95, class_name, transform=ax_line.transAxes,
                 fontsize=16, fontweight='bold', color=current_color, va='top', ha='right')

    ax_line.set_ylim(0, p_max)
    ax_line.set_yticks([0, 0.002, 0.004, 0.006])
    ax_line.set_yticklabels(['0', '2', '4', '6'])
    ax_line.set_xlim(left=x_positions.min(), right=x_positions.max())

    ax_line.xaxis.set_minor_locator(MultipleLocator(20))
    ax_line.tick_params(axis='x', which='minor', bottom=True, length=4, color='gray', direction='out')
    ax_line.tick_params(axis='both', which='major', labelsize=12)
    plt.setp(ax_line.get_xticklabels(), fontweight='bold')
    plt.setp(ax_line.get_yticklabels(), fontweight='bold')

    ax_line.spines['top'].set_visible(False)
    ax_line.spines['right'].set_visible(False)

    # ===== B. 绘制下方的热点 Logo 图，并添加关联虚线 =====
    hotspots = HOTSPOTS_CONFIG.get(class_name, [])

    raw_df_logo, v_ratios_logo = calculate_shap_logo_matrix(df)
    raw_df_logo.loc[v_ratios_logo < 0.15, :] = 0.0

    aligned_df = raw_df_logo.copy()
    valid_msa_indices = [idx for idx in aligned_df.index if idx in msa_to_real_x]
    aligned_df = aligned_df.loc[valid_msa_indices]
    aligned_df.index = [msa_to_real_x[idx] for idx in aligned_df.index]

    for j in range(max_logo_cols):
        ax_logo = fig.add_subplot(gs[row_logo, j])

        if j >= len(hotspots):
            ax_logo.axis('off')
            continue

        start_pos, end_pos = hotspots[j]
        chunk_df = aligned_df[(aligned_df.index >= start_pos) & (aligned_df.index <= end_pos)]

        if not chunk_df.empty and (chunk_df.sum(axis=1) > 1e-6).any():
            logo = logomaker.Logo(chunk_df, ax=ax_logo, color_scheme='skylign_protein', width=0.9, vpad=0.0)

            logo.style_spines(visible=False)
            logo.style_spines(spines=['bottom'], visible=True, linewidth=1.5)

            ax_logo.set_xlim(start_pos - 0.5, end_pos + 0.5)
            ax_logo.set_ylim(0, p_max)
            ax_logo.get_yaxis().set_visible(False)

            # ==============================
            # 新的刻度设计：只标首、中、尾，清晰明了
            # ==============================
            custom_ticks = np.unique([start_pos, (start_pos + end_pos) // 2, end_pos])
            ax_logo.set_xticks(custom_ticks)
            ax_logo.set_xticklabels([str(t) for t in custom_ticks], fontsize=11, fontweight='bold', color='#333333')

            # 美化刻度线
            ax_logo.tick_params(axis='x', which='major', bottom=True, length=5, width=1.5, direction='out',
                                color='black')
            ax_logo.tick_params(axis='x', which='minor', bottom=False)  # 关掉多余的小刻度防止杂乱

            # ==============================
            # 视觉映射设计：高亮与跨图连线
            # ==============================
            # 1. 在上方曲线图中增加阴影，精准高亮区间 (去除了文字)
            ax_line.axvspan(start_pos - 0.5, end_pos + 0.5, color='gray', alpha=0.15, lw=0, zorder=0)

            # 2. 画左侧连接虚线
            con_left = ConnectionPatch(xyA=(start_pos - 0.5, 0), xyB=(start_pos - 0.5, p_max),
                                       coordsA="data", coordsB="data",
                                       axesA=ax_line, axesB=ax_logo,
                                       color="gray", linestyle="--", linewidth=1.2, alpha=0.5, zorder=0)
            con_left.set_clip_on(False)
            ax_line.add_artist(con_left)

            # 3. 画右侧连接虚线
            con_right = ConnectionPatch(xyA=(end_pos + 0.5, 0), xyB=(end_pos + 0.5, p_max),
                                        coordsA="data", coordsB="data",
                                        axesA=ax_line, axesB=ax_logo,
                                        color="gray", linestyle="--", linewidth=1.2, alpha=0.5, zorder=0)
            con_right.set_clip_on(False)
            ax_line.add_artist(con_right)

        else:
            ax_logo.axis('off')

# ==========================================
# 4. 全局图例、标题与收尾
# ==========================================
fig.text(0.04, 0.5, r"RF-SHAP Score ($\times 10^{-3}$)", va='center', rotation='vertical', fontsize=20,
         fontweight='bold')

fig.text(0.5, 0.04, "Alignment Position (Structural Coordinates)", ha='center', fontsize=18, fontweight='bold')

plt.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.08)

legend_elements = [
    Line2D([0], [0], color='#444444', alpha=0.4, lw=1.5, label='Individual Enzyme SHAP Trajectory'),
    Line2D([0], [0], color='#444444', alpha=0.9, lw=2.5, label='Consensus Feature Importance Profile'),
    Line2D([0], [0], color='black', linestyle='--', lw=1.8, label=global_legend_label)
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=14, frameon=False)

plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
print(f"✅ 完美融合！刻度已移至Logo X轴的底端，排版图已保存在:\n{os.path.abspath(OUTPUT_IMAGE)}")

try:
    os.startfile(OUTPUT_IMAGE)
except:
    pass