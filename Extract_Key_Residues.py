import os
import pandas as pd
import numpy as np

# ==========================================
# 1. 核心参数与路径设置 (请确保路径与你本地一致)
# ==========================================
DATA_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\SHAP_Results"
REF_MAPPING_CSV = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析\3D_GAT_Mapped_Scores\3D_GAT_Mapped_2_Linear_Only.csv"

# 输出表格的保存路径
OUTPUT_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Table_Hotspots_Key_AA_Only.csv")

FILE_MAPPING = {
    "Linear": "1D_SHAP_Group_2_Linear_Only.csv",
    "Monocyclic": "1D_SHAP_Group_3_Monocyclic_Only.csv",
    "Bicyclic": "1D_SHAP_Group_1_Bicyclic_Only.csv",
    "Linear+Monocyclic": "1D_SHAP_Group_6_Linear_Monocyclic.csv",
    "Linear+Bicyclic": "1D_SHAP_Group_4_Bicyclic_Linear.csv",
    "Monocyclic+Bicyclic": "1D_SHAP_Group_5_Bicyclic_Monocyclic.csv",
    "Linear+Monocyclic+Bicyclic": "1D_SHAP_Group_7_All_Three.csv"
}

HOTSPOTS_CONFIG = {
    "Linear": [(23, 33), (115, 126), (405, 415), (575, 595), (665, 680)],
    "Monocyclic": [(21, 37), (52, 65), (320, 335), (520, 540), (686, 697)],
    "Bicyclic": [(20, 40), (70, 90), (115, 140), (320, 340), (440, 460)],
    "Linear+Monocyclic": [(20, 40), (70, 90), (130, 140), (350, 370), (530, 550)],
    "Linear+Bicyclic": [(20, 35), (70, 90), (120, 140), (540, 570), (665, 680)],
    "Monocyclic+Bicyclic": [(115, 137), (310, 330), (405, 415), (530, 550), (630, 650)],
    "Linear+Monocyclic+Bicyclic": [(20, 38), (170, 183), (320, 340), (580, 595), (665, 680)]
}

# ==========================================
# 2. 读取坐标映射与 Logo 矩阵计算函数
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
# 3. 提取每段热点位置上的核心氨基酸 (去除分数)
# ==========================================
results = []

for class_name, hotspots in HOTSPOTS_CONFIG.items():
    filename = FILE_MAPPING.get(class_name)
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}")
        continue

    df = pd.read_csv(file_path)
    if 'MSA_Column' not in df.columns:
        df['MSA_Column'] = df.index + 1

    raw_df_logo, v_ratios_logo = calculate_shap_logo_matrix(df)
    raw_df_logo.loc[v_ratios_logo < 0.15, :] = 0.0

    aligned_df = raw_df_logo.copy()
    valid_msa_indices = [idx for idx in aligned_df.index if idx in msa_to_real_x]
    aligned_df = aligned_df.loc[valid_msa_indices]
    aligned_df.index = [msa_to_real_x[idx] for idx in aligned_df.index]

    for i, (start_pos, end_pos) in enumerate(hotspots):
        chunk_df = aligned_df[(aligned_df.index >= start_pos) & (aligned_df.index <= end_pos)]

        if not chunk_df.empty and (chunk_df.sum(axis=1) > 1e-6).any():
            # 1. 锁定该区间内产生最大SHAP值的“位点”
            max_idx = chunk_df.stack().idxmax()
            max_pos = max_idx[0]

            # 2. 提取该位点上所有20种氨基酸的数据
            pos_data = chunk_df.loc[max_pos]

            # 3. 过滤掉无贡献(=0)的氨基酸，并按分数降序排列
            active_aas = pos_data[pos_data > 1e-6].sort_values(ascending=False)

            # 4. 格式化输出：仅提取氨基酸字母，用 " | " 连接
            aa_only_list = active_aas.index.tolist()
            aa_combined_str = " | ".join(aa_only_list)

            results.append({
                "Product_Class": class_name,
                "Peak_Index": f"Peak {i + 1}",
                "Interval (Pos)": f"{start_pos}-{end_pos}",
                "Key_Position": max_pos,
                "Contributing_Amino_Acids": aa_combined_str
            })
        else:
            results.append({
                "Product_Class": class_name,
                "Peak_Index": f"Peak {i + 1}",
                "Interval (Pos)": f"{start_pos}-{end_pos}",
                "Key_Position": "None",
                "Contributing_Amino_Acids": "None"
            })

# ==========================================
# 4. 输出终端表格并保存
# ==========================================
results_df = pd.DataFrame(results)

print("\n========================================================================")
print(" 🎯 各特异性类别 - 峰值热点中核心位点的氨基酸集合 (按重要性排序)")
print("========================================================================")

# 设置 Pandas 打印格式
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.colheader_justify', 'center')

print(results_df.to_string(index=False))

# 保存为 CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ 分析完成！提取的表格已保存至:\n{OUTPUT_CSV}")

try:
    os.startfile(OUTPUT_CSV)
except:
    pass