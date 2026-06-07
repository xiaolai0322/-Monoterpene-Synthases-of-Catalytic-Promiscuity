import os
import pandas as pd
import numpy as np
import shap
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 路径设置 =================
csv_path = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\2.1_processed_multilabel_data.csv"
msa_path = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析\mTPS_aligned_aa.fa"
output_dir = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\SHAP_Results"
reference_pdb_name = "Q93X23"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ================= 2. 数据与七分类定义 =================
print("1. 正在读取CSV并构建7大分类...")
df = pd.read_csv(csv_path)

df['Category'] = df['Bicyclic'].astype(str) + "_" + df['Linear'].astype(str) + "_" + df['Monocyclic'].astype(str)
category_map = {
    '1_0_0': '1_Bicyclic_Only',
    '0_1_0': '2_Linear_Only',
    '0_0_1': '3_Monocyclic_Only',
    '1_1_0': '4_Bicyclic_Linear',
    '1_0_1': '5_Bicyclic_Monocyclic',
    '0_1_1': '6_Linear_Monocyclic',
    '1_1_1': '7_All_Three'
}
df['Class_Name'] = df['Category'].map(category_map)

# ================= 3. 基于 FoldMason MSA 的全长特征提取 =================
print("2. 正在解析结构约束的 MSA 文件，建立全长绝对坐标系...")

msa_dict = {}
for record in SeqIO.parse(msa_path, "fasta"):
    clean_id = record.id.split('.')[0]
    msa_dict[clean_id] = str(record.seq)

if reference_pdb_name not in msa_dict:
    raise ValueError(f"在 MSA 文件中找不到基准酶 {reference_pdb_name}，请检查 Fasta 文件的 ID 格式。")

ref_seq = msa_dict[reference_pdb_name]
alignment_length = len(ref_seq)

msa_col_ids = list(range(1, alignment_length + 1))
ref_res_names = list(ref_seq)

print(f"全局 MSA 坐标系建立完毕，总对齐长度为 {alignment_length} 列 (严格包含所有的 '-' 空位)。")

features_list = []
valid_classes = []
valid_uids = []

for index, row in df.iterrows():
    uid = row['Uniprot_ID']
    if uid in msa_dict:
        tgt_seq = msa_dict[uid]
        aligned_seq = [tgt_seq[col] if tgt_seq[col] != '-' else 'GAP' for col in range(alignment_length)]
        features_list.append(aligned_seq)
        valid_classes.append(row['Class_Name'])
        valid_uids.append(uid)

print(f"提取完成，共计 {len(valid_classes)} 个合法对齐结构进入训练。")

# ================= 4. 模型编码与训练 =================
print("3. 正在编码序列特征并训练 Random Forest 模型...")
feature_cols = [f"Col_{idx}" for idx in msa_col_ids]
feature_df = pd.DataFrame(features_list, columns=feature_cols)

all_aas = pd.unique(feature_df.values.ravel('K'))
le = LabelEncoder()
le.fit(all_aas)

X = feature_df.apply(le.transform)
y = np.array(valid_classes)
valid_uids = np.array(valid_uids)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X, y)

# ================= 5. 计算并严格过滤空位 SHAP (全部取正值) =================
print("4. 正在计算 SHAP，转化为绝对正值，并将空位('-')得分清空...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

for i, cls_name in enumerate(rf.classes_):
    cls_indices = np.where(y == cls_name)[0]
    if len(cls_indices) == 0: continue

    if isinstance(shap_values, list):
        cls_shap = shap_values[i][cls_indices]
    elif len(shap_values.shape) == 3:
        cls_shap = shap_values[cls_indices, :, i]
    else:
        cls_shap = shap_values[cls_indices]

    # 去除上下 1% 的极端离群值抗噪
    p_low = np.percentile(cls_shap, 1)
    p_high = np.percentile(cls_shap, 99)
    cls_shap_clipped = np.clip(cls_shap, p_low, p_high)

    # 建立输出基础字典
    result_data = {
        'MSA_Column': msa_col_ids,
        'Q93X23_Residue': ref_res_names
    }

    cls_uids = valid_uids[cls_indices]
    masked_shaps_for_mean = []

    for j, enzyme_uid in enumerate(cls_uids):
        original_seq = msa_dict[enzyme_uid]
        result_data[f"{enzyme_uid}_AA"] = list(original_seq)

        # 【核心修改 1】: 将个体 SHAP 值强制取绝对值 (转化为全部正值)
        sample_shap = np.abs(cls_shap_clipped[j].copy())

        # 遍历序列，将 '-' 处的得分挖空 (置为 NaN)
        for col_idx in range(alignment_length):
            if original_seq[col_idx] == '-':
                sample_shap[col_idx] = np.nan

        # 写入个体数据
        result_data[f"{enzyme_uid}_SHAP"] = sample_shap
        masked_shaps_for_mean.append(sample_shap)

    masked_shaps_for_mean = np.array(masked_shaps_for_mean)

    # 【核心修改 2】: 计算均值。因为个体数据已经是绝对值了，直接求 nanmean 即可
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_shap = np.nanmean(masked_shaps_for_mean, axis=0)

    # 计算缩放后的注意力权重
    scaler = MinMaxScaler()
    valid_mask = ~np.isnan(mean_shap)
    scaled_shap = np.full_like(mean_shap, np.nan)
    if np.any(valid_mask):
        scaled_shap[valid_mask] = scaler.fit_transform(mean_shap[valid_mask].reshape(-1, 1)).flatten()

    # 将均值写入数据字典 (移除了原来的 Mean_Directional_SHAP，因为现在只有绝对值均值)
    result_data['Mean_Abs_SHAP'] = mean_shap
    result_data['Scaled_Attention_Weight'] = scaled_shap

    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values(by='MSA_Column', ascending=True)

    out_file = os.path.join(output_dir, f"1D_SHAP_Group_{cls_name}.csv")
    result_df.to_csv(out_file, index=False, na_rep='')
    print(f" -> 已成功导出: {out_file} (已将个体与均值转化为纯正值并挖空)")

print("\n🎉 任务圆满完成！全局对齐的正值纯净版 SHAP 表格已生成。")