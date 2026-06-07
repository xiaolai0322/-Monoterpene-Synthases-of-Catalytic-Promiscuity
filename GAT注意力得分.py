import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from torch_geometric.nn import GATConv
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 配置路径 =================
CSV_PATH = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\2.1_processed_multilabel_data.csv"
MSA_PATH = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析\mTPS_aligned_aa.fa"
PT_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.4_enzyme_graphs_pyg"
MODEL_PATH = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\final_gat_basic_model.pth"
OUTPUT_DIR = r"D:\AAA研究生生涯\研二上\酶-产物特异性研究\补充数据\期刊论文数据处理\Deepseek版数据处理\3.6 可解释性分析\3D_GAT_Mapped_Scores"

os.makedirs(OUTPUT_DIR, exist_ok=True)
REFERENCE_PDB_NAME = "Q93X23"


# ================= 2. 定义 GAT 模型 =================
class EnzymeGAT_Basic(nn.Module):
    def __init__(self, in_channels=1280, hidden_channels=64, out_channels=3, heads=4):
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels * 2)
        self.bn_emb = nn.BatchNorm1d(hidden_channels * 2)
        self.conv1 = GATConv(hidden_channels * 2, hidden_channels, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        self.lin1 = nn.Linear((hidden_channels * heads) * 2, 128)
        self.lin2 = nn.Linear(128, out_channels)

    def extract_node_attention(self, x, edge_index):
        x = F.elu(self.bn_emb(self.node_emb(x)))
        x_1, (e1, a1) = self.conv1(x, edge_index, return_attention_weights=True)
        x_2, (e2, a2) = self.conv2(F.elu(self.bn1(x_1)), edge_index, return_attention_weights=True)
        scores = torch.zeros(x.size(0), device=x.device)
        scores.scatter_add_(0, e1[0], a1.mean(dim=-1))
        scores.scatter_add_(0, e2[0], a2.mean(dim=-1))
        return scores.detach().cpu().numpy()


# ================= 3. 主流程 =================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # 3.1 加载模型
    model = EnzymeGAT_Basic().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    model.eval()

    # 3.2 加载分类数据
    df = pd.read_csv(CSV_PATH)
    df['Category'] = df['Bicyclic'].astype(str) + "_" + df['Linear'].astype(str) + "_" + df['Monocyclic'].astype(str)
    category_map = {
        '1_0_0': '1_Bicyclic_Only', '0_1_0': '2_Linear_Only', '0_0_1': '3_Monocyclic_Only',
        '1_1_0': '4_Bicyclic_Linear', '1_0_1': '5_Bicyclic_Monocyclic',
        '0_1_1': '6_Linear_Monocyclic', '1_1_1': '7_All_Three'
    }
    df['Class_Name'] = df['Category'].map(category_map)

    # 3.3 解析结构约束的 MSA (建立完全不删减的全局坐标系)
    print("正在解析 FoldMason MSA 文件建立全长全局坐标系...")
    msa_dict = {}
    for record in SeqIO.parse(MSA_PATH, "fasta"):
        clean_id = record.id.split('.')[0]
        msa_dict[clean_id] = str(record.seq)

    if REFERENCE_PDB_NAME not in msa_dict:
        raise ValueError(f"MSA 文件中找不到基准酶 {REFERENCE_PDB_NAME}！")

    # 获取整个 MSA 的绝对长度（所有序列的长度在此刻都是一样的）
    alignment_length = len(msa_dict[REFERENCE_PDB_NAME])

    # 建立全局横坐标 (1 到 MSA最大长度)
    msa_col_ids = list(range(1, alignment_length + 1))
    # 保留 Q93X23 的完整序列（包含它自己的空位 '-'）
    ref_res_names = list(msa_dict[REFERENCE_PDB_NAME])

    print(f"全局 MSA 坐标系建立完毕，总对齐长度为 {alignment_length} 列 (严格包含所有的 - 空位)。")

    # 3.4 遍历分类，计算图注意力得分并映射
    groups = df.groupby('Class_Name')

    for cls_name, group_df in groups:
        mapped_attentions = []
        valid_uids = []
        dropped_count = 0

        for uid in tqdm(group_df['Uniprot_ID'], desc=f"处理类: {cls_name}"):
            pt_file = os.path.join(PT_DIR, f"{uid}_graph.pt")
            if not os.path.exists(pt_file) or uid not in msa_dict:
                dropped_count += 1
                continue

            try:
                # 1. 获取原生 GAT 得分 (长度等于图中节点数，即真实氨基酸数)
                data = torch.load(pt_file, map_location=device, weights_only=False)
                if not (100 <= data.x.size(0) <= 1000):
                    continue
                scores = model.extract_node_attention(data.x.to(device), data.edge_index.to(device))

                # 2. 获取该蛋白的 MSA 对齐序列
                tgt_msa_seq = msa_dict[uid]

                # 3. 严格映射：遍历 MSA 每一列
                mapped = np.full(alignment_length, np.nan)  # 初始化全部为空 (NaN 在 CSV 中就是空着)
                curr_node_idx = 0

                for col_idx, aa in enumerate(tgt_msa_seq):
                    if aa != '-':  # 如果这列是真实氨基酸
                        if curr_node_idx < len(scores):
                            mapped[col_idx] = scores[curr_node_idx]  # 填入得分
                        curr_node_idx += 1
                    # 如果 aa == '-'，mapped[col_idx] 保持为 np.nan，在表格里绝对空着！

                mapped_attentions.append(mapped)
                valid_uids.append(uid)

            except Exception as e:
                dropped_count += 1
                continue

        # 3.5 结果汇总与导出
        if len(mapped_attentions) > 0:
            att_matrix = np.array(mapped_attentions)

            # 计算每列的群体均值 (np.nanmean 自动忽略空着的 '-')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_scores = np.nanmean(att_matrix, axis=0)

            # 构建基础数据框 (行数 = MSA 的总长度)
            out_data = {
                "MSA_Column": msa_col_ids,
                "Q93X23_Residue": ref_res_names,
                "Mean_GAT_Attention": mean_scores
            }

            # 追加个体蛋白的真实字符序列和得分
            for j, enzyme_uid in enumerate(valid_uids):
                out_data[f"{enzyme_uid}_AA"] = list(msa_dict[enzyme_uid])  # 写入字符: A, G, - 等
                out_data[f"{enzyme_uid}_GAT_ATT"] = att_matrix[j]  # 写入得分: 0.12, 0.45, 空 等

            out_df = pd.DataFrame(out_data)

            out_file = os.path.join(OUTPUT_DIR, f"3D_GAT_Mapped_{cls_name}.csv")
            out_df.to_csv(out_file, index=False)
            print(f"✅ [{cls_name}]: 成功融合 {len(mapped_attentions)} 个结构，丢失 {dropped_count} 个。")
        else:
            print(f"❌ [{cls_name}]: 无可用数据。")


if __name__ == "__main__":
    main()