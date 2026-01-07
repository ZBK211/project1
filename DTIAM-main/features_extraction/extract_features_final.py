"""
完整的1D特征提取流程
- 蛋白质: 从id_seq.csv使用ESM2提取
- 药物: 直接使用project1-main中的smiles_embeddings_merged.npz
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def extract_protein_features_from_csv(csv_path, output_path, test_mode=False):
    """
    从CSV文件提取蛋白质1D特征(ESM2)
    """
    print("=" * 80)
    print("步骤1: 提取蛋白质1D特征 (ESM2模型)")
    print("=" * 80)
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    if test_mode:
        df = df.head(10)
        print(f"[测试模式] 只处理前10个蛋白质")
    
    print(f"蛋白质数量: {len(df)}")
    
    # 导入ESM
    try:
        import torch
        import esm
    except ImportError:
        print("错误: 需要安装ESM库")
        print("运行: pip install fair-esm")
        return None
    
    # 加载模型
    print("\n加载ESM2-650M模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print("✓ 模型加载完成")
    
    # 提取特征
    print(f"\n开始提取...")
    protein_features = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取蛋白质特征"):
        protein_id = row['Pdbid']
        sequence = row['Acid_Sequence']
        
        try:
            data = [(protein_id, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
            
            # 平均池化
            embedding = token_representations[0, 1:len(sequence)+1].mean(0).cpu().numpy()
            protein_features[protein_id] = embedding
            
        except Exception as e:
            print(f"\n警告: {protein_id} 失败 - {e}")
            continue
    
    print(f"\n✓ 成功提取 {len(protein_features)} 个蛋白质特征")
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(protein_features, f)
    print(f"✓ 已保存: {output_path}")
    
    # 显示示例
    if protein_features:
        sample_id = list(protein_features.keys())[0]
        sample_feat = protein_features[sample_id]
        print(f"\n示例 ({sample_id}):")
        print(f"  维度: {sample_feat.shape}")
        print(f"  范围: [{sample_feat.min():.4f}, {sample_feat.max():.4f}]")
    
    return protein_features


def load_drug_features_from_npz(npz_path, output_path):
    """
    从现有的NPZ文件加载药物特征
    """
    print("\n" + "=" * 80)
    print("步骤2: 加载药物1D特征 (使用现有NPZ文件)")
    print("=" * 80)
    
    print(f"NPZ文件: {npz_path}")
    
    # 加载npz
    data = np.load(npz_path, allow_pickle=True)
    print(f"✓ NPZ文件加载成功")
    print(f"  包含的key数量: {len(data.files)}")
    
    # 转换为字典
    drug_features = {}
    for key in data.files:
        embedding = data[key]
        drug_features[key] = embedding
    
    print(f"✓ 药物特征数量: {len(drug_features)}")
    
    # 保存为pkl格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(drug_features, f)
    print(f"✓ 已保存: {output_path}")
    
    # 显示示例
    if drug_features:
        sample_id = list(drug_features.keys())[0]
        sample_feat = drug_features[sample_id]
        print(f"\n示例 ({sample_id}):")
        print(f"  维度: {sample_feat.shape}")
        print(f"  数据类型: {sample_feat.dtype}")
        print(f"  范围: [{sample_feat.min():.4f}, {sample_feat.max():.4f}]")
    
    return drug_features


def verify_features(protein_pkl, drug_pkl):
    """
    验证提取的特征
    """
    print("\n" + "=" * 80)
    print("步骤3: 验证特征")
    print("=" * 80)
    
    # 加载特征
    with open(protein_pkl, 'rb') as f:
        protein_features = pickle.load(f)
    
    with open(drug_pkl, 'rb') as f:
        drug_features = pickle.load(f)
    
    print(f"✓ 蛋白质特征: {len(protein_features)} 个")
    print(f"✓ 药物特征: {len(drug_features)} 个")
    
    # 检查维度
    if protein_features:
        sample_prot = next(iter(protein_features.values()))
        print(f"\n蛋白质embedding维度: {sample_prot.shape}")
    
    if drug_features:
        sample_drug = next(iter(drug_features.values()))
        print(f"药物embedding维度: {sample_drug.shape}")
    
    print("\n" + "=" * 80)
    print("✓ 特征提取完成！可以用于后续任务")
    print("=" * 80)
    
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='提取1D特征')
    parser.add_argument('--test', action='store_true', help='测试模式(只处理前10个)')
    parser.add_argument('--full', action='store_true', help='完整模式(处理所有数据)')
    args = parser.parse_args()
    
    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输入文件
    protein_csv = os.path.join(base_dir, 'project1-main', 'id_seq.csv')
    drug_npz = os.path.join(base_dir, 'project1-main', 'smiles_embeddings_merged.npz')
    
    # 输出文件
    output_dir = os.path.join(base_dir, 'features_extraction', 'features_1d')
    
    if args.full:
        protein_output = os.path.join(output_dir, 'protein_features_1d_full.pkl')
        drug_output = os.path.join(output_dir, 'drug_features_1d_full.pkl')
        test_mode = False
    else:
        protein_output = os.path.join(output_dir, 'protein_features_1d.pkl')
        drug_output = os.path.join(output_dir, 'drug_features_1d.pkl')
        test_mode = args.test if not args.full else False
    
    print("配置:")
    print(f"  蛋白质CSV: {protein_csv}")
    print(f"  药物NPZ: {drug_npz}")
    print(f"  输出目录: {output_dir}")
    print(f"  模式: {'完整' if args.full else ('测试' if test_mode else '默认')}")
    print()
    
    # 检查文件
    if not os.path.exists(protein_csv):
        print(f"错误: 找不到 {protein_csv}")
        return
    if not os.path.exists(drug_npz):
        print(f"错误: 找不到 {drug_npz}")
        return
    
    # 提取蛋白质特征
    protein_features = extract_protein_features_from_csv(protein_csv, protein_output, test_mode)
    if protein_features is None:
        return
    
    # 加载药物特征
    drug_features = load_drug_features_from_npz(drug_npz, drug_output)
    if drug_features is None:
        return
    
    # 验证
    verify_features(protein_output, drug_output)


if __name__ == "__main__":
    main()
