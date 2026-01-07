"""
测试1D特征提取 - 只提取前5个样本
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def test_protein_extraction():
    """测试蛋白质特征提取"""
    print("=" * 60)
    print("测试蛋白质1D特征提取")
    print("=" * 60)
    
    # 读取CSV
    csv_path = "../project1-main/id_seq.csv"
    df = pd.read_csv(csv_path)
    df_test = df.head(5)  # 只取前5个
    
    print(f"测试数据: {len(df_test)} 个蛋白质")
    
    # 导入ESM
    try:
        import torch
        import esm
        print("✓ ESM模块已安装")
    except ImportError as e:
        print(f"✗ ESM导入失败: {e}")
        return
    
    # 加载模型
    print("\n加载ESM2模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print("✓ ESM2模型加载成功")
    
    # 提取特征
    print("\n开始提取特征...")
    protein_features = {}
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        protein_id = row['Pdbid']
        sequence = row['Acid_Sequence']
        
        print(f"\n处理: {protein_id}")
        print(f"  序列长度: {len(sequence)}")
        
        try:
            data = [(protein_id, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
            
            embedding = token_representations[0, 1:len(sequence)+1].mean(0).cpu().numpy()
            protein_features[protein_id] = embedding
            
            print(f"  ✓ 特征维度: {embedding.shape}")
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print(f"\n成功提取 {len(protein_features)} 个蛋白质特征")
    
    # 保存
    output_dir = "../features_extraction/features_1d"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "protein_features_1d_test.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(protein_features, f)
    
    print(f"✓ 测试特征已保存: {output_path}")
    return protein_features


def test_drug_extraction():
    """测试药物特征提取"""
    print("\n" + "=" * 60)
    print("测试药物1D特征提取")
    print("=" * 60)
    
    # 读取CSV
    csv_path = "../project1-main/id_smile.csv"
    df = pd.read_csv(csv_path)
    df_test = df.head(10)  # 取前10个
    
    print(f"测试数据: {len(df_test)} 个药物")
    
    # 导入库
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        print("✓ RDKit已安装")
    except ImportError as e:
        print(f"✗ RDKit导入失败: {e}")
        return
    
    # 使用分子指纹
    print("\n使用Morgan指纹提取特征...")
    drug_features = {}
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        drug_id = row['pdbid']
        smiles = row['SMILES']
        
        print(f"\n处理: {drug_id}")
        print(f"  SMILES: {smiles[:50]}...")
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  ✗ 无效SMILES")
                continue
            
            # Morgan指纹
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.float32)
            for i in range(2048):
                arr[i] = fp[i]
            
            drug_features[str(drug_id)] = arr
            print(f"  ✓ 特征维度: {arr.shape}")
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print(f"\n成功提取 {len(drug_features)} 个药物特征")
    
    # 保存
    output_dir = "../features_extraction/features_1d"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "drug_features_1d_test.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(drug_features, f)
    
    print(f"✓ 测试特征已保存: {output_path}")
    return drug_features


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 测试蛋白质
    protein_feat = test_protein_extraction()
    
    # 测试药物
    drug_feat = test_drug_extraction()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
