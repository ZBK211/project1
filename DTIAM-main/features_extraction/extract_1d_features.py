"""
1D特征提取模块
从CSV文件提取蛋白质序列和药物SMILES的embedding
使用ESM2模型提取蛋白质特征，使用BerMol提取药物特征
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加BerMol路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code', 'BerMol'))

def extract_protein_features_esm2(csv_path, output_path):
    """
    使用ESM2模型从CSV文件提取蛋白质序列的1D embedding
    
    Args:
        csv_path: id_seq.csv路径
        output_path: 输出特征文件路径
    """
    print("=" * 60)
    print("开始提取蛋白质1D特征...")
    print("=" * 60)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"读取到 {len(df)} 个蛋白质序列")
    print(f"列名: {df.columns.tolist()}")
    print(f"前3个蛋白质ID: {df['Pdbid'].head(3).tolist()}")
    
    # 检查是否已安装ESM
    try:
        import torch
        import esm
        print("✓ ESM模块已安装")
    except ImportError:
        print("✗ 需要安装ESM模块")
        print("请运行: pip install fair-esm")
        return None
    
    # 加载ESM2模型
    print("\n加载ESM2模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 使用ESM2-t33_650M模型
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print("✓ ESM2模型加载完成")
    
    # 提取特征
    protein_features = {}
    batch_size = 1  # 蛋白质序列可能很长，使用batch_size=1
    
    print(f"\n开始提取特征 (batch_size={batch_size})...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取蛋白质特征"):
        protein_id = row['Pdbid']
        sequence = row['Acid_Sequence']
        
        try:
            # 准备数据
            data = [(protein_id, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            # 提取特征
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
            
            # 使用平均池化得到固定长度的向量
            # token_representations: [batch_size, seq_len, hidden_dim]
            # 去掉首尾的特殊token，取平均
            embedding = token_representations[0, 1:len(sequence)+1].mean(0).cpu().numpy()
            
            protein_features[protein_id] = embedding
            
        except Exception as e:
            print(f"\n警告: 蛋白质 {protein_id} 提取失败: {e}")
            continue
    
    print(f"\n成功提取 {len(protein_features)} 个蛋白质特征")
    
    # 保存特征
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(protein_features, f)
    
    print(f"✓ 蛋白质1D特征已保存到: {output_path}")
    
    # 输出特征维度信息
    if protein_features:
        sample_feat = next(iter(protein_features.values()))
        print(f"特征维度: {sample_feat.shape}")
    
    return protein_features


def extract_drug_features_bermol(csv_path, output_path):
    """
    使用BerMol模型从CSV文件提取药物SMILES的1D embedding
    
    Args:
        csv_path: id_smile.csv路径
        output_path: 输出特征文件路径
    """
    print("\n" + "=" * 60)
    print("开始提取药物1D特征...")
    print("=" * 60)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"读取到 {len(df)} 个药物SMILES")
    print(f"列名: {df.columns.tolist()}")
    print(f"前3个药物ID: {df['pdbid'].head(3).tolist()}")
    
    # 检查是否已安装RDKit和transformers
    try:
        from rdkit import Chem
        from transformers import AutoTokenizer, AutoModel
        import torch
        print("✓ RDKit和transformers已安装")
    except ImportError as e:
        print(f"✗ 需要安装依赖: {e}")
        print("请运行: pip install rdkit transformers torch")
        return None
    
    # 加载BerMol模型
    print("\n加载BerMol模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 使用ChemBERTa模型作为替代（BerMol的预训练模型）
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        model = model.to(device)
        model.eval()
        print("✓ ChemBERTa模型加载完成")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("将使用简单的分子指纹作为替代...")
        return extract_drug_features_fingerprint(csv_path, output_path)
    
    # 提取特征
    drug_features = {}
    batch_size = 32
    
    print(f"\n开始提取特征 (batch_size={batch_size})...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取药物特征"):
        drug_id = row['pdbid']
        smiles = row['SMILES']
        
        try:
            # 验证SMILES有效性
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"\n警告: 无效的SMILES - {drug_id}: {smiles}")
                continue
            
            # Tokenize
            inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用[CLS] token的embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            drug_features[str(drug_id)] = embedding
            
        except Exception as e:
            print(f"\n警告: 药物 {drug_id} 提取失败: {e}")
            continue
    
    print(f"\n成功提取 {len(drug_features)} 个药物特征")
    
    # 保存特征
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(drug_features, f)
    
    print(f"✓ 药物1D特征已保存到: {output_path}")
    
    # 输出特征维度信息
    if drug_features:
        sample_feat = next(iter(drug_features.values()))
        print(f"特征维度: {sample_feat.shape if hasattr(sample_feat, 'shape') else len(sample_feat)}")
    
    return drug_features


def extract_drug_features_fingerprint(csv_path, output_path):
    """
    使用RDKit分子指纹作为备用方案提取药物特征
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    print("使用RDKit分子指纹提取特征...")
    
    df = pd.read_csv(csv_path)
    drug_features = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取药物指纹"):
        drug_id = row['pdbid']
        smiles = row['SMILES']
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # 生成Morgan指纹 (类似ECFP4)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.float32)
            for i in range(2048):
                arr[i] = fp[i]
            
            drug_features[str(drug_id)] = arr
            
        except Exception as e:
            continue
    
    # 保存特征
    with open(output_path, 'wb') as f:
        pickle.dump(drug_features, f)
    
    print(f"✓ 药物指纹特征已保存: {len(drug_features)} 个")
    return drug_features


def main():
    """主函数"""
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输入文件
    protein_csv = os.path.join(base_dir, 'project1-main', 'id_seq.csv')
    drug_csv = os.path.join(base_dir, 'project1-main', 'id_smile.csv')
    
    # 输出文件
    output_dir = os.path.join(base_dir, 'features_extraction', 'features_1d')
    protein_output = os.path.join(output_dir, 'protein_features_1d.pkl')
    drug_output = os.path.join(output_dir, 'drug_features_1d.pkl')
    
    print("文件路径配置:")
    print(f"  蛋白质CSV: {protein_csv}")
    print(f"  药物CSV: {drug_csv}")
    print(f"  蛋白质输出: {protein_output}")
    print(f"  药物输出: {drug_output}")
    print()
    
    # 检查输入文件
    if not os.path.exists(protein_csv):
        print(f"错误: 找不到蛋白质CSV文件: {protein_csv}")
        return
    if not os.path.exists(drug_csv):
        print(f"错误: 找不到药物CSV文件: {drug_csv}")
        return
    
    # 提取蛋白质特征
    protein_features = extract_protein_features_esm2(protein_csv, protein_output)
    
    # 提取药物特征
    drug_features = extract_drug_features_bermol(drug_csv, drug_output)
    
    print("\n" + "=" * 60)
    print("1D特征提取完成!")
    print("=" * 60)
    print(f"蛋白质特征数量: {len(protein_features) if protein_features else 0}")
    print(f"药物特征数量: {len(drug_features) if drug_features else 0}")


if __name__ == "__main__":
    main()
