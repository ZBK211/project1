"""
从蛋白质序列生成3D特征（不依赖PDB文件）
使用序列衍生的3D描述符
"""
import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


def calculate_sequence_3d_features(sequence):
    """
    从蛋白质序列计算3D相关特征
    不需要真实的3D结构，使用氨基酸的物理化学性质
    """
    # 氨基酸的物理化学性质字典
    aa_properties = {
        'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'charge': 0, 'polarity': 0},
        'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'charge': 1, 'polarity': 1},
        'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'charge': 0, 'polarity': 1},
        'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'charge': -1, 'polarity': 1},
        'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'charge': 0, 'polarity': 0},
        'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'charge': 0, 'polarity': 1},
        'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'charge': -1, 'polarity': 1},
        'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'charge': 0, 'polarity': 0},
        'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'charge': 0.5, 'polarity': 1},
        'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'charge': 0, 'polarity': 0},
        'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'charge': 0, 'polarity': 0},
        'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'charge': 1, 'polarity': 1},
        'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'charge': 0, 'polarity': 0},
        'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'charge': 0, 'polarity': 0},
        'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'charge': 0, 'polarity': 0},
        'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'charge': 0, 'polarity': 1},
        'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'charge': 0, 'polarity': 1},
        'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'charge': 0, 'polarity': 0},
        'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'charge': 0, 'polarity': 1},
        'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'charge': 0, 'polarity': 0},
    }
    
    # 默认值（用于未知氨基酸）
    default_props = {'hydrophobicity': 0, 'volume': 100, 'charge': 0, 'polarity': 0}
    
    features = []
    
    # 1. 序列长度
    seq_length = len(sequence)
    features.append(seq_length)
    
    # 2. 提取各种性质的统计量
    hydrophobicity = []
    volumes = []
    charges = []
    polarities = []
    
    for aa in sequence:
        props = aa_properties.get(aa.upper(), default_props)
        hydrophobicity.append(props['hydrophobicity'])
        volumes.append(props['volume'])
        charges.append(props['charge'])
        polarities.append(props['polarity'])
    
    hydrophobicity = np.array(hydrophobicity)
    volumes = np.array(volumes)
    charges = np.array(charges)
    polarities = np.array(polarities)
    
    # 3. 疏水性统计 (4个特征)
    features.extend([
        hydrophobicity.mean(),
        hydrophobicity.std(),
        hydrophobicity.min(),
        hydrophobicity.max()
    ])
    
    # 4. 体积统计 (4个特征)
    features.extend([
        volumes.mean(),
        volumes.std(),
        volumes.min(),
        volumes.max()
    ])
    
    # 5. 电荷统计 (4个特征)
    features.extend([
        charges.mean(),
        charges.std(),
        charges.min(),
        charges.max()
    ])
    
    # 6. 极性统计 (4个特征)
    features.extend([
        polarities.mean(),
        polarities.std() if len(polarities) > 1 else 0,
        polarities.min(),
        polarities.max()
    ])
    
    # 7. 氨基酸组成 (20个特征)
    aa_composition = {}
    for aa in 'ARNDCQEGHILKMFPSTWYV':
        count = sequence.upper().count(aa)
        aa_composition[aa] = count / seq_length if seq_length > 0 else 0
        features.append(aa_composition[aa])
    
    # 8. 二肽组成 (统计最常见的10个二肽)
    dipeptide_counts = {}
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2].upper()
        dipeptide_counts[dipeptide] = dipeptide_counts.get(dipeptide, 0) + 1
    
    # 取top10二肽的频率
    top_dipeptides = sorted(dipeptide_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for _, count in top_dipeptides:
        features.append(count / (seq_length - 1) if seq_length > 1 else 0)
    
    # 补齐到10个
    while len(features) < (1 + 4*4 + 20 + 10):
        features.append(0.0)
    
    # 9. 计算局部疏水性矩 (滑动窗口)
    window_size = 7
    if len(hydrophobicity) >= window_size:
        local_hydro = []
        for i in range(len(hydrophobicity) - window_size + 1):
            window = hydrophobicity[i:i+window_size]
            local_hydro.append(window.mean())
        local_hydro = np.array(local_hydro)
        
        features.extend([
            local_hydro.mean(),
            local_hydro.std(),
            local_hydro.min(),
            local_hydro.max()
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # 10. 模拟空间分布 - 使用位置编码
    position_encoding = []
    for pos in range(min(100, len(sequence))):  # 只取前100个位置
        position_encoding.append(np.sin(pos / 10.0))
    
    if position_encoding:
        features.extend([
            np.mean(position_encoding),
            np.std(position_encoding),
            np.min(position_encoding),
            np.max(position_encoding)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    return np.array(features, dtype=np.float32)


def extract_pseudo_3d_features(csv_path, output_path):
    """
    从CSV文件提取伪3D特征
    """
    print("=" * 60)
    print("从蛋白质序列提取伪3D特征")
    print("=" * 60)
    print("注意: 由于无法获取真实的PDB结构文件,")
    print("      使用序列的物理化学性质生成伪3D特征")
    print("=" * 60)
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    print(f"\n读取到 {len(df)} 个蛋白质序列")
    
    # 只处理前5个用于测试
    df_test = df.head(5)
    
    protein_features_3d = {}
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="提取伪3D特征"):
        protein_id = row['Pdbid']
        sequence = row['Acid_Sequence']
        
        try:
            features = calculate_sequence_3d_features(sequence)
            protein_features_3d[protein_id] = features
        except Exception as e:
            print(f"\n警告: {protein_id} 提取失败: {e}")
            continue
    
    print(f"\n成功提取 {len(protein_features_3d)} 个伪3D特征")
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(protein_features_3d, f)
    
    print(f"✓ 伪3D特征已保存: {output_path}")
    
    # 显示示例
    if protein_features_3d:
        sample_id = list(protein_features_3d.keys())[0]
        sample_feat = protein_features_3d[sample_id]
        print(f"\n示例特征 ({sample_id}):")
        print(f"  特征维度: {sample_feat.shape}")
        print(f"  数据类型: {sample_feat.dtype}")
        print(f"  特征范围: [{sample_feat.min():.4f}, {sample_feat.max():.4f}]")
        print(f"  特征均值: {sample_feat.mean():.4f}")
    
    return protein_features_3d


def main():
    csv_path = "../project1-main/id_seq.csv"
    output_dir = "../features_extraction/features_3d"
    output_path = os.path.join(output_dir, "protein_features_3d_test.pkl")
    
    protein_features = extract_pseudo_3d_features(csv_path, output_path)
    
    print("\n" + "=" * 60)
    print("伪3D特征提取完成!")
    print("=" * 60)
    print(f"提取数量: {len(protein_features)}")
    print("\n说明:")
    print("  由于数据集中的蛋白质ID不是标准PDB ID,")
    print("  我们使用了基于序列的物理化学性质作为3D特征的替代")
    print("  这些特征包括:")
    print("    - 疏水性、体积、电荷、极性等物理化学性质")
    print("    - 氨基酸组成和二肽组成")
    print("    - 局部疏水性矩")
    print("    - 位置编码信息")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
