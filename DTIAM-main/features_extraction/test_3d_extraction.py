"""
测试3D特征提取 - 下载PDB并提取3D特征
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import requests
from tqdm import tqdm
import time


def download_pdb_test(protein_ids, output_dir):
    """测试下载PDB文件"""
    print("=" * 60)
    print("测试PDB文件下载")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    success_list = []
    failed_list = []
    
    for protein_id in protein_ids:
        # 尝试不同的PDB ID格式
        pdb_id = protein_id[:4].upper() if len(protein_id) >= 4 else protein_id.upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_path = os.path.join(output_dir, f"{protein_id}.pdb")
        
        print(f"\n尝试下载: {protein_id} (PDB ID: {pdb_id})")
        print(f"  URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'w') as f:
                    f.write(response.text)
                print(f"  ✓ 成功下载")
                success_list.append(protein_id)
            else:
                print(f"  ✗ 失败 (状态码: {response.status_code})")
                failed_list.append(protein_id)
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            failed_list.append(protein_id)
        
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print(f"下载完成: 成功 {len(success_list)}, 失败 {len(failed_list)}")
    print("=" * 60)
    
    return success_list, failed_list


def parse_pdb_file(pdb_file):
    """解析PDB文件提取原子坐标"""
    coords = []
    elements = []
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        element = line[76:78].strip()
                        
                        coords.append([x, y, z])
                        elements.append(element if element else 'C')
                    except:
                        continue
        
        if not coords:
            return None, None
        
        return np.array(coords, dtype=np.float32), elements
    
    except Exception as e:
        return None, None


def extract_3d_features(pdb_dir, protein_ids):
    """从PDB文件提取3D特征"""
    print("\n" + "=" * 60)
    print("提取3D特征")
    print("=" * 60)
    
    from scipy.spatial.distance import pdist
    
    protein_features_3d = {}
    
    for protein_id in protein_ids:
        pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
        
        if not os.path.exists(pdb_file):
            print(f"\n✗ {protein_id}: PDB文件不存在")
            continue
        
        print(f"\n处理: {protein_id}")
        
        # 解析PDB
        coords, elements = parse_pdb_file(pdb_file)
        
        if coords is None:
            print(f"  ✗ 无法解析PDB文件")
            continue
        
        print(f"  原子数量: {len(coords)}")
        
        # 提取统计特征
        features = []
        
        # 1. 中心坐标 (3)
        center = coords.mean(axis=0)
        features.extend(center)
        
        # 2. 标准差 (3)
        std = coords.std(axis=0)
        features.extend(std)
        
        # 3. 最小值 (3)
        min_coords = coords.min(axis=0)
        features.extend(min_coords)
        
        # 4. 最大值 (3)
        max_coords = coords.max(axis=0)
        features.extend(max_coords)
        
        # 5. 原子数量 (1)
        features.append(len(coords))
        
        # 6. 坐标范围 (3)
        ranges = max_coords - min_coords
        features.extend(ranges)
        
        # 7. 质心距离统计 (4)
        distances = np.linalg.norm(coords - center, axis=1)
        features.extend([
            distances.mean(),
            distances.std(),
            distances.min(),
            distances.max()
        ])
        
        # 8. 原子对距离统计 (4)
        if len(coords) > 100:
            sample_idx = np.random.choice(len(coords), 100, replace=False)
            sample_coords = coords[sample_idx]
        else:
            sample_coords = coords
        
        pairwise_distances = pdist(sample_coords)
        features.extend([
            pairwise_distances.mean(),
            pairwise_distances.std(),
            pairwise_distances.min(),
            pairwise_distances.max()
        ])
        
        feature_vec = np.array(features, dtype=np.float32)
        protein_features_3d[protein_id] = feature_vec
        
        print(f"  ✓ 3D特征维度: {feature_vec.shape}")
    
    return protein_features_3d


def main():
    # 读取测试的蛋白质ID
    csv_path = "../project1-main/id_seq.csv"
    df = pd.read_csv(csv_path)
    test_proteins = df['Pdbid'].head(5).tolist()
    
    print("测试蛋白质:", test_proteins)
    
    # 输出目录
    pdb_dir = "../features_extraction/pdb_files"
    
    # 步骤1: 下载PDB
    success_list, failed_list = download_pdb_test(test_proteins, pdb_dir)
    
    if not success_list:
        print("\n警告: 没有成功下载任何PDB文件")
        print("这些蛋白质ID可能不是标准的PDB ID")
        print("可能需要通过UniProt或其他方式获取PDB ID")
        return
    
    # 步骤2: 提取3D特征
    protein_features_3d = extract_3d_features(pdb_dir, success_list)
    
    # 保存
    output_dir = "../features_extraction/features_3d"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "protein_features_3d_test.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(protein_features_3d, f)
    
    print("\n" + "=" * 60)
    print("3D特征提取完成")
    print("=" * 60)
    print(f"成功提取: {len(protein_features_3d)} 个蛋白质3D特征")
    print(f"保存路径: {output_path}")
    
    # 显示特征示例
    if protein_features_3d:
        sample_id = list(protein_features_3d.keys())[0]
        sample_feat = protein_features_3d[sample_id]
        print(f"\n示例特征 ({sample_id}):")
        print(f"  维度: {sample_feat.shape}")
        print(f"  范围: [{sample_feat.min():.4f}, {sample_feat.max():.4f}]")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
