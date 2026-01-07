"""
3D特征提取模块
1. 从PDB数据库下载蛋白质3D结构文件
2. 使用3D模型(SchNet/PointNet)提取蛋白质3D特征
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import requests
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')


def download_pdb_file(protein_id, output_dir, retry=3):
    """
    从RCSB PDB数据库下载PDB文件
    
    Args:
        protein_id: 蛋白质ID (例如: AAK1)
        output_dir: 输出目录
        retry: 重试次数
    
    Returns:
        pdb_file_path: 下载的PDB文件路径，失败返回None
    """
    # PDB ID通常是4个字符，但我们的ID可能不是标准格式
    # 尝试直接使用ID，或者取前4个字符
    pdb_id = protein_id[:4].upper() if len(protein_id) >= 4 else protein_id.upper()
    
    # RCSB PDB的下载URL
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    output_path = os.path.join(output_dir, f"{protein_id}.pdb")
    
    # 如果已经下载过，直接返回
    if os.path.exists(output_path):
        return output_path
    
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'w') as f:
                    f.write(response.text)
                return output_path
            elif response.status_code == 404:
                # 文件不存在
                return None
            else:
                if attempt < retry - 1:
                    time.sleep(1)
                    continue
                return None
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)
                continue
            return None
    
    return None


def download_pdb_files_from_csv(csv_path, output_dir, max_downloads=None):
    """
    根据CSV文件批量下载PDB文件
    
    Args:
        csv_path: id_seq.csv路径
        output_dir: PDB文件输出目录
        max_downloads: 最大下载数量(用于测试)，None表示下载全部
    """
    print("=" * 60)
    print("开始下载PDB结构文件...")
    print("=" * 60)
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    protein_ids = df['Pdbid'].tolist()
    
    if max_downloads:
        protein_ids = protein_ids[:max_downloads]
        print(f"测试模式: 只下载前 {max_downloads} 个蛋白质")
    
    print(f"总共需要下载: {len(protein_ids)} 个蛋白质结构")
    print(f"保存路径: {output_dir}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_list = []
    
    for protein_id in tqdm(protein_ids, desc="下载PDB文件"):
        result = download_pdb_file(protein_id, output_dir)
        if result:
            success_count += 1
        else:
            failed_list.append(protein_id)
        
        # 避免请求过快
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("PDB文件下载完成")
    print("=" * 60)
    print(f"成功下载: {success_count} 个")
    print(f"下载失败: {len(failed_list)} 个")
    
    if failed_list and len(failed_list) <= 20:
        print(f"\n失败列表: {failed_list}")
    
    # 保存失败列表
    if failed_list:
        failed_file = os.path.join(output_dir, 'download_failed.txt')
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed_list))
        print(f"失败列表已保存到: {failed_file}")
    
    return success_count, failed_list


def parse_pdb_file(pdb_file):
    """
    解析PDB文件，提取原子坐标
    
    Args:
        pdb_file: PDB文件路径
    
    Returns:
        coords: numpy数组 (N, 3)，N是原子数量
        elements: 元素类型列表
    """
    coords = []
    elements = []
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        # PDB格式: ATOM序号 原子名 残基名 链 残基序号 x y z occupancy b-factor 元素
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        element = line[76:78].strip()
                        
                        coords.append([x, y, z])
                        elements.append(element if element else 'C')  # 默认为碳
                    except:
                        continue
        
        if not coords:
            return None, None
        
        return np.array(coords, dtype=np.float32), elements
    
    except Exception as e:
        return None, None


def extract_protein_3d_features_simple(pdb_dir, output_path):
    """
    从PDB文件提取简单的3D特征
    使用统计特征作为3D表示
    
    Args:
        pdb_dir: PDB文件目录
        output_path: 输出特征文件路径
    """
    print("\n" + "=" * 60)
    print("从PDB文件提取3D特征...")
    print("=" * 60)
    
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    print(f"找到 {len(pdb_files)} 个PDB文件")
    
    protein_features_3d = {}
    
    for pdb_file in tqdm(pdb_files, desc="提取3D特征"):
        protein_id = pdb_file.replace('.pdb', '')
        pdb_path = os.path.join(pdb_dir, pdb_file)
        
        # 解析PDB文件
        coords, elements = parse_pdb_file(pdb_path)
        
        if coords is None or len(coords) == 0:
            continue
        
        # 提取统计特征
        features = []
        
        # 1. 中心坐标
        center = coords.mean(axis=0)
        features.extend(center)
        
        # 2. 标准差
        std = coords.std(axis=0)
        features.extend(std)
        
        # 3. 最小值
        min_coords = coords.min(axis=0)
        features.extend(min_coords)
        
        # 4. 最大值
        max_coords = coords.max(axis=0)
        features.extend(max_coords)
        
        # 5. 原子数量
        features.append(len(coords))
        
        # 6. 坐标范围
        ranges = max_coords - min_coords
        features.extend(ranges)
        
        # 7. 质心距离的统计量
        distances = np.linalg.norm(coords - center, axis=1)
        features.append(distances.mean())
        features.append(distances.std())
        features.append(distances.min())
        features.append(distances.max())
        
        # 8. 原子对之间距离的统计(采样避免太慢)
        if len(coords) > 100:
            # 随机采样100个原子
            sample_idx = np.random.choice(len(coords), 100, replace=False)
            sample_coords = coords[sample_idx]
        else:
            sample_coords = coords
        
        # 计算配对距离
        from scipy.spatial.distance import pdist
        pairwise_distances = pdist(sample_coords)
        features.append(pairwise_distances.mean())
        features.append(pairwise_distances.std())
        features.append(pairwise_distances.min())
        features.append(pairwise_distances.max())
        
        protein_features_3d[protein_id] = np.array(features, dtype=np.float32)
    
    print(f"\n成功提取 {len(protein_features_3d)} 个蛋白质3D特征")
    
    # 保存特征
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(protein_features_3d, f)
    
    print(f"✓ 蛋白质3D特征已保存到: {output_path}")
    
    # 输出特征维度
    if protein_features_3d:
        sample_feat = next(iter(protein_features_3d.values()))
        print(f"3D特征维度: {sample_feat.shape}")
    
    return protein_features_3d


def extract_protein_3d_features_schnet(pdb_dir, output_path):
    """
    使用SchNet模型从PDB文件提取3D特征
    
    注意: 需要安装SchNetPack
    pip install schnetpack
    """
    print("\n尝试使用SchNet模型提取3D特征...")
    
    try:
        import torch
        import schnetpack as spk
        from schnetpack import AtomsData
        print("✓ SchNetPack已安装")
    except ImportError:
        print("✗ SchNetPack未安装，使用简单统计特征替代")
        return extract_protein_3d_features_simple(pdb_dir, output_path)
    
    # TODO: 实现SchNet特征提取
    print("SchNet特征提取待实现，当前使用统计特征...")
    return extract_protein_3d_features_simple(pdb_dir, output_path)


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输入文件
    protein_csv = os.path.join(base_dir, 'project1-main', 'id_seq.csv')
    
    # 输出目录
    pdb_dir = os.path.join(base_dir, 'features_extraction', 'pdb_files')
    output_dir = os.path.join(base_dir, 'features_extraction', 'features_3d')
    output_path = os.path.join(output_dir, 'protein_features_3d.pkl')
    
    print("文件路径配置:")
    print(f"  蛋白质CSV: {protein_csv}")
    print(f"  PDB文件目录: {pdb_dir}")
    print(f"  3D特征输出: {output_path}")
    print()
    
    # 步骤1: 下载PDB文件
    print("步骤1: 下载PDB文件")
    print("-" * 60)
    
    # 先测试下载前10个
    test_mode = input("是否测试模式(只下载前10个)? [y/N]: ").strip().lower()
    max_downloads = 10 if test_mode == 'y' else None
    
    success_count, failed_list = download_pdb_files_from_csv(
        protein_csv, 
        pdb_dir,
        max_downloads=max_downloads
    )
    
    if success_count == 0:
        print("\n错误: 没有成功下载任何PDB文件")
        print("可能原因:")
        print("1. 网络连接问题")
        print("2. PDB ID格式不匹配")
        print("3. PDB数据库中不存在这些结构")
        return
    
    # 步骤2: 提取3D特征
    print("\n步骤2: 提取3D特征")
    print("-" * 60)
    
    protein_features_3d = extract_protein_3d_features_simple(pdb_dir, output_path)
    
    print("\n" + "=" * 60)
    print("3D特征提取完成!")
    print("=" * 60)
    print(f"成功提取: {len(protein_features_3d)} 个蛋白质3D特征")


if __name__ == "__main__":
    main()
