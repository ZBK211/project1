"""
蛋白质3D结构特征提取器
使用PDB文件提取蛋白质的3D结构特征
"""

import os
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import protein_letters_3to1


class ProteinStructureFeatureExtractor:
    """蛋白质3D结构特征提取器"""
    
    # 氨基酸字母表 (20种标准氨基酸 + X为未知)
    AA_DICT = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    AA_DICT['X'] = 0  # 未知氨基酸
    
    # 原子类型映射
    ATOM_TYPES = {
        'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
        'H': 1, 'SE': 34, 'FE': 26, 'ZN': 30, 'MG': 12,
        'CA': 20, 'MN': 25, 'CU': 29, 'NA': 11, 'K': 19, 'CL': 17
    }
    
    def __init__(self, pdb_dir: str = "../data/pdb_structures"):
        self.pdb_dir = Path(pdb_dir)
        self.parser = PDBParser(QUIET=True)
        
    def extract_residue_features(self, pdb_path: str, max_residues: int = 500) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从PDB文件提取残基级别的特征
        
        Args:
            pdb_path: PDB文件路径
            max_residues: 最大残基数量（截断长序列）
            
        Returns:
            (residue_types, positions): 残基类型和位置坐标
            residue_types: shape (n_residues,)
            positions: shape (n_residues, 3) - CA原子的坐标
        """
        try:
            structure = self.parser.get_structure('protein', pdb_path)
            
            residue_types = []
            positions = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # 只处理氨基酸残基
                        if residue.id[0] == ' ':
                            try:
                                # 获取残基名称并转换为单字母代码
                                res_name = residue.get_resname()
                                aa_code = protein_letters_3to1.get(res_name, 'X')
                                res_type = self.AA_DICT.get(aa_code, 0)
                                
                                # 获取CA原子坐标
                                if 'CA' in residue:
                                    ca_coord = residue['CA'].get_coord()
                                    residue_types.append(res_type)
                                    positions.append(ca_coord)
                                    
                            except Exception:
                                continue
                                
                # 只处理第一个模型
                break
            
            if len(residue_types) == 0:
                return None
                
            residue_types = np.array(residue_types[:max_residues], dtype=np.int32)
            positions = np.array(positions[:max_residues], dtype=np.float32)
            
            return residue_types, positions
            
        except Exception as e:
            print(f"❌ 解析PDB文件失败: {pdb_path}, 错误: {str(e)}")
            return None
    
    def extract_atom_features(self, pdb_path: str, max_atoms: int = 3000) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从PDB文件提取原子级别的特征（更精细）
        
        Args:
            pdb_path: PDB文件路径
            max_atoms: 最大原子数量
            
        Returns:
            (atom_types, positions): 原子类型和位置坐标
        """
        try:
            structure = self.parser.get_structure('protein', pdb_path)
            
            atom_types = []
            positions = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':  # 标准残基
                            for atom in residue:
                                element = atom.element.upper()
                                atom_type = self.ATOM_TYPES.get(element, 0)
                                
                                if atom_type > 0:  # 只保留已知元素
                                    atom_types.append(atom_type)
                                    positions.append(atom.get_coord())
                                    
                                if len(atom_types) >= max_atoms:
                                    break
                            if len(atom_types) >= max_atoms:
                                break
                    if len(atom_types) >= max_atoms:
                        break
                break
            
            if len(atom_types) == 0:
                return None
                
            atom_types = np.array(atom_types, dtype=np.int32)
            positions = np.array(positions, dtype=np.float32)
            
            return atom_types, positions
            
        except Exception as e:
            print(f"❌ 解析原子特征失败: {pdb_path}, 错误: {str(e)}")
            return None
    
    def compute_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """计算距离矩阵"""
        n = len(positions)
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                
        return dist_matrix
    
    def create_graph_features(self, residue_types: np.ndarray, 
                             positions: np.ndarray,
                             cutoff: float = 10.0) -> Dict:
        """
        创建图结构特征（用于GNN模型）
        
        Args:
            residue_types: 残基类型
            positions: 残基坐标
            cutoff: 距离截断阈值（埃）
            
        Returns:
            图特征字典
        """
        n_residues = len(residue_types)
        
        # 计算距离矩阵
        dist_matrix = self.compute_distance_matrix(positions)
        
        # 构建边（距离小于cutoff的残基对）
        edge_index = []
        edge_attr = []
        
        for i in range(n_residues):
            for j in range(n_residues):
                if i != j and dist_matrix[i, j] < cutoff:
                    edge_index.append([i, j])
                    edge_attr.append(dist_matrix[i, j])
        
        edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32) if edge_attr else np.zeros(0, dtype=np.float32)
        
        return {
            'node_features': residue_types,  # (n_residues,)
            'positions': positions,           # (n_residues, 3)
            'edge_index': edge_index,         # (2, n_edges)
            'edge_attr': edge_attr,           # (n_edges,)
            'num_nodes': n_residues
        }
    
    def extract_simple_3d_features(self, residue_types: np.ndarray, 
                                   positions: np.ndarray,
                                   feature_dim: int = 256) -> np.ndarray:
        """
        提取简化的3D特征向量（用于传统机器学习）
        
        Args:
            residue_types: 残基类型
            positions: 残基坐标
            feature_dim: 输出特征维度
            
        Returns:
            特征向量
        """
        features = []
        
        # 1. 统计特征
        features.extend([
            len(residue_types),  # 序列长度
            np.mean(residue_types),  # 平均残基类型
            np.std(residue_types),   # 残基类型标准差
        ])
        
        # 2. 几何特征
        if len(positions) > 0:
            center = np.mean(positions, axis=0)
            features.extend(center)  # 质心坐标
            
            # 距质心的距离统计
            distances = np.linalg.norm(positions - center, axis=1)
            features.extend([
                np.mean(distances),
                np.std(distances),
                np.max(distances),
                np.min(distances),
            ])
            
            # 主成分分析（协方差矩阵的特征值）
            cov_matrix = np.cov(positions.T)
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            features.extend(eigenvalues)
        else:
            features.extend([0] * 11)
        
        # 3. 距离统计特征
        if len(positions) > 1:
            dist_matrix = self.compute_distance_matrix(positions)
            upper_tri = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
            
            features.extend([
                np.mean(upper_tri),
                np.std(upper_tri),
                np.median(upper_tri),
                np.percentile(upper_tri, 25),
                np.percentile(upper_tri, 75),
            ])
        else:
            features.extend([0] * 5)
        
        # 4. 氨基酸组成（归一化）
        aa_composition = np.bincount(residue_types, minlength=21) / max(len(residue_types), 1)
        features.extend(aa_composition)
        
        # 转换为numpy数组
        feature_vector = np.array(features, dtype=np.float32)
        
        # 如果维度不足，用零填充；如果过多，截断
        if len(feature_vector) < feature_dim:
            feature_vector = np.pad(feature_vector, (0, feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:feature_dim]
            
        return feature_vector


def extract_protein_3d_features(protein_ids: list, 
                               pdb_dir: str = "../data/pdb_structures",
                               output_path: str = None,
                               mode: str = "simple") -> Dict[str, np.ndarray]:
    """
    批量提取蛋白质3D特征
    
    Args:
        protein_ids: 蛋白质ID列表
        pdb_dir: PDB文件目录
        output_path: 输出文件路径
        mode: 'simple' (简单特征向量) 或 'graph' (图结构)
        
    Returns:
        特征字典 {protein_id: features}
    """
    extractor = ProteinStructureFeatureExtractor(pdb_dir)
    features_dict = {}
    
    print(f"🧬 提取3D结构特征 (模式: {mode})")
    
    for pid in tqdm(protein_ids, desc="处理PDB文件"):
        # 尝试标准PDB ID
        pdb_path = Path(pdb_dir) / f"{pid}.pdb"
        if not pdb_path.exists():
            # 尝试AlphaFold格式
            pdb_path = Path(pdb_dir) / f"{pid}_alphafold.pdb"
        
        if not pdb_path.exists():
            print(f"⚠️ 找不到PDB文件: {pid}")
            # 使用零向量作为占位符
            if mode == "simple":
                features_dict[pid] = np.zeros(256, dtype=np.float32)
            continue
        
        # 提取残基特征
        result = extractor.extract_residue_features(str(pdb_path))
        
        if result is None:
            print(f"⚠️ 无法提取特征: {pid}")
            if mode == "simple":
                features_dict[pid] = np.zeros(256, dtype=np.float32)
            continue
        
        residue_types, positions = result
        
        # 根据模式生成不同类型的特征
        if mode == "simple":
            features = extractor.extract_simple_3d_features(residue_types, positions)
            features_dict[pid] = features
        elif mode == "graph":
            features = extractor.create_graph_features(residue_types, positions)
            features_dict[pid] = features
    
    # 保存特征
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"✅ 3D特征已保存到: {output_path}")
    
    return features_dict


if __name__ == "__main__":
    # 测试代码
    import pandas as pd
    
    csv_path = "../project1-main/id_seq.csv"
    df = pd.read_csv(csv_path)
    protein_ids = df['Pdbid'].dropna().unique()[:10]  # 测试前10个
    
    print(f"测试提取 {len(protein_ids)} 个蛋白质的3D特征...")
    features = extract_protein_3d_features(
        protein_ids, 
        mode="simple",
        output_path="../data/test_protein_3d_features.pkl"
    )
    
    print(f"✅ 成功提取 {len(features)} 个蛋白质的特征")
    for pid, feat in list(features.items())[:3]:
        print(f"  {pid}: shape {feat.shape}")
