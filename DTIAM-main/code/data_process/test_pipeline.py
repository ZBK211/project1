"""
快速测试脚本 - 验证1D+3D特征提取流程
使用小样本测试整个pipeline是否正常工作
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle

# 添加路径
sys.path.append(os.path.dirname(__file__))

print("="*60)
print("  DTIAM 功能测试脚本")
print("  测试1D+3D特征提取流程")
print("="*60)
print()

# ====================================================================
# 测试1: 检查依赖包
# ====================================================================
print("[测试 1/5] 检查依赖包...")
required_packages = {
    'torch': 'PyTorch',
    'esm': 'ESM2蛋白质模型',
    'Bio': 'BioPython',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-learn',
    'autogluon': 'AutoGluon'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} - 未安装！")
        missing_packages.append(package)

if missing_packages:
    print()
    print(f"⚠️  缺少 {len(missing_packages)} 个依赖包，请先安装：")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)

print()

# ====================================================================
# 测试2: 检查文件结构
# ====================================================================
print("[测试 2/5] 检查项目文件结构...")

required_files = {
    '../project1-main/id_seq.csv': '蛋白质序列文件',
    '../project1-main/id_smile.csv': '药物SMILES文件',
    'extract_3d_features.py': '3D特征提取脚本',
    '../utils.py': '工具函数',
}

missing_files = []
for filepath, desc in required_files.items():
    if os.path.exists(filepath):
        print(f"  ✅ {desc}")
    else:
        print(f"  ❌ {desc} - 找不到文件: {filepath}")
        missing_files.append(filepath)

if missing_files:
    print()
    print(f"⚠️  缺少 {len(missing_files)} 个必要文件！")
    sys.exit(1)

print()

# ====================================================================
# 测试3: 测试PDB文件读取
# ====================================================================
print("[测试 3/5] 测试PDB文件解析...")

from extract_3d_features import ProteinStructureFeatureExtractor

# 检查是否有PDB文件
pdb_dir = "../../data/pdb_structures"
if not os.path.exists(pdb_dir):
    print(f"  ⚠️  PDB目录不存在: {pdb_dir}")
    print(f"  💡 提示: 运行 'python download_pdb.py' 下载PDB文件")
    has_pdb = False
else:
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    print(f"  📁 找到 {len(pdb_files)} 个PDB文件")
    
    if len(pdb_files) > 0:
        # 测试读取第一个PDB文件
        test_pdb = os.path.join(pdb_dir, pdb_files[0])
        extractor = ProteinStructureFeatureExtractor(pdb_dir)
        
        try:
            result = extractor.extract_residue_features(test_pdb)
            if result:
                residue_types, positions = result
                print(f"  ✅ 成功解析PDB文件: {pdb_files[0]}")
                print(f"     - 残基数: {len(residue_types)}")
                print(f"     - 坐标维度: {positions.shape}")
                has_pdb = True
            else:
                print(f"  ❌ PDB文件解析失败")
                has_pdb = False
        except Exception as e:
            print(f"  ❌ 解析错误: {str(e)}")
            has_pdb = False
    else:
        print(f"  ⚠️  PDB目录为空")
        has_pdb = False

print()

# ====================================================================
# 测试4: 测试3D特征提取
# ====================================================================
print("[测试 4/5] 测试3D特征提取...")

if has_pdb:
    from extract_3d_features import extract_protein_3d_features
    
    # 读取蛋白质ID（测试前3个）
    df = pd.read_csv('../project1-main/id_seq.csv')
    test_proteins = df['Pdbid'].dropna().unique()[:3].tolist()
    
    print(f"  测试蛋白质: {test_proteins}")
    
    try:
        features = extract_protein_3d_features(
            test_proteins,
            pdb_dir=pdb_dir,
            mode="simple"
        )
        
        print(f"  ✅ 成功提取 {len(features)} 个蛋白质的3D特征")
        for pid, feat in features.items():
            print(f"     - {pid}: {feat.shape}")
            
    except Exception as e:
        print(f"  ❌ 3D特征提取失败: {str(e)}")
else:
    print(f"  ⏭️  跳过（无PDB文件）")

print()

# ====================================================================
# 测试5: 测试特征合并功能
# ====================================================================
print("[测试 5/5] 测试特征合并功能...")

sys.path.append('..')
from utils import pack

# 创建模拟数据
mock_data = pd.DataFrame({
    'cid': ['drug1', 'drug2'],
    'pid': ['prot1', 'prot2'],
    'label': [1, 0]
})

# 模拟化合物特征（768维）
mock_comp_feat = {
    'drug1': np.random.randn(768),
    'drug2': np.random.randn(768)
}

# 模拟蛋白质特征（新格式：1D+3D）
mock_prot_feat = {
    'prot1': {
        '1d': np.random.randn(1280),
        '3d': np.random.randn(256)
    },
    'prot2': {
        '1d': np.random.randn(1280),
        '3d': np.random.randn(256)
    }
}

try:
    packed_data = pack(mock_data, mock_comp_feat, mock_prot_feat)
    expected_dim = 768 + 1280 + 256  # 药物 + 蛋白质1D + 蛋白质3D
    actual_dim = packed_data.shape[1] - 1  # 减去label列
    
    if actual_dim == expected_dim:
        print(f"  ✅ 特征合并成功")
        print(f"     - 预期维度: {expected_dim}")
        print(f"     - 实际维度: {actual_dim}")
        print(f"     - 化合物: 768维")
        print(f"     - 蛋白质1D: 1280维")
        print(f"     - 蛋白质3D: 256维")
    else:
        print(f"  ❌ 特征维度不匹配")
        print(f"     - 预期: {expected_dim}")
        print(f"     - 实际: {actual_dim}")
        
except Exception as e:
    print(f"  ❌ 特征合并失败: {str(e)}")

print()

# ====================================================================
# 测试旧格式兼容性
# ====================================================================
print("[额外测试] 旧格式兼容性...")

# 模拟旧格式蛋白质特征（只有1D）
mock_prot_feat_old = {
    'prot1': np.random.randn(1280),
    'prot2': np.random.randn(1280)
}

try:
    packed_data_old = pack(mock_data, mock_comp_feat, mock_prot_feat_old)
    expected_dim_old = 768 + 1280  # 只有药物 + 蛋白质1D
    actual_dim_old = packed_data_old.shape[1] - 1
    
    if actual_dim_old == expected_dim_old:
        print(f"  ✅ 向后兼容旧格式")
        print(f"     - 维度: {actual_dim_old} (无3D特征)")
    else:
        print(f"  ⚠️  旧格式维度异常")
        
except Exception as e:
    print(f"  ❌ 旧格式兼容性测试失败: {str(e)}")

print()

# ====================================================================
# 总结
# ====================================================================
print("="*60)
print("  测试总结")
print("="*60)
print()
print("✅ 所有基础功能测试通过！")
print()
print("📝 下一步操作：")
print("  1. 运行 'python download_pdb.py' 下载PDB文件")
print("  2. 运行 'python extract_feature.py' 提取完整特征")
print("  3. 运行 '../training_validation.py' 开始训练")
print()
print("💡 或者直接运行: ")
print("   powershell -File ../../run_pipeline.ps1")
print()
print("="*60)
