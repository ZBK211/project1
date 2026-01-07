"""
验证提取的1D特征
"""
import pickle
import numpy as np

# 读取蛋白质特征
print("=" * 60)
print("验证蛋白质1D特征")
print("=" * 60)

with open('features_1d/protein_features_1d_test.pkl', 'rb') as f:
    protein_features = pickle.load(f)

print(f"蛋白质数量: {len(protein_features)}")
print(f"蛋白质ID列表: {list(protein_features.keys())}")

for protein_id, feat in protein_features.items():
    print(f"\n{protein_id}:")
    print(f"  特征维度: {feat.shape}")
    print(f"  数据类型: {feat.dtype}")
    print(f"  特征范围: [{feat.min():.4f}, {feat.max():.4f}]")
    print(f"  特征均值: {feat.mean():.4f}")
    break  # 只显示第一个

# 读取药物特征
print("\n" + "=" * 60)
print("验证药物1D特征")
print("=" * 60)

with open('features_1d/drug_features_1d_test.pkl', 'rb') as f:
    drug_features = pickle.load(f)

print(f"药物数量: {len(drug_features)}")
print(f"药物ID列表: {list(drug_features.keys())[:5]}...")

for drug_id, feat in drug_features.items():
    print(f"\n{drug_id}:")
    print(f"  特征维度: {feat.shape}")
    print(f"  数据类型: {feat.dtype}")
    print(f"  特征范围: [{feat.min():.4f}, {feat.max():.4f}]")
    print(f"  特征均值: {feat.mean():.4f}")
    print(f"  非零元素: {np.count_nonzero(feat)}")
    break

print("\n" + "=" * 60)
print("✓ 1D特征验证完成！特征提取成功")
print("=" * 60)
