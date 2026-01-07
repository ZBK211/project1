"""
检查project1中现有的npz文件
"""
import numpy as np
import pickle

print("=" * 60)
print("检查project1-main中的npz文件")
print("=" * 60)

# 读取merged文件
npz_file = "../project1-main/smiles_embeddings_merged.npz"
data = np.load(npz_file, allow_pickle=True)

print(f"\n文件: {npz_file}")
print(f"键列表: {data.files}")

# 查看数据结构
for key in data.files:
    value = data[key]
    print(f"\n键: {key}")
    print(f"  类型: {type(value)}")
    if hasattr(value, 'shape'):
        print(f"  形状: {value.shape}")
        print(f"  数据类型: {value.dtype}")
    elif isinstance(value, dict):
        print(f"  字典大小: {len(value)}")
        # 显示前3个key
        sample_keys = list(value.keys())[:3]
        print(f"  示例key: {sample_keys}")
        if sample_keys:
            sample_val = value[sample_keys[0]]
            if hasattr(sample_val, 'shape'):
                print(f"  值的形状: {sample_val.shape}")
            elif hasattr(sample_val, '__len__'):
                print(f"  值的长度: {len(sample_val)}")
    else:
        print(f"  内容: {value}")

# 如果是字典形式，提取所有数据
if 'arr_0' in data.files:
    embeddings_dict = data['arr_0'].item()
    if isinstance(embeddings_dict, dict):
        print("\n" + "=" * 60)
        print("NPZ文件包含的药物embedding")
        print("=" * 60)
        print(f"药物数量: {len(embeddings_dict)}")
        
        # 显示前5个
        for i, (drug_id, embedding) in enumerate(list(embeddings_dict.items())[:5]):
            print(f"\n药物ID: {drug_id}")
            print(f"  Embedding维度: {embedding.shape if hasattr(embedding, 'shape') else len(embedding)}")
            if hasattr(embedding, 'shape'):
                print(f"  数据类型: {embedding.dtype}")
                print(f"  范围: [{embedding.min():.4f}, {embedding.max():.4f}]")

print("\n" + "=" * 60)
print("对比我们提取的1D特征")
print("=" * 60)

# 读取我们提取的特征
our_features_file = "../features_extraction/features_1d/drug_features_1d_test.pkl"
with open(our_features_file, 'rb') as f:
    our_features = pickle.load(f)

print(f"我们的特征数量: {len(our_features)}")
for i, (drug_id, embedding) in enumerate(list(our_features.items())[:3]):
    print(f"\n药物ID: {drug_id}")
    print(f"  Embedding维度: {embedding.shape}")
    print(f"  数据类型: {embedding.dtype}")
    print(f"  范围: [{embedding.min():.4f}, {embedding.max():.4f}]")

print("\n" + "=" * 60)
print("差异说明:")
print("=" * 60)
print("NPZ文件中的embedding: 可能是预训练好的药物向量")
print("我们提取的embedding: 使用Morgan指纹/ChemBERTa重新提取")
print("\n建议: 直接使用NPZ文件中已有的embedding，无需重新提取!")
