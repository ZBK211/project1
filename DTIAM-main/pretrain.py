import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
import pymol
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
# import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import tqdm
from Bio.Seq import Seq
import numpy as np

def parse_index_file(filepath, output_csv):
    records = []
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # 跳过注释行，提取有效数据行（一般从第6或第7行开始）
    for line in lines:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        parts = line.strip().split()
        if len(parts) >= 4:
            pdb_id = parts[0]
            log_affinity = float(parts[3])
            records.append([pdb_id, log_affinity])

    # 保存为 CSV
    df = pd.DataFrame(records, columns=['pdbid', '-logKd/Ki'])
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

def mol2_to_smiles(mol2_path):
    try:
        mol = Chem.MolFromMol2File(mol2_path, sanitize=True)
        if mol is None:
            return None
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except Exception as e:
        print(f"Error reading {mol2_path}: {e}")
        return None

def pdb_to_sequence(pdb_path):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        ppb = PPBuilder()
        sequences = []
        for pp in ppb.build_peptides(structure):
            seq = str(pp.get_sequence())
            sequences.append(seq)
        return "".join(sequences)
    except Exception as e:
        print(f"Error reading {pdb_path}: {e}")
        return None

def extract_smiles_and_sequence(data_dir, output_csv):
    data = []
    for cid in os.listdir(data_dir):
        complex_dir = os.path.join(data_dir, cid)
        ligand_path = os.path.join(complex_dir, f"{cid}_ligand.mol2")
        protein_path = os.path.join(complex_dir, f"{cid}_protein.pdb")

        if not os.path.exists(ligand_path) or not os.path.exists(protein_path):
            print(f"Missing files for {cid}, skipping.")
            continue

        smiles = mol2_to_smiles(ligand_path)
        sequence = pdb_to_sequence(protein_path)

        if smiles is None or sequence is None:
            print(f"Error extracting data for {cid}, skipping.")
            continue

        data.append({
            "pdbid": cid,
            "SMILES": smiles,
            "Acid_Sequence": sequence
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved output to {output_csv}")

def generate_pocket(data_dir, distance=5):
    complex_id = os.listdir(data_dir)
    for cid in complex_id:
        print(cid)
        complex_dir = os.path.join(data_dir, cid)
        lig_native_path = os.path.join(complex_dir, f"{cid}_ligand.mol2")
        protein_path= os.path.join(complex_dir, f"{cid}_protein.pdb")

        if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, distance=5, input_ligand_format='mol2'):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
        complex_dir = os.path.join(data_dir, cid)
        pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        if input_ligand_format != 'pdb':
            ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
            ligand_path = ligand_input_path.replace(f".{input_ligand_format}", ".pdb")
            os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')
        else:
            ligand_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')

        save_path = os.path.join(complex_dir, f"{cid}_{distance}A.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        if ligand == None:
            print(f"Unable to process ligand of {cid}")
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        pbar.update(1)

def extract_embedding():
    import pandas as pd
    import torch
    import numpy as np
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel
    import os

    # 检查输入文件是否存在
    input_file = "id_seq.csv"
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    print("加载数据")
    # 加载 CSV 数据并检查必要的列
    df = pd.read_csv(input_file)
    required_columns = ["Pdbid", "Acid_Sequence"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain columns: {required_columns}")
        return
    print("加载模型")
    # 加载模型
    model_name = "facebook/esm2_t33_650M_UR50D"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        #model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D", force_download=True)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 是否使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_parts = 5
    chunk_size = len(df) // total_parts

    # 创建临时文件存储目录
    temp_dir = "temp_embeddings"
    os.makedirs(temp_dir, exist_ok=True)

    for part in range(total_parts):
        print(f"\n🔹 正在处理第 {part + 1}/{total_parts} 部分")
        start_idx = part * chunk_size
        end_idx = (part + 1) * chunk_size if part < total_parts - 1 else len(df)
        chunk = df.iloc[start_idx:end_idx]

        embeddings = {}
        pdbids = []

        for idx, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Part {part + 1}"):
            pdbid = row["Pdbid"]
            sequence = row["Acid_Sequence"]

            try:
                inputs = tokenizer(sequence, return_tensors="pt", truncation=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # 使用平均池化
                embedding = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1).squeeze().cpu().numpy()
                embeddings[pdbid] = embedding
                pdbids.append(pdbid)
            except Exception as e:
                print(f"[ERROR] {pdbid}: {e}")
                continue

        # 保存每一部分
        part_file = os.path.join(temp_dir, f"esm2_embeddings_part_{part}.npz")
        np.savez(part_file, **embeddings)
        print(f"Saved part {part + 1} to {part_file}")

    # 合并所有部分
    print("\n🔹 合并所有部分...")
    merged = {}
    for i in range(total_parts):
        part_file = os.path.join(temp_dir, f"esm2_embeddings_part_{i}.npz")
        try:
            part = dict(np.load(part_file, allow_pickle=True))
            merged.update(part)
        except Exception as e:
            print(f"Error loading part {i}: {e}")

    # 保存最终结果
    final_file = "esm2_embeddings.npz"
    np.savez(final_file, **merged)
    print(f"\n✅ 完成！最终结果保存为: {final_file}")

    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    print("临时文件已清理")

def merge():
    import pandas as pd

    # 读取两个CSV文件
    df1 = pd.read_csv('train_seq.csv')
    df2 = pd.read_csv('valid_seq.csv')

    # 合并两个DataFrame
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 去除重复项（基于'smiles'列）
    combined_df = combined_df.drop_duplicates(subset='pdbid')

    # 保存为新的CSV文件
    combined_df.to_csv('all.csv', index=False)

    print("合并并去重完成，结果保存为 merged_deduplicated_seq.csv")

def smile_embedding():
    import torch
    import pandas as pd
    import numpy as np
    from transformers import RobertaTokenizer, RobertaModel
    from tqdm import tqdm

    # 加载 ChemBERTa 模型
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model.eval()

    # 加载数据
    df = pd.read_csv("id_smile.csv")
    num_parts = 5
    chunk_size = len(df) // num_parts

    for part in range(num_parts):
        start_idx = part * chunk_size
        end_idx = (part + 1) * chunk_size if part < num_parts - 1 else len(df)
        chunk_df = df.iloc[start_idx:end_idx]

        embeddings = {}
        ids = []

        for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Embedding part {part}"):
            pdbid = row["pdbid"]
            smiles = row["SMILES"]

            try:
                inputs = tokenizer(smiles, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings[pdbid] = embedding
                ids.append(pdbid)
            except Exception as e:
                print(f"Failed for {pdbid}: {e}")

        # 保存每一份
        np.savez(f"smiles_embedding_part_{part}.npz", **embeddings)

    merged = {}
    for i in range(5):  # 假设有5个文件
        part = dict(np.load(f"smiles_embedding_part_{i}.npz", allow_pickle=True))
        merged.update(part)

    np.savez("smiles_embeddings_merged.npz", **merged)


if __name__ == '__main__':
    distance = 5
    input_ligand_format = 'mol2'
    data_root = './data'
    # parse_index_file('v2020-other-PL/index/INDEX_general_PL_data.2020', 'pdbbind2020.csv')
    # extract_smiles_and_sequence('unmatch', 'unmatch-seq.csv')

    # merge()
    # data = np.load("smiles_embeddings_merged.npz")
    # print(data.files)
    smile_embedding()
    #extract_embedding()
    # data_dir = os.path.join(data_root, 'toy_set')
    # data_df = pd.read_csv(os.path.join(data_root, 'pdbbind_2020.csv'))
    #
    # ## generate pocket within 5 Ångström around ligand
    # generate_pocket(data_dir=data_dir, distance=distance)
    # generate_complex(data_dir, data_df, distance=distance, input_ligand_format=input_ligand_format)



# %%
