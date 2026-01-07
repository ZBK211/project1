import json
import pickle
from collections import OrderedDict
import pandas as pd
import numpy as np
import csv


def load_davis_mapping():
    """Load the mapping file for DAVIS dataset (target-id to uniprot)"""
    try:
        mapping_df = pd.read_csv("davis_target_uniprot.csv")
        return dict(zip(mapping_df['target-id'], mapping_df['uniprot']))
    except FileNotFoundError:
        raise FileNotFoundError("DAVIS mapping file 'davis_target_uniprot.csv' not found.")
    except KeyError:
        raise KeyError("DAVIS mapping file must contain 'target-id' and 'uniprot' columns")


def process_dataset(dataset: str):
    """
    Process the dataset and save as CSV with columns:
    - For DAVIS: drug-id, uniprot, affinity, Label
    - For KIBA: drug-id, uniprot, affinity, Label
    """
    # Load mapping for DAVIS
    davis_mapping = load_davis_mapping() if dataset == "DAVIS" else None

    # Load original data
    fpath = f"../utils/{dataset}/"
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding="latin1")

    # Process affinity matrix
    if dataset == "DAVIS":
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    # Get all valid interactions
    rows, cols = np.where(~np.isnan(affinity))

    # Prepare data collection
    data = []
    missing_mappings = set()

    for i in range(len(rows)):
        drug_idx = rows[i]
        prot_idx = cols[i]
        aff = affinity[drug_idx, prot_idx]
        original_target = list(proteins.keys())[prot_idx]

        # Get the final identifier (uniprot)
        if dataset == "DAVIS":
            final_id = davis_mapping.get(original_target, None)
            if final_id is None:
                missing_mappings.add(original_target)
                continue
        else:  # KIBA
            final_id = original_target  # Use target-id as uniprot

        # Set label based on dataset threshold
        label = 1 if (aff > (5 if dataset == "DAVIS" else 12.1)) else 0

        data.append((list(ligands.keys())[drug_idx], final_id, aff, label))

    # Create DataFrame
    columns = ["drug-id", "target-key", "affinity", "Label"]
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    output_file = f"{dataset}_processed.csv"
    df.to_csv(output_file, index=False)

    # Print summary
    print(f"\n=== {dataset} Dataset Processing Summary ===")
    print(f"Output file: {output_file}")
    print(f"Total interactions: {len(df):,}")
    print("\nLabel Distribution:")
    print(df['Label'].value_counts().to_string())
    print(f"\nThreshold used: {5 if dataset == 'DAVIS' else 12.1}")

    if dataset == "DAVIS" and missing_mappings:
        print(f"\nWarning: {len(missing_mappings):,} target IDs had no uniprot mapping")
        print("Sample missing targets:", list(missing_mappings)[:3])


def merge_json_to_csv():
    file1 = f"../utils/DAVIS/"+"proteins.txt"
    file2 = f"../utils/KIBA/" + "proteins.txt"

    with open(file1, 'r') as f1:
        data1 = json.load(f1)

    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    combined_data = {**data1, **data2}

    # 构建CSV行数据（包含标题行）
    rows = [("Pdbid", "Acid_Sequence")]  # 指定列名
    rows.extend([(k, v) for k, v in combined_data.items()])

    # 写入CSV文件
    with open("id_seq.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print(f"\nProcessing completed! ")

def merge_json_to_csv2():
    file1 = f"../utils/DAVIS/"+"ligands_can.txt"
    file2 = f"../utils/KIBA/" + "ligands_can.txt"

    with open(file1, 'r') as f1:
        data1 = json.load(f1)

    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    combined_data = {**data1, **data2}

    # 构建CSV行数据（包含标题行）
    rows = [("pdbid", "SMILES")]  # 指定列名
    rows.extend([(k, v) for k, v in combined_data.items()])

    # 写入CSV文件
    with open("id_smile.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print(f"\nProcessing completed! ")

if __name__ == "__main__":
    datasets = ["DAVIS", "KIBA"]

    print("Starting dataset processing...")
    for dataset in datasets:
        try:
            print(f"\nProcessing {dataset} dataset...")
            process_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue

    print("\nProcessing completed!")

    merge_json_to_csv()
    merge_json_to_csv2()

