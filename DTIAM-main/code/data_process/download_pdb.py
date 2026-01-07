"""
PDB结构文件下载工具
从RCSB PDB数据库下载蛋白质3D结构文件
"""

import os
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time


class PDBDownloader:
    """PDB文件下载器"""
    
    def __init__(self, save_dir: str = "../data/pdb_structures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://files.rcsb.org/download"
        
    def download_pdb(self, pdb_id: str, retry: int = 3) -> bool:
        """
        下载单个PDB文件
        
        Args:
            pdb_id: PDB ID (例如: '1ABC')
            retry: 重试次数
            
        Returns:
            是否下载成功
        """
        # 清理PDB ID - 只取前4个字符作为标准PDB ID
        pdb_id_clean = pdb_id.strip().upper()
        
        # 如果包含括号或其他字符，只取前面的部分
        if '(' in pdb_id_clean:
            pdb_id_clean = pdb_id_clean.split('(')[0]
        
        # PDB ID通常是4个字符
        if len(pdb_id_clean) > 4:
            # 可能是UniProt ID或其他格式，暂时跳过
            print(f"⚠️ 跳过非标准PDB ID: {pdb_id} (可能是UniProt ID)")
            return False
            
        save_path = self.save_dir / f"{pdb_id_clean}.pdb"
        
        # 如果已存在，跳过
        if save_path.exists():
            return True
            
        url = f"{self.base_url}/{pdb_id_clean}.pdb"
        
        for attempt in range(retry):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(save_path, 'w') as f:
                        f.write(response.text)
                    return True
                elif response.status_code == 404:
                    print(f"❌ PDB ID {pdb_id_clean} 不存在于数据库")
                    return False
                else:
                    print(f"⚠️ 下载失败 {pdb_id_clean}: HTTP {response.status_code}")
                    
            except Exception as e:
                if attempt < retry - 1:
                    time.sleep(2)
                    continue
                else:
                    print(f"❌ 下载错误 {pdb_id_clean}: {str(e)}")
                    return False
        
        return False
    
    def download_from_csv(self, csv_path: str, id_column: str = "Pdbid") -> dict:
        """
        从CSV文件读取PDB ID列表并批量下载
        
        Args:
            csv_path: CSV文件路径
            id_column: PDB ID列名
            
        Returns:
            下载统计信息
        """
        print(f"📖 读取CSV文件: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if id_column not in df.columns:
            raise ValueError(f"找不到列 '{id_column}'，可用列: {df.columns.tolist()}")
        
        pdb_ids = df[id_column].dropna().unique().tolist()
        print(f"📊 找到 {len(pdb_ids)} 个唯一的蛋白质ID")
        
        stats = {
            'total': len(pdb_ids),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'failed_ids': []
        }
        
        print(f"⬇️ 开始下载PDB文件到: {self.save_dir}")
        
        for pdb_id in tqdm(pdb_ids, desc="下载PDB结构"):
            result = self.download_pdb(pdb_id)
            if result:
                stats['success'] += 1
            elif len(pdb_id.strip()) > 4:
                stats['skipped'] += 1
            else:
                stats['failed'] += 1
                stats['failed_ids'].append(pdb_id)
            
            # 避免请求过快
            time.sleep(0.2)
        
        return stats
    
    def search_alphafold(self, uniprot_id: str) -> bool:
        """
        从AlphaFold数据库下载预测结构（用于没有实验结构的蛋白质）
        
        Args:
            uniprot_id: UniProt ID
            
        Returns:
            是否下载成功
        """
        save_path = self.save_dir / f"{uniprot_id}_alphafold.pdb"
        
        if save_path.exists():
            return True
        
        # AlphaFold URL格式
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(save_path, 'w') as f:
                    f.write(response.text)
                print(f"✅ 从AlphaFold下载: {uniprot_id}")
                return True
            else:
                return False
        except Exception as e:
            print(f"❌ AlphaFold下载错误 {uniprot_id}: {str(e)}")
            return False


def main():
    """主函数 - 下载所有需要的PDB文件"""
    
    downloader = PDBDownloader()
    
    # 从project1-main下载蛋白质结构
    csv_path = "../project1-main/id_seq.csv"
    
    if os.path.exists(csv_path):
        print("=" * 60)
        print("🧬 开始下载蛋白质PDB结构文件")
        print("=" * 60)
        
        stats = downloader.download_from_csv(csv_path, id_column="Pdbid")
        
        print("\n" + "=" * 60)
        print("📊 下载统计:")
        print(f"  总计: {stats['total']}")
        print(f"  ✅ 成功: {stats['success']}")
        print(f"  ⏭️ 跳过 (非标准ID): {stats['skipped']}")
        print(f"  ❌ 失败: {stats['failed']}")
        
        if stats['failed_ids']:
            print(f"\n失败的PDB ID (前10个):")
            for pdb_id in stats['failed_ids'][:10]:
                print(f"  - {pdb_id}")
                
        print("=" * 60)
        
        # 对于失败的ID，尝试从AlphaFold获取
        if stats['failed_ids']:
            print("\n🔬 尝试从AlphaFold数据库获取预测结构...")
            alphafold_success = 0
            for pdb_id in tqdm(stats['failed_ids'][:20], desc="AlphaFold"):  # 限制前20个
                if downloader.search_alphafold(pdb_id):
                    alphafold_success += 1
                time.sleep(0.5)
            print(f"✅ 从AlphaFold成功获取: {alphafold_success} 个结构")
    else:
        print(f"❌ 找不到文件: {csv_path}")


if __name__ == "__main__":
    main()
