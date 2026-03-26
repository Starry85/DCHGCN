import os
import json
import time
import logging
import random
import pandas as pd
from dotenv import load_dotenv 
import torch
from typing import Dict, List, Tuple
from datetime import datetime
from data_prep import generate_encoding_mapping
from hetero_gnn import DrugDiseaseHeteroGNN

def setup_logging(log_dir: str = "logs"):
   
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"), 
            logging.StreamHandler()  
        ]
    )
    return logging.getLogger(__name__)


def split_k_fold(drug_disease_pairs: List[Tuple], k: int = 5) -> List[List[Tuple]]:
  
    random.shuffle(drug_disease_pairs)
    fold_size = len(drug_disease_pairs) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k-1 else len(drug_disease_pairs)
        folds.append(drug_disease_pairs[start:end])
    return folds


def main(k_folds: int = 5, 
         lr: float = 0.001,
         buffer_size: int = 1000,
         max_pubmed_results: int = 10):
    
   
    load_dotenv()
    pubmed_api_key = os.getenv("PUBMED_API_KEY")
    public_data_sources = {
        "ctd": os.getenv("CTD_API_URL"),
        "drugbank": os.getenv("DRUGBANK_API_URL")
    }
    
    
   
    
    if os.path.exists("encoding_mapping.json"):
        with open("encoding_mapping.json", "r", encoding="utf-8") as f:
            encoding_mapping = json.load(f)
    else:
        encoding_mapping = generate_encoding_mapping("drug_disease_data.xlsx")
    
    
    df = pd.read_excel("drug_disease_data.xlsx", sheet_name="Sheet1")
    drug_disease_pairs = list(df[["drug_name", "Disease"]].drop_duplicates().itertuples(index=False, name=None))
    
   
    folds = split_k_fold(drug_disease_pairs, k=k_folds)
    
  
    evidence_collector = EvidenceCollector(
        db_path="evidence_database.db",
        pubmed_api_key=pubmed_api_key,
        public_data_sources=public_data_sources
    )
   
    model = DrugDiseaseHeteroGNN()
    model.init_embeddings(encoding_mapping)
    
    il_manager = IncrementalLearningManager(
        model=model,
        evidence_collector=evidence_collector,
        encoding_mapping=encoding_mapping,
        learning_rate=lr,
        buffer_size=buffer_size
    )
    
   
    model_version_manager = ModelVersionManager(model_save_dir="model_versions")
    
  
    for fold in range(k_folds):
        fold_start_time = time.time()  
        fold_pairs = folds[fold]
     
        
    results_df = pd.DataFrame(fold_results)
    results_path = f"fold_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    results_df.to_excel(results_path, index=False, engine="openpyxl")
    

if __name__ == "__main__":
  
    main(k_folds=5, lr=0.001, buffer_size=1000)