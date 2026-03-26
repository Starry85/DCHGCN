import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import networkx as nx
from collections import defaultdict
import requests
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DrugRepurposingDataPreprocessor:
    """药物重定位数据预处理和多模态特征工程（支持多数据集加载+分组划分）"""
    
    def __init__(self, main_data_path: str, public_data_paths: Optional[Dict] = None):
        self.main_data_path = main_data_path
        self.public_data_paths = public_data_paths  # 存储公开数据集路径：{'drugbank': path, 'disgenet': path, 'ctd': path}
        self.drug_encoder = LabelEncoder()
        self.disease_encoder = LabelEncoder()
        self.gene_encoder = LabelEncoder()
        self.atc_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # 多模态特征容器
        self.drug_features = {}
        self.disease_features = {}
        self.gene_features = {}
        self.edge_data = defaultdict(list)
    
    def load_public_datasets(self) -> Dict[str, pd.DataFrame]:
        """加载常用公开数据集（DrugBank、DisGeNET、CTD）"""
        public_dfs = {}
        if not self.public_data_paths:
            return public_dfs
        
        # 1. 加载DrugBank（药物-基因靶点数据）
        if 'drugbank' in self.public_data_paths:
            drugbank_df = pd.read_csv(self.public_data_paths['drugbank'], sep='\t')
            drugbank_df = drugbank_df[['Name', 'Gene_Symbol', 'ATC_Code', 'Description']].rename(
                columns={'Name': 'drug_name', 'Gene_Symbol': 'gene_name', 'ATC_Code': 'atc_codes', 'Description': 'description'}
            )
            drugbank_df['indication'] = 'approved'  # DrugBank药物默认已批准
            drugbank_df['Disease'] = 'Unknown'  # 后续与DisGeNET关联补充疾病
            public_dfs['drugbank'] = drugbank_df
        
        # 2. 加载DisGeNET（疾病-基因关联数据）
        if 'disgenet' in self.public_data_paths:
            disgenet_df = pd.read_csv(self.public_data_paths['disgenet'], sep=';')
            disgenet_df = disgenet_df[['GeneSymbol', 'DiseaseName', 'DiseaseType']].rename(
                columns={'GeneSymbol': 'gene_name', 'DiseaseName': 'Disease', 'DiseaseType': 'disease_type'}
            )
            disgenet_df['drug_name'] = 'Unknown'  # 后续与DrugBank关联补充药物
            disgenet_df['indication'] = 'under investigation'
            public_dfs['disgenet'] = disgenet_df
        
        # 3. 加载CTD（药物-疾病-基因关联数据）
        if 'ctd' in self.public_data_paths:
            ctd_df = pd.read_csv(self.public_data_paths['ctd'], sep='\t')
            ctd_df = ctd_df[['ChemicalName', 'GeneSymbol', 'DiseaseName', 'InteractionType']].rename(
                columns={'ChemicalName': 'drug_name', 'GeneSymbol': 'gene_name', 'DiseaseName': 'Disease', 'InteractionType': 'indication'}
            )
            ctd_df['description'] = 'CTD public data'
            ctd_df['atc_codes'] = 'UNKNOWN'
            public_dfs['ctd'] = ctd_df
        
        return public_dfs
    
    def merge_datasets(self, main_df: pd.DataFrame, public_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并主数据集与公开数据集"""
        merged_df = main_df.copy()
        
        # 合并DrugBank（补充药物-基因-ATC数据）
        if 'drugbank' in public_dfs:
            drugbank_df = public_dfs['drugbank']
            merged_df = pd.merge(
                merged_df, 
                drugbank_df[['drug_name', 'gene_name', 'atc_codes', 'description']],
                on=['drug_name', 'gene_name'],
                how='outer',
                suffixes=('', '_db')
            )
            # 填充缺失值（主数据集优先）
            merged_df['description'] = merged_df['description'].fillna(merged_df['description_db'])
            merged_df['atc_codes'] = merged_df['atc_codes'].fillna(merged_df['atc_codes_db'])
            merged_df = merged_df.drop(columns=['description_db', 'atc_codes_db'])
        
        # 合并DisGeNET（补充疾病-基因数据）
        if 'disgenet' in public_dfs:
            disgenet_df = public_dfs['disgenet']
            merged_df = pd.merge(
                merged_df,
                disgenet_df[['gene_name', 'Disease', 'disease_type']],
                on=['gene_name', 'Disease'],
                how='outer',
                suffixes=('', '_dg')
            )
            merged_df = merged_df.drop(columns=['disease_type_dg'])
        
        # 合并CTD（补充药物-疾病关联）
        if 'ctd' in public_dfs:
            ctd_df = public_dfs['ctd']
            merged_df = pd.merge(
                merged_df,
                ctd_df[['drug_name', 'gene_name', 'Disease', 'indication']],
                on=['drug_name', 'gene_name', 'Disease'],
                how='outer',
                suffixes=('', '_ctd')
            )
            merged_df['indication'] = merged_df['indication'].fillna(merged_df['indication_ctd'])
            merged_df = merged_df.drop(columns=['indication_ctd'])
        
        return merged_df.drop_duplicates()
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """加载和清洗数据（删除Disease1/Disease2，仅保留Disease列）"""
        print("Loading and cleaning data...")
        
        # 1. 加载主数据集（用户自己的数据集）
        main_df = pd.read_excel(self.main_data_path, sheet_name='1019-drug_disease_filled')
        # 重命名列（确保只有Disease列，删除Disease1/Disease2）
        if 'Disease1' in main_df.columns or 'Disease2' in main_df.columns:
            # 若存在旧列，合并后删除（用户已修改为单Disease列，此处兼容旧数据）
            if 'Disease' not in main_df.columns:
                main_df['Disease'] = main_df['Disease1'].fillna(main_df['Disease2'])
            main_df = main_df.drop(columns=['Disease1', 'Disease2'], errors='ignore')
        
        # 2. 加载并合并公开数据集
        public_dfs = self.load_public_datasets()
        if public_dfs:
            main_df = self.merge_datasets(main_df, public_dfs)
        
        # 3. 数据清洗
        df_clean = main_df.copy()
        # 处理缺失值（核心列不可缺失）
        df_clean = df_clean.dropna(subset=['drug_name', 'gene_name', 'Disease'])
        # 去重
        df_clean = df_clean.drop_duplicates(subset=['drug_name', 'gene_name', 'Disease'])
        # 标准化文本格式
        df_clean['drug_name'] = df_clean['drug_name'].str.strip().str.lower()
        df_clean['gene_name'] = df_clean['gene_name'].str.strip().str.upper()
        df_clean['Disease'] = df_clean['Disease'].str.strip().str.title()
        # 填充ATC代码和描述
        df_clean['atc_codes'] = df_clean['atc_codes'].fillna('UNKNOWN')
        df_clean['description'] = df_clean['description'].fillna('No description available')
        # 标准化indication
        df_clean['indication'] = df_clean['indication'].str.strip().str.lower()
        df_clean['indication'] = df_clean['indication'].fillna('experimental')
        
        print(f"Cleaned data shape: {df_clean.shape} (merged main + public datasets)")
        return df_clean
    
    def grouped_train_test_split(self, df: pd.DataFrame, group_col: str = 'drug_name', test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按药物/疾病分组划分训练集、验证集、测试集（避免数据泄露）"""
        print(f"Grouped split by {group_col}, test_size={test_size}, val_size={val_size}")
        
        # 按分组列获取唯一ID
        unique_groups = df[group_col].unique()
        # 划分训练集和测试集（先分测试集）
        train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=42)
        # 从训练集中划分验证集
        train_groups, val_groups = train_test_split(train_groups, test_size=val_size/(1-test_size), random_state=42)
        
        # 按分组筛选数据
        train_df = df[df[group_col].isin(train_groups)]
        val_df = df[df[group_col].isin(val_groups)]
        test_df = df[df[group_col].isin(test_groups)]
        
        print(f"Split result: Train={train_df.shape[0]}, Val={val_df.shape[0]}, Test={test_df.shape[0]}")
        return train_df, val_df, test_df
    
    def extract_drug_features(self, df: pd.DataFrame) -> Dict:
        """提取药物多模态特征（适配单Disease列）"""
        print("Extracting drug features...")
        
        drug_features = {}
        indication_mapping = {
            'approved': 2,
            'undergoing clinical investigation': 1,
            'under investigation': 1,
            'experimental': 0
        }
        
        for drug in df['drug_name'].unique():
            drug_data = df[df['drug_name'] == drug]
            
            # 1. 基因靶点特征
            target_genes = drug_data['gene_name'].unique()
            gene_count = len(target_genes)
            
            # 2. 治疗领域特征（基于indication）
            indications = drug_data['indication'].unique()
            max_indication_score = max([indication_mapping.get(ind, 0) for ind in indications])
            
            # 3. ATC代码特征
            atc_codes = drug_data['atc_codes'].unique()
            
            # 4. 疾病关联特征（仅单Disease列）
            disease_count = len(drug_data['Disease'].unique())
            
            # 5. 分子类型特征（从描述提取）
            description = drug_data['description'].iloc[0].lower()
            molecule_type = 0  # 小分子
            if any(word in description for word in ['biotech', 'recombinant', 'fusion', 'antibody']):
                molecule_type = 1  # 生物药
            
            # 组合特征向量
            feature_vector = np.array([
                gene_count, max_indication_score, len(atc_codes), disease_count, molecule_type
            ], dtype=np.float32)
            drug_features[drug] = feature_vector
        
        # 标准化特征
        feature_matrix = np.array(list(drug_features.values()))
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        for i, drug in enumerate(drug_features.keys()):
            drug_features[drug] = feature_matrix_scaled[i]
        
        self.drug_features = drug_features
        return drug_features
    
    def extract_disease_features(self, df: pd.DataFrame) -> Dict:
        """提取疾病多模态特征（适配单Disease列）"""
        print("Extracting disease features...")
        
        disease_features = {}
        all_diseases = df['Disease'].unique()
        
        for disease in all_diseases:
            disease_data = df[df['Disease'] == disease]
            if len(disease_data) == 0:
                continue
            
            # 1. 关联药物数量
            drug_count = len(disease_data['drug_name'].unique())
            # 2. 关联基因数量
            gene_count = len(disease_data['gene_name'].unique())
            # 3. 疾病出现次数
            disease_count = len(disease_data)
            # 4. 治疗阶段分布（approved比例）
            indication_counts = disease_data['indication'].value_counts()
            approved_ratio = indication_counts.get('approved', 0) / len(disease_data)
            # 5. 疾病类型特征（从公开数据补充，无则默认0）
            disease_type = 0  # 0=常见疾病，1=罕见病
            if 'disease_type' in disease_data.columns:
                disease_type = 1 if 'rare' in disease_data['disease_type'].iloc[0].lower() else 0
            
            # 组合特征向量
            feature_vector = np.array([
                drug_count, gene_count, disease_count, approved_ratio, disease_type
            ], dtype=np.float32)
            disease_features[disease] = feature_vector
        
        self.disease_features = disease_features
        return disease_features
    
    def extract_gene_features(self, df: pd.DataFrame) -> Dict:
        """提取基因多模态特征（适配单Disease列）"""
        print("Extracting gene features...")
        
        gene_features = {}
        for gene in df['gene_name'].unique():
            gene_data = df[df['gene_name'] == gene]
            
            # 1. 关联药物数量
            drug_count = len(gene_data['drug_name'].unique())
            # 2. 关联疾病数量（单Disease列）
            disease_count = len(gene_data['Disease'].unique())
            # 3. 治疗阶段分布
            indication_counts = gene_data['indication'].value_counts()
            experimental_ratio = indication_counts.get('experimental', 0) / len(gene_data)
            approved_ratio = indication_counts.get('approved', 0) / len(gene_data)
            # 4. 药物类型分布（生物药比例）
            descriptions = gene_data['description'].str.lower()
            biotech_ratio = descriptions.str.contains('biotech|antibody').mean()
            
            # 组合特征向量
            feature_vector = np.array([
                drug_count, disease_count, experimental_ratio, approved_ratio, biotech_ratio
            ], dtype=np.float32)
            gene_features[gene] = feature_vector
        
        self.gene_features = gene_features
        return gene_features
    
    def build_multimodal_features(self, df: pd.DataFrame) -> Dict:
        """构建多模态特征矩阵（适配修改后的数据）"""
        print("Building multimodal features...")
        self.extract_drug_features(df)
        self.extract_disease_features(df)
        self.extract_gene_features(df)
        
        multimodal_features = {
            'drugs': self.drug_features,
            'diseases': self.disease_features,
            'genes': self.gene_features
        }
        return multimodal_features
    
    def create_heterogeneous_edges(self, df: pd.DataFrame) -> Dict:
        """构建异质图边关系（适配单Disease列）"""
        print("Creating heterogeneous edges...")
        
        edges = defaultdict(list)
        for _, row in df.iterrows():
            drug = row['drug_name']
            gene = row['gene_name']
            disease = row['Disease']
            
            # 药物-基因边
            edges[('drug', 'targets', 'gene')].append((drug, gene))
            # 药物-疾病边
            edges[('drug', 'treats', 'disease')].append((drug, disease))
            # 基因-疾病边
            edges[('gene', 'associated_with', 'disease')].append((gene, disease))
        
        self.edge_data = edges
        return edges
    
    def encode_entities(self, df: pd.DataFrame) -> Dict:
        """对实体进行编码（适配单Disease列）"""
        print("Encoding entities...")
        
        # 编码药物
        drugs = df['drug_name'].unique()
        drug_mapping = {drug: idx for idx, drug in enumerate(drugs)}
        # 编码疾病（仅单Disease列）
        diseases = df['Disease'].unique()
        disease_mapping = {disease: idx for idx, disease in enumerate(diseases)}
        # 编码基因
        genes = df['gene_name'].unique()
        gene_mapping = {gene: idx for idx, gene in enumerate(genes)}
        
        encoding_mapping = {
            'drugs': drug_mapping,
            'diseases': disease_mapping,
            'genes': gene_mapping
        }
        return encoding_mapping
    
    def prepare_training_data(self, df: pd.DataFrame, split_group: str = 'drug_name') -> Dict:
        """准备训练数据（含分组划分）"""
        print("Preparing training data...")
        
        # 1. 分组划分数据集
        train_df, val_df, test_df = self.grouped_train_test_split(df, group_col=split_group)
        
        # 2. 构建多模态特征
        multimodal_features = self.build_multimodal_features(df)  # 用全量数据的特征编码器
        
        # 3. 构建边关系（用训练集构建边，避免测试集信息泄露）
        edges = self.create_heterogeneous_edges(train_df)
        
        # 4. 编码实体
        encoding_mapping = self.encode_entities(df)
        
        # 5. 准备正负样本对（仅训练集）
        positive_pairs = [(row['drug_name'], row['Disease']) for _, row in train_df.iterrows()]
        
        training_data = {
            'multimodal_features': multimodal_features,
            'edges': edges,
            'encoding_mapping': encoding_mapping,
            'positive_pairs': positive_pairs,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'drug_encoder': self.drug_encoder,
            'disease_encoder': self.disease_encoder,
            'gene_encoder': self.gene_encoder,
            'scaler': self.scaler
        }
        return training_data