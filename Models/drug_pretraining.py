import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.utils import negative_sampling
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
import random

class DrugPretrainingModel(nn.Module):
    """药物预训练表示学习模型（适配模块1的药物特征）"""
    
    def __init__(self, 
                 drug_feature_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super(DrugPretrainingModel, self).__init__()
        
        self.drug_feature_dim = drug_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 药物特征编码器
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 图注意力层
        self.gat_conv1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.gat_conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        
        # Transformer层用于捕获全局依赖
        self.transformer_conv = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        # 边预测头
        self.edge_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index):
        """前向传播"""
        # 药物特征编码
        x_encoded = self.drug_encoder(x)
        
        # 图注意力编码
        x_gat1 = F.elu(self.gat_conv1(x_encoded, edge_index))
        x_gat1 = F.dropout(x_gat1, p=self.dropout, training=self.training)
        x_gat2 = self.gat_conv2(x_gat1, edge_index)
        
        # Transformer编码
        x_transformer = self.transformer_conv(x_gat2, edge_index)
        
        # 残差连接
        x_final = x_gat2 + x_transformer
        
        # 输出投影
        drug_embeddings = self.output_projection(x_final)
        return drug_embeddings
    
    def predict_edge(self, drug_embeddings, edge_index):
        """预测边存在概率"""
        src, dst = edge_index
        src_emb = drug_embeddings[src]
        dst_emb = drug_embeddings[dst]
        
        edge_features = torch.cat([src_emb, dst_emb], dim=1)
        edge_probs = self.edge_predictor(edge_features)
        return edge_probs.squeeze()

class DDIPretrainer:
    """药物-药物相互作用预训练器（适配模块1的多数据集药物特征）"""
    
    def __init__(self, 
                 drug_features: Dict,
                 drug_mapping: Dict,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 learning_rate: float = 0.001):
        
        self.drug_features = drug_features
        self.drug_mapping = drug_mapping
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # 构建药物图数据（基于多数据集的药物特征）
        self.drug_graph_data = self._build_drug_graph()
        
        # 初始化模型
        feature_dim = len(list(drug_features.values())[0])
        self.model = DrugPretrainingModel(
            drug_feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
    
    def _build_drug_graph(self) -> Data:
        """构建药物图数据（基于多数据集的药物特征相似性）"""
        print("Building drug interaction graph...")
        
        drug_list = list(self.drug_mapping.keys())
        num_drugs = len(drug_list)
        
        # 构建特征矩阵
        feature_matrix = []
        for drug in drug_list:
            feature_matrix.append(self.drug_features[drug])
        x = torch.tensor(feature_matrix, dtype=torch.float)
        
        # 构建边索引（基于余弦相似性）
        edge_list = []
        similarity_threshold = 0.7
        for i in range(num_drugs):
            for j in range(i + 1, num_drugs):
                vec_i = x[i]
                vec_j = x[j]
                similarity = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0))
                if similarity > similarity_threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # 无向图
        
        # 兜底：无足够边时添加随机连接
        if len(edge_list) == 0:
            for i in range(min(num_drugs, 10)):
                for j in range(i + 1, min(num_drugs, i + 5)):
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    
    def generate_negative_edges(self, num_negative_samples: int) -> torch.Tensor:
        """生成负样本边"""
        num_drugs = self.drug_graph_data.num_nodes
        positive_edges = self.drug_graph_data.edge_index
        negative_edges = negative_sampling(
            edge_index=positive_edges,
            num_nodes=num_drugs,
            num_neg_samples=num_negative_samples
        )
        return negative_edges
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        # 正样本
        positive_edges = self.drug_graph_data.edge_index
        positive_labels = torch.ones(positive_edges.size(1), dtype=torch.float)
        
        # 负样本（与正样本数量一致）
        negative_edges = self.generate_negative_edges(positive_edges.size(1))
        negative_labels = torch.zeros(negative_edges.size(1), dtype=torch.float)
        
        # 合并样本
        all_edges = torch.cat([positive_edges, negative_edges], dim=1)
        all_labels = torch.cat([positive_labels, negative_labels])
        
        # 前向传播
        drug_embeddings = self.model(self.drug_graph_data.x, self.drug_graph_data.edge_index)
        predictions = self.model.predict_edge(drug_embeddings, all_edges)
        
        # 计算损失
        loss = self.criterion(predictions, all_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def pretrain(self, num_epochs: int = 100, patience: int = 10):
        """预训练模型"""
        print("Starting drug pretraining...")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
            
            # 早停与模型保存
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_drug_pretrain_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_drug_pretrain_model.pth'))
        print("Drug pretraining completed!")
    
    def get_drug_embeddings(self) -> Dict:
        """获取药物嵌入表示（适配多数据集药物）"""
        self.model.eval()
        with torch.no_grad():
            drug_embeddings = self.model(self.drug_graph_data.x, self.drug_graph_data.edge_index)
        
        # 映射回药物名称
        drug_embedding_dict = {}
        drug_list = list(self.drug_mapping.keys())
        for i, drug in enumerate(drug_list):
            drug_embedding_dict[drug] = drug_embeddings[i].cpu().numpy()
        
        return drug_embedding_dict

class FunctionalSimilarityCalculator:
    """功能性相似性计算器（无修改）"""
    
    def __init__(self, drug_embeddings: Dict):
        self.drug_embeddings = drug_embeddings
    
    def calculate_functional_similarity(self, drug1: str, drug2: str) -> float:
        if drug1 not in self.drug_embeddings or drug2 not in self.drug_embeddings:
            return 0.0
        
        emb1 = torch.tensor(self.drug_embeddings[drug1])
        emb2 = torch.tensor(self.drug_embeddings[drug2])
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return similarity.item()
    
    def get_most_similar_drugs(self, target_drug: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if target_drug not in self.drug_embeddings:
            return []
        
        similarities = []
        target_embedding = torch.tensor(self.drug_embeddings[target_drug])
        for drug, embedding in self.drug_embeddings.items():
            if drug == target_drug:
                continue
            drug_embedding = torch.tensor(embedding)
            similarity = F.cosine_similarity(target_embedding.unsqueeze(0), drug_embedding.unsqueeze(0))
            similarities.append((drug, similarity.item()))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]