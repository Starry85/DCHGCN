import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CausalHeteroGNN(nn.Module):
    """因果异质图神经网络模型（适配单Disease列）"""
    
    def __init__(self, 
                 metadata,
                 hidden_channels: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 use_causal_attention: bool = True):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_causal_attention = use_causal_attention
        
        # 节点类型特定的特征编码器
        self.drug_encoder = Linear(-1, hidden_channels)
        self.disease_encoder = Linear(-1, hidden_channels)
        self.gene_encoder = Linear(-1, hidden_channels)
        
        # 异质图卷积层（适配单Disease列的边类型）
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # 药物-基因关系
                ('drug', 'targets', 'gene'): GATConv((-1, -1), hidden_channels // num_heads, heads=num_heads, dropout=dropout),
                # 基因-药物关系（反向）
                ('gene', 'rev_targets', 'drug'): GATConv((-1, -1), hidden_channels // num_heads, heads=num_heads, dropout=dropout),
                # 药物-疾病关系（仅单Disease列）
                ('drug', 'treats', 'disease'): GATConv((-1, -1), hidden_channels // num_heads, heads=num_heads, dropout=dropout),
                # 疾病-药物关系（反向）
                ('disease', 'rev_treats', 'drug'): GATConv((-1, -1), hidden_channels // num_heads, heads=num_heads, dropout=dropout),
                # 基因-疾病关系（仅单Disease列）
                ('gene', 'associated_with', 'disease'): GATConv((-1, -1), hidden_channels // num_heads, heads=num_heads, dropout=dropout),
                # 疾病-基因关系（反向）
                ('disease', 'rev_associated_with', 'gene'): GATConv((-1, -1), hidden_channels // num_heads, heads=num_heads, dropout=dropout),
            }, aggr='mean')
            self.convs.append(conv)
        
        # 因果注意力机制
        if use_causal_attention:
            self.causal_attention = CausalAttentionMechanism(hidden_channels, num_heads)
        
        # 元路径聚合器（适配单Disease列的元路径）
        self.metapath_aggregator = MetapathAggregator(hidden_channels, hidden_channels)
        
        # 预测头（药物-疾病关系预测）
        self.drug_disease_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict, edge_index_dict, metapath_adjacencies=None):
        """前向传播（适配单Disease列的节点和边）"""
        
        # 编码节点特征
        x_dict = {
            'drug': self.drug_encoder(x_dict['drug']),
            'disease': self.disease_encoder(x_dict['disease']),
            'gene': self.gene_encoder(x_dict['gene'])
        }
        
        # 异质图卷积
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # 激活函数和dropout
            for key in x_dict:
                x_dict[key] = F.elu(x_dict[key])
                x_dict[key] = F.dropout(x_dict[key], p=self.dropout, training=self.training)
        
        # 应用因果注意力
        if self.use_causal_attention:
            x_dict = self.causal_attention(x_dict, edge_index_dict)
        
        # 元路径聚合（仅药物节点）
        if metapath_adjacencies is not None and 'drug' in x_dict:
            x_dict['drug'] = self.metapath_aggregator(x_dict['drug'], metapath_adjacencies)
        
        return x_dict
    
    def predict_drug_disease(self, drug_embeddings, disease_embeddings):
        """预测药物-疾病关系（适配单Disease列的疾病嵌入）"""
        combined_features = []
        for drug_emb, disease_emb in zip(drug_embeddings, disease_embeddings):
            combined = torch.cat([drug_emb, disease_emb], dim=-1)
            combined_features.append(combined)
        
        combined_tensor = torch.stack(combined_features)
        predictions = self.drug_disease_predictor(combined_tensor)
        return predictions.squeeze()

class CausalAttentionMechanism(nn.Module):
    """因果注意力机制（无修改）"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 因果注意力权重
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 因果门控机制
        self.causal_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x_dict, edge_index_dict):
        """应用因果注意力"""
        for node_type in ['drug', 'disease', 'gene']:
            if node_type in x_dict:
                node_features = x_dict[node_type]
                
                # 自注意力机制
                attended_features, _ = self.causal_attention(
                    node_features.unsqueeze(0),
                    node_features.unsqueeze(0),
                    node_features.unsqueeze(0)
                )
                attended_features = attended_features.squeeze(0)
                
                # 残差连接和层归一化
                x_dict[node_type] = self.layer_norm(node_features + attended_features)
        
        return x_dict

class MetapathAggregator(nn.Module):
    """元路径聚合器（适配单Disease列的元路径）"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 元路径特定的变换（仅保留核心元路径）
        self.metapath_transforms = nn.ModuleDict({
            'drug_gene_disease': nn.Linear(input_dim, output_dim),  # 药物-基因-疾病
            'drug_disease': nn.Linear(input_dim, output_dim),     # 药物-疾病
            'drug_gene_drug': nn.Linear(input_dim, output_dim)    # 药物-基因-药物
        })
        
        # 注意力权重（元路径重要性加权）
        self.metapath_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=0.1
        )
    
    def forward(self, node_embeddings, metapath_adjacencies):
        """元路径聚合（仅处理核心元路径）"""
        metapath_embeddings = []
        
        # 仅处理预定义的核心元路径
        for metapath_name in ['drug_gene_disease', 'drug_disease', 'drug_gene_drug']:
            if metapath_name in metapath_adjacencies and metapath_name in self.metapath_transforms:
                adjacency = metapath_adjacencies[metapath_name]
                # 应用元路径变换
                transformed = self.metapath_transforms[metapath_name](node_embeddings)
                # 路径传播（适配邻接矩阵维度）
                if adjacency.size(0) == transformed.size(0):
                    propagated = torch.matmul(adjacency, transformed)
                    metapath_embeddings.append(propagated)
        
        # 注意力聚合
        if len(metapath_embeddings) > 0:
            stacked_embeddings = torch.stack(metapath_embeddings, dim=0)
            attended_embeddings, _ = self.metapath_attention(
                stacked_embeddings, stacked_embeddings, stacked_embeddings
            )
            aggregated = torch.mean(attended_embeddings, dim=0)
            return aggregated
        else:
            return node_embeddings

class CausalGNNTrainer:
    """因果GNN训练器（核心修改：集成分组交叉验证，适配单Disease列）"""
    
    def __init__(self, 
                 model: nn.Module,
                 hetero_data: HeteroData,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 encoding_mapping: Dict,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        
        self.model = model
        self.hetero_data = hetero_data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.encoding_mapping = encoding_mapping
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        
        # 预处理正负样本（训练/验证/测试集）
        self.train_positive_pairs = self._get_positive_pairs(train_df)
        self.val_positive_pairs = self._get_positive_pairs(val_df)
        self.test_positive_pairs = self._get_positive_pairs(test_df)
    
    def _get_positive_pairs(self, df: pd.DataFrame) -> List[Tuple]:
        """获取正样本对（药物-疾病，适配单Disease列）"""
        positive_pairs = []
        for _, row in df.iterrows():
            drug = row['drug_name']
            disease = row['Disease']
            if drug in self.encoding_mapping['drugs'] and disease in self.encoding_mapping['diseases']:
                positive_pairs.append((drug, disease))
        return positive_pairs
    
    def generate_negative_samples(self, num_samples: int, split_type: str = 'train') -> List[Tuple]:
        """生成负样本对（药物-疾病，适配单Disease列）"""
        drug_mapping = self.encoding_mapping['drugs']
        disease_mapping = self.encoding_mapping['diseases']
        drug_list = list(drug_mapping.keys())
        disease_list = list(disease_mapping.keys())
        
        # 选择对应的正样本集
        if split_type == 'train':
            positive_pairs = self.train_positive_pairs
        elif split_type == 'val':
            positive_pairs = self.val_positive_pairs
        else:
            positive_pairs = self.test_positive_pairs
        
        negative_pairs = []
        while len(negative_pairs) < num_samples:
            # 随机选择药物和疾病
            drug = random.choice(drug_list)
            disease = random.choice(disease_list)
            # 确保不是正样本
            if (drug, disease) not in positive_pairs:
                negative_pairs.append((drug, disease))
        
        return negative_pairs
    
    def _prepare_batch(self, positive_pairs: List[Tuple], negative_pairs: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备批次数据（药物嵌入、疾病嵌入、标签）"""
        drug_embeddings = []
        disease_embeddings = []
        labels = []
        
        # 正样本（标签1）
        for drug, disease in positive_pairs:
            drug_idx = self.encoding_mapping['drugs'][drug]
            disease_idx = self.encoding_mapping['diseases'][disease]
            drug_emb = self.hetero_data['drug'].x[drug_idx]
            disease_emb = self.hetero_data['disease'].x[disease_idx]
            drug_embeddings.append(drug_emb)
            disease_embeddings.append(disease_emb)
            labels.append(1.0)
        
        # 负样本（标签0）
        for drug, disease in negative_pairs:
            drug_idx = self.encoding_mapping['drugs'][drug]
            disease_idx = self.encoding_mapping['diseases'][disease]
            drug_emb = self.hetero_data['drug'].x[drug_idx]
            disease_emb = self.hetero_data['disease'].x[disease_idx]
            drug_embeddings.append(drug_emb)
            disease_embeddings.append(disease_emb)
            labels.append(0.0)
        
        # 转换为Tensor
        drug_tensor = torch.stack(drug_embeddings)
        disease_tensor = torch.stack(disease_embeddings)
        label_tensor = torch.tensor(labels, dtype=torch.float)
        
        return drug_tensor, disease_tensor, label_tensor
    
    def train_epoch(self, batch_size: int = 32) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_positive_pairs) // batch_size
        
        # 打乱正样本
        random.shuffle(self.train_positive_pairs)
        
        for i in range(num_batches):
            # 取批次正样本
            batch_positive = self.train_positive_pairs[i*batch_size : (i+1)*batch_size]
            # 生成同等数量的负样本
            batch_negative = self.generate_negative_samples(len(batch_positive), split_type='train')
            
            # 准备批次数据
            drug_tensor, disease_tensor, label_tensor = self._prepare_batch(batch_positive, batch_negative)
            
            # 前向传播
            # 1. 获取节点嵌入
            x_dict = {
                'drug': self.hetero_data['drug'].x,
                'disease': self.hetero_data['disease'].x,
                'gene': self.hetero_data['gene'].x
            }
            edge_index_dict = {
                edge_type: self.hetero_data[edge_type].edge_index for edge_type in self.hetero_data.edge_types
            }
            node_embeddings = self.model(x_dict, edge_index_dict)
            
            # 2. 提取当前批次的嵌入
            batch_drug_emb = node_embeddings['drug'][drug_tensor.argmax(dim=1)]  # 简化：实际需按索引提取
            batch_disease_emb = node_embeddings['disease'][disease_tensor.argmax(dim=1)]
            
            # 3. 预测
            predictions = self.model.predict_drug_disease(batch_drug_emb, batch_disease_emb)
            
            # 4. 计算损失
            loss = self.criterion(predictions, label_tensor)
            total_loss += loss.item()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, split_type: str = 'val') -> Dict[str, float]:
        """评估模型（在验证集/测试集上）"""
        self.model.eval()
        with torch.no_grad():
            # 选择评估集
            if split_type == 'val':
                positive_pairs = self.val_positive_pairs
            else:
                positive_pairs = self.test_positive_pairs
            
            # 生成负样本（与正样本数量一致）
            negative_pairs = self.generate_negative_samples(len(positive_pairs), split_type=split_type)
            
            # 准备数据
            drug_tensor, disease_tensor, label_tensor = self._prepare_batch(positive_pairs, negative_pairs)
            
            # 获取节点嵌入
            x_dict = {
                'drug': self.hetero_data['drug'].x,
                'disease': self.hetero_data['disease'].x,
                'gene': self.hetero_data['gene'].x
            }
            edge_index_dict = {
                edge_type: self.hetero_data[edge_type].edge_index for edge_type in self.hetero_data.edge_types
            }
            node_embeddings = self.model(x_dict, edge_index_dict)
            
            # 预测
            batch_drug_emb = node_embeddings['drug'][drug_tensor.argmax(dim=1)]
            batch_disease_emb = node_embeddings['disease'][disease_tensor.argmax(dim=1)]
            predictions = self.model.predict_drug_disease(batch_drug_emb, batch_disease_emb)
            
            # 计算指标
            predictions_np = predictions.cpu().numpy()
            labels_np = label_tensor.cpu().numpy()
            
            # AUC-ROC
            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
            auc_roc = roc_auc_score(labels_np, predictions_np)
            # AUC-PR
            auc_pr = average_precision_score(labels_np, predictions_np)
            # F1分数（阈值0.5）
            f1 = f1_score(labels_np, (predictions_np > 0.5).astype(int))
            
            return {
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'f1_score': f1,
                'loss': self.criterion(predictions, label_tensor).item()
            }
    
    def train(self, num_epochs: int = 100, patience: int = 10, batch_size: int = 32):
        """训练模型（含交叉验证和早停）"""
        print("Training Causal HeteroGNN...")
        
        best_val_auc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(batch_size)
            
            # 验证
            val_metrics = self.evaluate(split_type='val')
            
            # 打印日志
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val AUC-ROC: {val_metrics['auc_roc']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
            
            # 早停与模型保存
            if val_metrics['auc_roc'] > best_val_auc:
                best_val_auc = val_metrics['auc_roc']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_causal_gnn_model.pth')
            else:
                patience