import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import random

class CausalHeterogeneousGraphBuilder:
    """因果异质图构建器（适配单Disease列）"""
    
    def __init__(self, 
                 multimodal_features: Dict,
                 edges: Dict,
                 encoding_mapping: Dict,
                 drug_embeddings: Dict):
        
        self.multimodal_features = multimodal_features
        self.edges = edges
        self.encoding_mapping = encoding_mapping
        self.drug_embeddings = drug_embeddings
        
        # 因果图结构
        self.causal_graph = None
        self.hetero_data = None
    
    def build_heterogeneous_graph(self) -> HeteroData:
        """构建异质图（适配单Disease列）"""
        print("Building heterogeneous graph...")
        
        data = HeteroData()
        
        # 添加节点特征
        self._add_node_features(data)
        
        # 添加边（仅单Disease列相关边）
        self._add_edges(data)
        
        # 添加元路径
        self._add_metapaths(data)
        
        self.hetero_data = data
        return data
    
    def _add_node_features(self, data: HeteroData):
        """添加节点特征（适配单Disease列的疾病特征）"""
        
        # 1. 药物节点特征（用预训练嵌入）
        drug_features = []
        drug_mapping = self.encoding_mapping['drugs']
        for drug, idx in drug_mapping.items():
            if drug in self.drug_embeddings:
                feature = self.drug_embeddings[drug]
            else:
                feature = self.multimodal_features['drugs'].get(drug, np.zeros(64))
            drug_features.append(feature)
        data['drug'].x = torch.tensor(np.array(drug_features), dtype=torch.float)
        
        # 2. 疾病节点特征（仅单Disease列的疾病）
        disease_features = []
        disease_mapping = self.encoding_mapping['diseases']
        for disease, idx in disease_mapping.items():
            feature = self.multimodal_features['diseases'].get(disease, np.zeros(5))
            disease_features.append(feature)
        data['disease'].x = torch.tensor(np.array(disease_features), dtype=torch.float)
        
        # 3. 基因节点特征
        gene_features = []
        gene_mapping = self.encoding_mapping['genes']
        for gene, idx in gene_mapping.items():
            feature = self.multimodal_features['genes'].get(gene, np.zeros(5))
            gene_features.append(feature)
        data['gene'].x = torch.tensor(np.array(gene_features), dtype=torch.float)
    
    def _add_edges(self, data: HeteroData):
        """添加边关系（删除Disease2相关边，仅保留单Disease列的边）"""
        
        # 1. 药物-基因边 (targets)
        if ('drug', 'targets', 'gene') in self.edges:
            drug_gene_edges = self.edges[('drug', 'targets', 'gene')]
            drug_indices = []
            gene_indices = []
            for drug, gene in drug_gene_edges:
                if drug in self.encoding_mapping['drugs'] and gene in self.encoding_mapping['genes']:
                    drug_indices.append(self.encoding_mapping['drugs'][drug])
                    gene_indices.append(self.encoding_mapping['genes'][gene])
            data['drug', 'targets', 'gene'].edge_index = torch.tensor(
                [drug_indices, gene_indices], dtype=torch.long
            )
        
        # 2. 药物-疾病边 (treats)（仅单Disease列）
        if ('drug', 'treats', 'disease') in self.edges:
            drug_disease_edges = self.edges[('drug', 'treats', 'disease')]
            drug_indices = []
            disease_indices = []
            for drug, disease in drug_disease_edges:
                if drug in self.encoding_mapping['drugs'] and disease in self.encoding_mapping['diseases']:
                    drug_indices.append(self.encoding_mapping['drugs'][drug])
                    disease_indices.append(self.encoding_mapping['diseases'][disease])
            data['drug', 'treats', 'disease'].edge_index = torch.tensor(
                [drug_indices, disease_indices], dtype=torch.long
            )
        
        # 3. 基因-疾病边 (associated_with)（仅单Disease列）
        if ('gene', 'associated_with', 'disease') in self.edges:
            gene_disease_edges = self.edges[('gene', 'associated_with', 'disease')]
            gene_indices = []
            disease_indices = []
            for gene, disease in gene_disease_edges:
                if gene in self.encoding_mapping['genes'] and disease in self.encoding_mapping['diseases']:
                    gene_indices.append(self.encoding_mapping['genes'][gene])
                    disease_indices.append(self.encoding_mapping['diseases'][disease])
            data['gene', 'associated_with', 'disease'].edge_index = torch.tensor(
                [gene_indices, disease_indices], dtype=torch.long
            )
    
    def _add_metapaths(self, data: HeteroData):
        """添加元路径信息（适配单Disease列的路径）"""
        print("Adding metapaths...")
        
        # 定义核心元路径（基于单Disease列）
        metapaths = {
            'drug_gene_disease': [('drug', 'targets', 'gene'), ('gene', 'associated_with', 'disease')],  # 药物-基因-疾病
            'drug_disease': [('drug', 'treats', 'disease')],  # 药物-疾病（直接）
            'drug_gene_drug': [('drug', 'targets', 'gene'), ('gene', 'targets', 'drug')]  # 药物-基因-药物（共享靶点）
        }
        
        # 计算元路径邻接矩阵
        self.metapath_adjacencies = {}
        for metapath_name, edge_types in metapaths.items():
            adjacency = self._compute_metapath_adjacency(data, edge_types)
            self.metapath_adjacencies[metapath_name] = adjacency
    
    def _compute_metapath_adjacency(self, data: HeteroData, edge_types: List[Tuple]) -> torch.Tensor:
        """计算元路径邻接矩阵"""
        # 针对药物节点的元路径邻接矩阵
        if edge_types[0][0] == 'drug' and edge_types[-1][-1] == 'drug':
            num_drugs = data['drug'].x.size(0)
            return torch.eye(num_drugs)  # 示例：单位矩阵，实际需计算真实路径
        # 针对药物-疾病的元路径邻接矩阵
        elif edge_types[0][0] == 'drug' and edge_types[-1][-1] == 'disease':
            num_drugs = data['drug'].x.size(0)
            num_diseases = data['disease'].x.size(0)
            return torch.zeros(num_drugs, num_diseases)  # 示例：零矩阵，实际需计算真实路径
        else:
            num_nodes = data[edge_types[0][0]].x.size(0)
            return torch.eye(num_nodes)
    
    def apply_causal_inference(self, treatment: str, outcome: str) -> float:
        """应用因果推断计算干预效果（适配单Disease列的疾病节点）"""
        print(f"Applying causal inference: {treatment} -> {outcome}")
        
        # 构建因果图
        self._build_causal_graph()
        
        # 计算平均因果效应
        ate = self._calculate_average_treatment_effect(treatment, outcome)
        return ate
    
    def _build_causal_graph(self):
        """构建因果图（适配单Disease列的疾病节点）"""
        if self.causal_graph is not None:
            return
        
        self.causal_graph = nx.DiGraph()
        
        # 添加节点（药物、疾病、基因）
        for node_type in ['drug', 'disease', 'gene']:
            if node_type in self.hetero_data:
                num_nodes = self.hetero_data[node_type].x.size(0)
                for i in range(num_nodes):
                    node_id = f"{node_type}_{i}"
                    self.causal_graph.add_node(node_id, type=node_type)
        
        # 添加因果边（基于单Disease列的边）
        self._add_causal_edges()
    
    def _add_causal_edges(self):
        """添加因果边（删除Disease2相关边）"""
        # 1. 药物 -> 基因（药物影响基因表达）
        if hasattr(self.hetero_data, ('drug', 'targets', 'gene')):
            edge_index = self.hetero_data[('drug', 'targets', 'gene')].edge_index
            for i in range(edge_index.size(1)):
                drug_idx = edge_index[0, i].item()
                gene_idx = edge_index[1, i].item()
                self.causal_graph.add_edge(
                    f"drug_{drug_idx}", f"gene_{gene_idx}", 
                    relation="targets", weight=1.0
                )
        
        # 2. 基因 -> 疾病（基因突变导致疾病）
        if hasattr(self.hetero_data, ('gene', 'associated_with', 'disease')):
            edge_index = self.hetero_data[('gene', 'associated_with', 'disease')].edge_index
            for i in range(edge_index.size(1)):
                gene_idx = edge_index[0, i].item()
                disease_idx = edge_index[1, i].item()
                self.causal_graph.add_edge(
                    f"gene_{gene_idx}", f"disease_{disease_idx}", 
                    relation="causes", weight=0.8
                )
        
        # 3. 药物 -> 疾病（药物治疗疾病）
        if hasattr(self.hetero_data, ('drug', 'treats', 'disease')):
            edge_index = self.hetero_data[('drug', 'treats', 'disease')].edge_index
            for i in range(edge_index.size(1)):
                drug_idx = edge_index[0, i].item()
                disease_idx = edge_index[1, i].item()
                self.causal_graph.add_edge(
                    f"drug_{drug_idx}", f"disease_{disease_idx}", 
                    relation="treats", weight=0.9
                )
    
    def _calculate_average_treatment_effect(self, treatment: str, outcome: str) -> float:
        """计算平均因果效应（适配单Disease列的疾病节点）"""
        # 查找处理和结果节点（药物和疾病）
        treatment_node = None
        outcome_node = None
        for node in self.causal_graph.nodes():
            node_type = self.causal_graph.nodes[node]['type']
            # 处理节点：药物（名称匹配）
            if node_type == 'drug' and treatment.lower() in node.lower():
                treatment_node = node
            # 结果节点：疾病（名称匹配）
            elif node_type == 'disease' and outcome.title() in node.title():
                outcome_node = node
        
        if treatment_node is None or outcome_node is None:
            return 0.0
        
        # 计算所有从处理到结果的路径
        try:
            paths = list(nx.all_simple_paths(self.causal_graph, treatment_node, outcome_node, cutoff=3))
        except:
            paths = []
        
        # 基于路径权重计算总效应
        total_effect = 0.0
        for path in paths:
            path_effect = 1.0
            for i in range(len(path) - 1):
                edge_data = self.causal_graph[path[i]][path[i+1]]
                path_effect *= edge_data.get('weight', 0.5)
            total_effect += path_effect
        
        # 归一化
        ate = total_effect / len(paths) if len(paths) > 0 else 0.0
        return min(max(ate, 0.0), 1.0)  # 限制在[0,1]
    
    def identify_confounders(self, treatment: str, outcome: str) -> List[str]:
        """识别混杂变量（适配单Disease列）"""
        confounders = []
        self._build_causal_graph()
        
        # 查找处理和结果节点
        treatment_node = None
        outcome_node = None
        for node in self.causal_graph.nodes():
            node_type = self.causal_graph.nodes[node]['type']
            if node_type == 'drug' and treatment.lower() in node.lower():
                treatment_node = node
            elif node_type == 'disease' and outcome.title() in node.title():
                outcome_node = node
        
        if treatment_node is None or outcome_node is None:
            return confounders
        
        # 查找共同祖先（混杂变量）
        treatment_ancestors = set(nx.ancestors(self.causal_graph, treatment_node))
        outcome_ancestors = set(nx.ancestors(self.causal_graph, outcome_node))
        common_ancestors = treatment_ancestors.intersection(outcome_ancestors)
        
        # 过滤处理和结果节点本身
        for ancestor in common_ancestors:
            if ancestor != treatment_node and ancestor != outcome_node:
                confounders.append(ancestor)
        
        return confounders
    
    def do_calculus_adjustment(self, treatment: str, outcome: str, confounders: List[str]) -> float:
        """应用Do-Calculus进行调整（无核心修改）"""
        print(f"Applying do-calculus adjustment for {treatment} -> {outcome}")
        
        base_ate = self.apply_causal_inference(treatment, outcome)
        
        # 根据混杂变量类型调整
        adjustment_factor = 1.0
        for confounder in confounders:
            if 'gene' in confounder:
                adjustment_factor *= 0.8  # 基因混杂因子
            elif 'disease' in confounder:
                adjustment_factor *= 0.9  # 疾病混杂因子
        
        adjusted_ate = base_ate * adjustment_factor
        return adjusted_ate

class CausalFeatureEnhancer:
    """因果特征增强器（适配单Disease列的因果图）"""
    
    def __init__(self, graph_builder: CausalHeterogeneousGraphBuilder):
        self.graph_builder = graph_builder
    
    def enhance_drug_features_with_causal_info(self, drug: str) -> np.ndarray:
        """使用因果信息增强药物特征（适配单Disease列的因果图）"""
        base_features = self.graph_builder.multimodal_features['drugs'].get(drug, np.zeros(5))
        causal_features = self._extract_causal_features(drug)
        
        enhanced_features = np.concatenate([base_features, causal_features])
        return enhanced_features
    
    def _extract_causal_features(self, drug: str) -> np.ndarray:
        """提取因果特征（适配单Disease列的因果图）"""
        causal_features = []
        
        # 1. 药物的因果出度（影响的基因和疾病数量）
        out_degree = 0
        drug_node = None
        for node in self.graph_builder.causal_graph.nodes():
            if self.graph_builder.causal_graph.nodes[node]['type'] == 'drug' and drug.lower() in node.lower():
                drug_node = node
                break
        if drug_node:
            out_degree = self.graph_builder.causal_graph.out_degree(drug_node)
        causal_features.append(out_degree)
        
        # 2. 药物的因果入度（被其他实体影响的程度）
        in_degree = self.graph_builder.causal_graph.in_degree(drug_node) if drug_node else 0
        causal_features.append(in_degree)
        
        # 3. 平均因果路径长度（到所有疾病节点的平均距离）
        avg_path_length = self._calculate_avg_causal_path_length(drug_node)
        causal_features.append(avg_path_length)
        
        return np.array(causal_features, dtype=np.float32)
    
    def _calculate_avg_causal_path_length(self, drug_node: str) -> float:
        """计算平均因果路径长度（仅到疾病节点）"""
        if drug_node is None:
            return 0.0
        
        path_lengths = []
        # 仅计算到疾病节点的路径
        disease_nodes = [n for n in self.graph_builder.causal_graph.nodes() if self.graph_builder.causal_graph.nodes[n]['type'] == 'disease']
        
        for target_node in disease_nodes:
            try:
                path_length = nx.shortest_path_length(self.graph_builder.causal_graph, drug_node, target_node)
                path_lengths.append(path_length)
            except:
                continue
        
        return np.mean(path_lengths) if path_lengths else 0.0