import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class InterpretabilityAnalyzer:
    """可解释性分析器 - 适配单Disease列 + 多数据源证据支持"""
    
    def __init__(self, 
                 model: nn.Module,
                 causal_graph: nx.DiGraph,  # 仅含单Disease节点的因果图
                 feature_names: Dict[str, List[str]],
                 evidence_db: Optional[Dict] = None):  # 多数据源证据库
        
        self.model = model
        self.causal_graph = causal_graph
        self.feature_names = feature_names
        self.evidence_db = evidence_db  # 用于获取多数据源支持
        
        # 解释方法映射（适配单Disease）
        self.interpretation_methods = {
            'feature_importance': self._analyze_feature_importance,
            'path_analysis': self._analyze_causal_paths,
            'attention_weights': self._analyze_attention,
            'counterfactual': self._generate_counterfactuals
        }
    
    def explain_prediction(self, 
                         drug: str, 
                         disease: str,  # 单Disease
                         prediction_score: float,
                         method: str = 'comprehensive') -> Dict[str, Any]:
        """解释单个预测（适配单Disease + 多数据源证据）"""
        print(f"Explaining prediction: {drug} -> {disease} (score: {prediction_score:.3f})")
        
        explanation = {
            'drug': drug,
            'disease': disease,
            'prediction_score': prediction_score,
            'confidence_level': self._assess_confidence(prediction_score),
            'biological_plausibility': self._assess_biological_plausibility(drug, disease),
            'evidence_support': self._get_evidence_support(drug, disease)  # 新增：多数据源支持
        }
        
        # 选择解释方法
        if method == 'comprehensive':
            explanation.update(self._comprehensive_explanation(drug, disease))
        elif method in self.interpretation_methods:
            explanation.update(self.interpretation_methods[method](drug, disease))
        
        return explanation
    
    def _get_evidence_support(self, drug: str, disease: str) -> Dict[str, Any]:
        """获取多数据源对该预测的支持情况"""
        if not self.evidence_db:
            return {'status': 'no_evidence_db'}
        
        key = f"{drug}_{disease}"
        evidence_list = self.evidence_db.get(key, [])
        if not evidence_list:
            return {'total_evidence': 0, 'source_support': {}, 'support_ratio': 0.0}
        
        # 统计各数据源的支持比例
        source_support = defaultdict(lambda: {'supporting': 0, 'opposing': 0, 'total': 0})
        for evidence in evidence_list:
            source = evidence['source']
            effect_size = evidence.get('effect_size', 0)
            p_value = evidence.get('p_value', 1.0)
            
            if p_value < 0.05 and effect_size > 0:
                source_support[source]['supporting'] += 1
            elif p_value < 0.05 and effect_size < 0:
                source_support[source]['opposing'] += 1
            source_support[source]['total'] += 1
        
        # 计算总体支持比例
        total_supporting = sum(s['supporting'] for s in source_support.values())
        total_evidence = sum(s['total'] for s in source_support.values())
        support_ratio = total_supporting / total_evidence if total_evidence > 0 else 0.0
        
        return {
            'total_evidence': total_evidence,
            'source_support': dict(source_support),
            'support_ratio': support_ratio,
            'dominant_source': max(source_support.keys(), key=lambda k: source_support[k]['total'], default=None)
        }
    
    def _comprehensive_explanation(self, drug: str, disease: str) -> Dict[str, Any]:
        """综合解释（仅分析药物到单Disease的路径）"""
        comprehensive_explanation = {}
        
        # 1. 特征重要性（药物/基因/疾病特征）
        comprehensive_explanation['feature_importance'] = self._analyze_feature_importance(drug, disease)
        
        # 2. 因果路径分析（仅药物→单Disease的路径）
        causal_paths = self._analyze_causal_paths(drug, disease)
        comprehensive_explanation['causal_paths'] = causal_paths
        
        # 3. 作用机制推断（基于单Disease路径）
        comprehensive_explanation['proposed_mechanisms'] = self._infer_mechanisms(drug, disease, causal_paths)
        
        # 4. 不确定性分析（加入数据源不确定性）
        comprehensive_explanation['uncertainty_analysis'] = self._analyze_uncertainty(drug, disease)
        
        return comprehensive_explanation
    
    def _analyze_causal_paths(self, drug: str, disease: str) -> Dict[str, Any]:
        """分析因果路径（仅保留药物→单Disease的路径）"""
        print("Analyzing causal paths (single disease)...")
        causal_paths = {
            'direct_paths': [],    # 药物→单Disease
            'indirect_paths': [],  # 药物→基因→单Disease
            'mediators': [],       # 中介变量（基因）
            'confounders': []      # 混杂变量
        }
        
        try:
            # 查找所有从药物到单Disease的路径（最大长度4，避免过复杂）
            all_paths = list(nx.all_simple_paths(self.causal_graph, drug, disease, cutoff=4))
            
            for path in all_paths:
                path_info = {
                    'path': path,
                    'length': len(path) - 1,
                    'strength': self._calculate_path_strength(path),
                    'biological_interpretation': self._interpret_path(path),
                    'evidence_support_count': self._count_path_evidence_support(path)  # 新增：路径的证据支持数
                }
                
                if len(path) == 2:  # 直接路径：药物→疾病
                    causal_paths['direct_paths'].append(path_info)
                else:  # 间接路径：药物→基因→疾病
                    causal_paths['indirect_paths'].append(path_info)
            
            # 识别中介变量（路径中间节点，排除药物和疾病）
            causal_paths['mediators'] = self._identify_mediators(drug, disease, all_paths)
            # 识别混杂变量（药物和疾病的共同祖先）
            causal_paths['confounders'] = self._identify_confounders(drug, disease)
        
        except Exception as e:
            print(f"Error in causal path analysis: {e}")
        
        return causal_paths
    
    def _count_path_evidence_support(self, path: List[str]) -> int:
        """统计支持该路径的证据数量（多数据源）"""
        if not self.evidence_db or len(path) < 2:
            return 0
        
        # 路径的核心关联：药物→中介→疾病
        if len(path) == 3:
            drug, mediator, disease = path
            key = f"{drug}_{disease}"
            evidence_list = self.evidence_db.get(key, [])
            # 统计提及该中介的证据
            return sum(1 for e in evidence_list if mediator in e.get('title', ''))
        elif len(path) == 2:
            drug, disease = path
            key = f"{drug}_{disease}"
            return len(self.evidence_db.get(key, []))
        else:
            return 0
    
    def _infer_mechanisms(self, drug: str, disease: str, causal_paths: Dict) -> List[Dict[str, Any]]:
        """推断作用机制（仅基于药物→单Disease的路径）"""
        print("Inferring mechanisms of action (single disease)...")
        mechanisms = []
        
        # 1. 直接机制（药物直接作用于疾病）
        for path_info in causal_paths['direct_paths']:
            mechanisms.append({
                'type': 'direct',
                'description': f"{drug} directly targets {disease} (path strength: {path_info['strength']:.2f})",
                'confidence': path_info['strength'],
                'supporting_evidence_count': path_info['evidence_support_count'],
                'biological_plausibility': 0.85
            })
        
        # 2. 中介机制（药物通过基因作用于疾病）
        for mediator in causal_paths['mediators']:
            mediation_strength = self._calculate_mediation_strength(drug, disease, mediator)
            mechanisms.append({
                'type': 'mediated',
                'description': f"{drug} affects {disease} through {mediator} (mediation strength: {mediation_strength:.2f})",
                'mediator': mediator,
                'confidence': mediation_strength,
                'supporting_evidence_count': self._count_mediator_evidence_support(drug, disease, mediator),
                'biological_plausibility': 0.75
            })
        
        # 按置信度排序
        mechanisms.sort(key=lambda x: x['confidence'], reverse=True)
        return mechanisms
    
    def _count_mediator_evidence_support(self, drug: str, disease: str, mediator: str) -> int:
        """统计支持该中介机制的证据数量"""
        if not self.evidence_db:
            return 0
        
        key = f"{drug}_{disease}"
        evidence_list = self.evidence_db.get(key, [])
        return sum(1 for e in evidence_list if mediator in e.get('title', '') or mediator in e.get('description', ''))
    
    # 以下方法（_analyze_feature_importance、_generate_counterfactuals等）适配单Disease，无核心逻辑修改
    def _analyze_feature_importance(self, drug: str, disease: str) -> Dict[str, float]:
        """分析特征重要性（适配单Disease的特征）"""
        print("Analyzing feature importance...")
        feature_importance = {}
        
        # 药物特征
        if 'drug' in self.feature_names:
            for i, feature in enumerate(self.feature_names['drug']):
                importance = np.random.uniform(0.2, 1.0)  # 模拟，实际用SHAP/LIME
                feature_importance[f"drug_{feature}"] = importance
        
        # 疾病特征（仅单Disease）
        if 'disease' in self.feature_names:
            for i, feature in enumerate(self.feature_names['disease']):
                importance = np.random.uniform(0.2, 1.0)
                feature_importance[f"disease_{feature}"] = importance
        
        # 基因特征
        if 'gene' in self.feature_names:
            for i, feature in enumerate(self.feature_names['gene']):
                importance = np.random.uniform(0.2, 1.0)
                feature_importance[f"gene_{feature}"] = importance
        
        # 归一化
        total = sum(feature_importance.values())
        if total > 0:
            feature_importance = {k: v/total for k, v in feature_importance.items()}
        
        # 按重要性排序
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def _generate_counterfactuals(self, drug: str, disease: str) -> Dict[str, Any]:
        """生成反事实解释（适配单Disease）"""
        print("Generating counterfactual explanations (single disease)...")
        counterfactuals = {
            'alternative_drugs': [],
            'mechanism_variants': [],
            'what_if_scenarios': []
        }
        
        # 1. 寻找类似药物（基于药物嵌入相似度）
        similar_drugs = self._find_similar_drugs(drug)
        for sim_drug in similar_drugs[:3]:
            counterfactuals['alternative_drugs'].append({
                'type': 'alternative_drug',
                'description': f"If using {sim_drug} instead of {drug} for {disease}",
                'expected_effect': 'similar' if np.random.random() > 0.3 else 'reduced',
                'confidence': 0.7
            })
        
        # 2. 机制变体（针对单Disease）
        mechanisms = ['inhibition', 'activation', 'modulation']
        for mech in mechanisms:
            counterfactuals['mechanism_variants'].append({
                'type': 'mechanism_variant',
                'description': f"If {drug} works through {mech} of {disease}-related genes",
                'expected_effect': 'enhanced' if np.random.random() > 0.5 else 'variable',
                'confidence': 0.5
            })
        
        return counterfactuals

class VisualizationEngine:
    """可视化引擎 - 适配单Disease路径 + 多数据源标注"""
    
    def __init__(self):
        self.color_palette = {
            'drug': '#FF6B6B',
            'disease': '#4ECDC4',  # 单Disease颜色
            'gene': '#45B7D1',
            'high_confidence': '#2ecc71',
            'medium_confidence': '#f39c12',
            'low_confidence': '#e74c3c',
            'pubmed': '#9B59B6',    # 数据源颜色
            'ctd': '#3498DB',
            'drugbank': '#E67E22'
        }
    
    def plot_causal_pathways(self, causal_paths: Dict, drug: str, disease: str, save_path: str = None):
        """绘制因果路径图（仅药物→单Disease）"""
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()
        
        # 添加所有路径的节点和边
        all_nodes = set()
        for path_type in ['direct_paths', 'indirect_paths']:
            for path_info in causal_paths[path_type]:
                path = path_info['path']
                # 添加节点
                for node in path:
                    node_type = 'drug' if node == drug else 'disease' if node == disease else 'gene'
                    G.add_node(node, type=node_type)
                # 添加边（带权重）
                for i in range(len(path)-1):
                    G.add_edge(
                        path[i], path[i+1],
                        weight=path_info['strength'],
                        evidence_support=path_info['evidence_support_count']
                    )
                all_nodes.update(path)
        
        # 布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制边（按权重和证据支持调整宽度）
        edges = G.edges()
        edge_weights = [G[u][v]['weight'] for u, v in edges]
        edge_support = [G[u][v]['evidence_support'] for u, v in edges]
        nx.draw_networkx_edges(
            G, pos, alpha=0.7,
            width=[w*5 + s*0.5 for w, s in zip(edge_weights, edge_support)],  # 证据支持增加宽度
            edge_color='#95a5a6'
        )
        
        # 绘制节点（按类型着色）
        node_colors = []
        for node in G.nodes():
            if node == drug:
                node_colors.append(self.color_palette['drug'])
            elif node == disease:
                node_colors.append(self.color_palette['disease'])
            else:
                node_colors.append(self.color_palette['gene'])
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.9)
        
        # 添加节点标签和边标签（证据支持数）
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        edge_labels = {(u, v): f"sup:{G[u][v]['evidence_support']}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f'Causal Pathways: {drug} → {disease}', size=15)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_evidence_support(self, evidence_support: Dict, save_path: str = None):
        """绘制多数据源证据支持情况"""
        if 'source_support' not in evidence_support:
            print("No evidence support data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        source_data = evidence_support['source_support']
        
        # 准备数据
        sources = list(source_data.keys())
        supporting = [source_data[s]['supporting'] for s in sources]
        opposing = [source_data[s]['opposing'] for s in sources]
        x = np.arange(len(sources))
        width = 0.35
        
        # 绘制堆叠条形图
        bars1 = plt.bar(x - width/2, supporting, width, label='Supporting', color=self.color_palette['high_confidence'])
        bars2 = plt.bar(x + width/2, opposing, width, label='Opposing', color=self.color_palette['low_confidence'])
        
        # 添加标签
        plt.xlabel('Evidence Source')
        plt.ylabel('Number of Evidence Items')
        plt.title('Evidence Support by Data Source')
        plt.xticks(x, sources)
        plt.legend()
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()