import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore')

class SpuriousCorrelationDetector:
    """伪关联检测器 - 含详细注释说明方法用途（所有方法均会用到，对应不同偏倚类型）
    
    核心功能：
    1. detect_confounding_bias：检测混杂偏倚（如基因同时影响药物和疾病）
    2. detect_selection_bias：检测选择偏倚（样本特征分布与总体差异）
    3. detect_collider_bias：检测碰撞偏倚（药物和疾病共同影响某个中间变量）
    4. detect_measurement_error：检测测量误差（特征与金标准的差异）
    """
    
    def __init__(self, multimodal_data: Dict, causal_graph: nx.DiGraph, hetero_data: HeteroData):
        self.multimodal_data = multimodal_data  # 多模态数据（药物/基因/疾病特征）
        self.causal_graph = causal_graph        # 单Disease因果图（用于偏倚路径分析）
        self.hetero_data = hetero_data        # 多数据集异质图（用于提取协变量）
        self.detected_spurious = {}           # 存储检测到的伪关联
    
    def detect_confounding_bias(self, treatment: str, outcome: str, 
                              candidate_confounders: List[str]) -> Dict[str, Any]:
        """检测混杂偏倚（必用）- 识别同时影响药物和疾病的变量（如基因、其他疾病）
        
        参数：
        - treatment: 处理变量（药物节点名，如drug_0）
        - outcome: 结果变量（单Disease节点名，如disease_1）
        - candidate_confounders: 候选混杂因子（如基因节点、其他疾病节点）
        
        返回：
        - confounding_scores: 各混杂因子的影响得分
        - significant_confounders: 显著混杂因子（得分>0.3）
        """
        print(f"Detecting confounding bias for {treatment} -> {outcome}")
        confounding_scores = {}
        
        for confounder in candidate_confounders:
            # 计算混杂得分：基于因果图中"药物→混杂因子→疾病"和"混杂因子→药物、混杂因子→疾病"的路径强度
            score = self._calculate_confounding_score(treatment, outcome, confounder)
            confounding_scores[confounder] = score
        
        # 筛选显著混杂因子（阈值0.3，可调整）
        significant_confounders = [
            conf for conf, score in confounding_scores.items() if score > 0.3
        ]
        
        result = {
            'confounding_scores': confounding_scores,
            'significant_confounders': significant_confounders,
            'total_confounding_bias': np.mean(list(confounding_scores.values())),
            'bias_type': 'confounding_bias'
        }
        
        self.detected_spurious[(treatment, outcome, 'confounding')] = result
        return result
    
    def _calculate_confounding_score(self, treatment: str, outcome: str, confounder: str) -> float:
        """计算混杂得分：路径强度之和 / 路径数量"""
        try:
            # 1. 检查因果图中是否存在混杂路径：treatment ← confounder → outcome 或 treatment → confounder → outcome
            has_confounder_to_treatment = self.causal_graph.has_edge(confounder, treatment)
            has_confounder_to_outcome = self.causal_graph.has_edge(confounder, outcome)
            has_treatment_to_confounder = self.causal_graph.has_edge(treatment, confounder)
            
            if not (has_confounder_to_treatment and has_confounder_to_outcome) and not (has_treatment_to_confounder and has_confounder_to_outcome):
                return 0.0  # 无混杂路径
            
            # 2. 计算所有通过该混杂因子的后门路径强度
            backdoor_paths = list(nx.all_simple_paths(self.causal_graph, treatment, outcome))
            confounding_paths = [p for p in backdoor_paths if confounder in p and len(p) > 2]
            
            if not confounding_paths:
                return 0.0
            
            # 3. 路径强度 = 路径中所有边权重的乘积
            path_strengths = []
            for path in confounding_paths:
                strength = 1.0
                for i in range(len(path)-1):
                    edge_data = self.causal_graph.get_edge_data(path[i], path[i+1], {})
                    strength *= edge_data.get('weight', 0.5)
                path_strengths.append(strength)
            
            return np.mean(path_strengths)  # 平均路径强度作为混杂得分
        except Exception as e:
            print(f"Error calculating confounding score: {e}")
            return 0.0
    
    def detect_selection_bias(self, sample_indices: List[int], 
                            feature_type: str = 'drug') -> Dict[str, Any]:
        """检测选择偏倚（必用）- 样本特征分布与总体差异（如多数据集抽样偏差）
        
        参数：
        - sample_indices: 样本索引（如训练集药物索引）
        - feature_type: 特征类型（drug/gene/disease，对应多模态数据）
        
        返回：
        - bias_score: 选择偏倚得分（KS统计量均值）
        - significant_bias: 是否存在显著偏倚（得分>0.1）
        """
        print("Detecting selection bias...")
        # 1. 提取总体特征（多数据集全部特征）和样本特征
        if feature_type not in self.multimodal_data:
            return {'bias_score': 0.0, 'significant_bias': False}
        
        total_features = np.array(list(self.multimodal_data[feature_type].values()))
        sample_features = total_features[sample_indices]
        population_features = total_features[np.setdiff1d(range(len(total_features)), sample_indices)]
        
        if len(population_features) == 0:
            return {'bias_score': 0.0, 'significant_bias': False}
        
        # 2. 用KS检验比较各特征的分布差异
        distribution_differences = []
        for feat_idx in range(total_features.shape[1]):
            ks_stat, p_value = stats.ks_2samp(
                sample_features[:, feat_idx], population_features[:, feat_idx]
            )
            distribution_differences.append({
                'feature_idx': feat_idx,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        # 3. 偏倚得分 = 所有特征KS统计量的均值
        bias_score = np.mean([diff['ks_statistic'] for diff in distribution_differences])
        result = {
            'distribution_differences': distribution_differences,
            'bias_score': bias_score,
            'significant_bias': bias_score > 0.1,  # 阈值0.1，KS均值>0.1视为显著偏倚
            'bias_type': 'selection_bias'
        }
        
        self.detected_spurious[('selection_bias', feature_type)] = result
        return result
    
    def detect_collider_bias(self, treatment: str, outcome: str, 
                           collider_candidates: List[str]) -> Dict[str, Any]:
        """检测碰撞偏倚（必用）- 药物和疾病共同影响某个中间变量（如炎症因子）
        
        参数：
        - treatment: 药物节点
        - outcome: 单Disease节点
        - collider_candidates: 候选碰撞变量（如基因、生物标志物节点）
        
        返回：
        - collider_scores: 各碰撞变量的得分
        - significant_colliders: 显著碰撞变量（得分>0.2）
        """
        print(f"Detecting collider bias for {treatment} -> {outcome}")
        collider_scores = {}
        
        for collider in collider_candidates:
            # 碰撞得分：药物→碰撞变量 和 疾病→碰撞变量 的边权重均值
            score = self._calculate_collider_score(treatment, outcome, collider)
            collider_scores[collider] = score
        
        # 筛选显著碰撞变量（阈值0.2）
        significant_colliders = [
            coll for coll, score in collider_scores.items() if score > 0.2
        ]
        
        result = {
            'collider_scores': collider_scores,
            'significant_colliders': significant_colliders,
            'total_collider_bias': np.mean(list(collider_scores.values())),
            'bias_type': 'collider_bias'
        }
        
        self.detected_spurious[(treatment, outcome, 'collider')] = result
        return result
    
    def _calculate_collider_score(self, treatment: str, outcome: str, collider: str) -> float:
        """计算碰撞得分：药物和疾病到碰撞变量的边权重均值"""
        try:
            # 碰撞变量需满足：药物→碰撞变量 且 疾病→碰撞变量
            has_treatment_to_collider = self.causal_graph.has_edge(treatment, collider)
            has_outcome_to_collider = self.causal_graph.has_edge(outcome, collider)
            
            if not (has_treatment_to_collider and has_outcome_to_collider):
                return 0.0
            
            # 提取边权重并计算均值
            treatment_weight = self.causal_graph[treatment][collider].get('weight', 0.5)
            outcome_weight = self.causal_graph[outcome][collider].get('weight', 0.5)
            return (treatment_weight + outcome_weight) / 2
        except Exception as e:
            print(f"Error calculating collider score: {e}")
            return 0.0
    
    def detect_measurement_error(self, feature_type: str = 'drug', 
                               gold_standard: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """检测测量误差（必用）- 特征测量值与金标准的差异（如多数据集特征一致性）
        
        参数：
        - feature_type: 特征类型（drug/gene/disease）
        - gold_standard: 金标准特征（如DrugBank的药物结构特征）
        
        返回：
        - overall_reliability: 总体可靠性（>0.7视为无显著误差）
        - significant_measurement_error: 是否存在显著测量误差
        """
        print(f"Detecting measurement error for {feature_type} features")
        error_metrics = {}
        features = np.array(list(self.multimodal_data[feature_type].values()))
        
        if gold_standard is not None and gold_standard.shape == features.shape:
            # 有金标准：直接计算相关性和MAE
            for feat_idx in range(features.shape[1]):
                corr = np.corrcoef(features[:, feat_idx], gold_standard[:, feat_idx])[0, 1]
                mae = np.mean(np.abs(features[:, feat_idx] - gold_standard[:, feat_idx]))
                error_metrics[feat_idx] = {
                    'correlation_with_gold': corr,
                    'mean_absolute_error': mae,
                    'reliability': max(0, corr)  # 可靠性=相关性（0-1）
                }
        else:
            # 无金标准：用内部一致性（与其他特征的平均相关性）
            for feat_idx in range(features.shape[1]):
                correlations = []
                for other_idx in range(features.shape[1]):
                    if other_idx != feat_idx:
                        corr = np.corrcoef(features[:, feat_idx], features[:, other_idx])[0, 1]
                        correlations.append(abs(corr))
                reliability = np.mean(correlations) if correlations else 0.0
                error_metrics[feat_idx] = {
                    'internal_consistency': reliability,
                    'estimated_reliability': reliability
                }
        
        # 总体可靠性 = 所有特征可靠性的均值
        overall_reliability = np.mean([
            metrics.get('reliability', metrics.get('estimated_reliability', 0)) 
            for metrics in error_metrics.values()
        ])
        
        result = {
            'feature_errors': error_metrics,
            'overall_reliability': overall_reliability,
            'significant_measurement_error': overall_reliability < 0.7,  # 可靠性<0.7视为显著误差
            'bias_type': 'measurement_error'
        }
        
        self.detected_spurious[('measurement_error', feature_type)] = result
        return result

class BiasCorrection:
    """偏倚校正器 - 对应上述检测方法，每种偏倚对应一种校正方法（均会用到）"""
    
    def __init__(self, detector: SpuriousCorrelationDetector):
        self.detector = detector
    
    def propensity_score_matching(self, treatment: np.ndarray, covariates: np.ndarray,
                                outcome: np.ndarray) -> Dict[str, Any]:
        """校正混杂偏倚 - 倾向性得分匹配（对应detect_confounding_bias）"""
        print("Applying propensity score matching (confounding correction)...")
        if len(treatment) < 10:
            return {'success': False, 'reason': 'Insufficient samples'}
        
        try:
            # 1. 训练倾向性得分模型（用基因等协变量预测药物干预）
            ps_model = LogisticRegression(random_state=42)
            ps_model.fit(covariates, treatment)
            propensity_scores = ps_model.predict_proba(covariates)[:, 1]
            
            # 2. 最近邻匹配（1:1匹配）
            matched_pairs = self._perform_matching(treatment, propensity_scores)
            
            # 3. 计算匹配后的ATE（校正混杂后的效果）
            matched_treat_outcome = outcome[matched_pairs['treatment_indices']]
            matched_control_outcome = outcome[matched_pairs['control_indices']]
            ate = np.mean(matched_treat_outcome) - np.mean(matched_control_outcome)
            
            return {
                'success': True,
                'propensity_scores': propensity_scores,
                'matched_pairs': matched_pairs,
                'ate_after_matching': ate,
                'balance_improvement': self._assess_balance_improvement(covariates, treatment, matched_pairs)
            }
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    def _perform_matching(self, treatment: np.ndarray, propensity_scores: np.ndarray, caliper: float = 0.1) -> Dict[str, List[int]]:
        """执行1:1最近邻匹配"""
        treat_indices = np.where(treatment == 1)[0]
        control_indices = np.where(treatment == 0)[0]
        matched_treat = []
        matched_control = []
        
        for treat_idx in treat_indices:
            treat_ps = propensity_scores[treat_idx]
            control_ps = propensity_scores[control_indices]
            diffs = np.abs(control_ps - treat_ps)
            
            if len(diffs) > 0 and np.min(diffs) <= caliper:
                best_control_idx = control_indices[np.argmin(diffs)]
                matched_treat.append(treat_idx)
                matched_control.append(best_control_idx)
                control_indices = control_indices[control_indices != best_control_idx]
        
        return {
            'treatment_indices': matched_treat,
            'control_indices': matched_control,
            'matching_ratio': len(matched_treat) / len(treat_indices)
        }
    
    def _assess_balance_improvement(self, covariates: np.ndarray, treatment: np.ndarray, matched_pairs: Dict) -> float:
        """评估匹配后协变量平衡性改善"""
        # 匹配前标准化均值差异（SMD）
        pre_smd = self._calculate_standardized_mean_differences(covariates, treatment)
        
        # 匹配后数据
        matched_treat_cov = covariates[matched_pairs['treatment_indices']]
        matched_control_cov = covariates[matched_pairs['control_indices']]
        matched_treat = np.ones(len(matched_treat_cov))
        matched_control = np.zeros(len(matched_control_cov))
        matched_cov = np.vstack([matched_treat_cov, matched_control_cov])
        matched_treat_full = np.concatenate([matched_treat, matched_control])
        
        # 匹配后SMD
        post_smd = self._calculate_standardized_mean_differences(matched_cov, matched_treat_full)
        
        # 平衡性改善 = 匹配前SMD均值 - 匹配后SMD均值
        return max(0, np.mean(pre_smd) - np.mean(post_smd))
    
    def _calculate_standardized_mean_differences(self, covariates: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """计算标准化均值差异（SMD）- 评估协变量平衡"""
        treat_group = covariates[treatment == 1]
        control_group = covariates[treatment == 0]
        smd = []
        
        for feat_idx in range(covariates.shape[1]):
            treat_mean = np.mean(treat_group[:, feat_idx])
            control_mean = np.mean(control_group[:, feat_idx])
            pooled_std = np.sqrt((np.var(treat_group[:, feat_idx]) + np.var(control_group[:, feat_idx])) / 2)
            
            if pooled_std > 0:
                smd.append(abs(treat_mean - control_mean) / pooled_std)
            else:
                smd.append(0.0)
        
        return np.array(smd)
    
    def inverse_probability_weighting(self, treatment: np.ndarray, covariates: np.ndarray,
                                    outcome: np.ndarray) -> Dict[str, Any]:
        """校正选择偏倚 - 逆概率加权（对应detect_selection_bias）"""
        print("Applying inverse probability weighting (selection bias correction)...")
        try:
            # 1. 估计选择概率（用协变量预测是否被选入样本）
            ps_model = LogisticRegression(random_state=42)
            ps_model.fit(covariates, treatment)
            propensity_scores = ps_model.predict_proba(covariates)[:, 1]
            
            # 2. 计算IPW权重
            weights = np.where(treatment == 1, 1/propensity_scores, 1/(1-propensity_scores))
            
            # 3. 加权计算ATE（校正选择偏倚）
            weighted_treat_outcome = np.average(outcome[treatment == 1], weights=weights[treatment == 1])
            weighted_control_outcome = np.average(outcome[treatment == 0], weights=weights[treatment == 0])
            ate = weighted_treat_outcome - weighted_control_outcome
            
            return {
                'success': True,
                'propensity_scores': propensity_scores,
                'weights': weights,
                'ate_after_ipw': ate,
                'effective_sample_size': (np.sum(weights) ** 2) / np.sum(weights ** 2)
            }
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    def regression_adjustment(self, treatment: np.ndarray, covariates: np.ndarray,
                           outcome: np.ndarray) -> Dict[str, Any]:
        """校正测量误差 - 回归调整（对应detect_measurement_error）"""
        print("Applying regression adjustment (measurement error correction)...")
        try:
            # 1. 构建回归模型（包含处理变量、协变量和交互项）
            X = np.column_stack([treatment, covariates, treatment * covariates])  # 加入交互项捕捉测量误差
            model = LinearRegression()
            model.fit(X, outcome)
            
            # 2. ATE = 处理变量的系数（校正测量误差后的效应）
            ate = model.coef_[0]
            
            return {
                'success': True,
                'model_coefficients': model.coef_,
                'ate_after_adjustment': ate,
                'r_squared': model.score(X, outcome)
            }
        except Exception as e:
            return {'success': False, 'reason': str(e)}

class MultimodalIntegration:
    """多模态数据整合 - 消除模态间伪关联（适配多数据集）"""
    
    def __init__(self, multimodal_features: Dict):
        self.multimodal_features = multimodal_features  # 多数据集多模态特征
        self.integrated_representations = {}  # 整合后的特征（消除伪关联后）
    
    def modality_alignment(self, modalities: List[str], alignment_method: str = 'cca') -> Dict[str, np.ndarray]:
        """模态对齐 - 消除模态间尺度差异（如药物结构特征与基因表达特征对齐）"""
        print(f"Aligning modalities: {modalities} (method: {alignment_method})")
        aligned_features = {}
        
        if alignment_method == 'cca':
            # 典型相关分析（CCA）- 最大化模态间相关性
            aligned_features = self._cca_alignment(modalities)
        elif alignment_method == 'manifold':
            # 流形对齐（t-SNE）- 映射到同一低维空间
            aligned_features = self._manifold_alignment(modalities)
        
        self.integrated_representations.update(aligned_features)
        return aligned_features
    
    def _cca_alignment(self, modalities: List[str]) -> Dict[str, np.ndarray]:
        """CCA对齐（适配多模态数据）"""
        from sklearn.cross_decomposition import CCA
        aligned = {}
        
        # 提取模态特征（需保证样本数一致）
        modal_features = []
        modal_keys = []
        for mod in modalities:
            if mod in self.multimodal_features:
                feats = np.array(list(self.multimodal_features[mod].values()))
                modal_features.append(feats)
                modal_keys.append(mod)
        
        if len(modal_features) < 2:
            return aligned
        
        # 两两CCA对齐（以第一个模态为基准）
        base_mod = modal_keys[0]
        base_feats = modal_features[0]
        
        for i in range(1, len(modal_features)):
            current_mod = modal_keys[i]
            current_feats = modal_features[i]
            
            # CCA降维到相同维度
            n_components = min(10, base_feats.shape[1], current_feats.shape[1])
            cca = CCA(n_components=n_components)
            base_aligned, current_aligned = cca.fit_transform(base_feats, current_feats)
            
            # 映射回原始键
            base_keys = list(self.multimodal_features[base_mod].keys())
            current_keys = list(self.multimodal_features[current_mod].keys())
            
            for j, key in enumerate(base_keys):
                aligned[f"{base_mod}_{key}"] = base_aligned[j]
            for j, key in enumerate(current_keys):
                aligned[f"{current_mod}_{key}"] = current_aligned[j]
        
        return aligned
    
    def _manifold_alignment(self, modalities: List[str]) -> Dict[str, np.ndarray]:
        """流形对齐（t-SNE）- 多模态映射到2D空间"""
        all_feats = []
        all_keys = []
        
        for mod in modalities:
            if mod in self.multimodal_features:
                for key, feat in self.multimodal_features[mod].items():
                    all_feats.append(feat)
                    all_keys.append(f"{mod}_{key}")
        
        if len(all_feats) < 2:
            return {}
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_feats)-1))
        aligned_emb = tsne.fit_transform(np.array(all_feats))
        
        return {key: aligned_emb[i] for i, key in enumerate(all_keys)}
    
    def detect_modality_conflicts(self, modalities: List[str], consensus_threshold: float = 0.7) -> Dict[str, Any]:
        """检测模态间冲突（如PubMed证据与CTD证据的差异）"""
        print("Detecting modality conflicts...")
        conflict_scores = {}
        
        # 两两比较模态一致性（余弦相似度）
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i < j and mod1 in self.multimodal_features and mod2 in self.multimodal_features:
                    # 计算模态间一致性得分（共同样本的平均余弦相似度）
                    agreement_score = self._calculate_modality_agreement(mod1, mod2)
                    conflict_scores[f"{mod1}_{mod2}"] = {
                        'agreement_score': agreement_score,
                        'has_conflict': agreement_score < consensus_threshold  # 相似度<0.7视为冲突
                    }
        
        overall_agreement = np.mean([s['agreement_score'] for s in conflict_scores.values()])
        return {
            'pairwise_agreement': conflict_scores,
            'overall_agreement': overall_agreement,
            'significant_conflicts': any(s['has_conflict'] for s in conflict_scores.values())
        }
    
    def _calculate_modality_agreement(self, mod1: str, mod2: str) -> float:
        """计算模态间一致性（共同样本的余弦相似度均值）"""
        mod1_feats = self.multimodal_features[mod1]
        mod2_feats = self.multimodal_features[mod2]
        common_keys = set(mod1_feats.keys()) & set(mod2_feats.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            vec1 = mod1_feats[key]
            vec2 = mod2_feats[key]
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            similarities.append(sim)
        
        return np.mean(similarities)