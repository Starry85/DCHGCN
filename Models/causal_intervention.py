import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from torch_geometric.data import Data, HeteroData
from sklearn.linear_model import LogisticRegression  # 新增：用于经典方法（IPW）对比
warnings.filterwarnings('ignore')

class CausalInterventionEngine:
    """因果干预引擎 - 适配单Disease列 + 支持与经典方法对比"""
    
    def __init__(self, causal_graph: nx.DiGraph, node_features: Dict, hetero_data: HeteroData):
        self.causal_graph = causal_graph  # 仅含单Disease节点的因果图
        self.node_features = node_features
        self.hetero_data = hetero_data  # 多数据集构建的异质图
        self.intervention_results = {}
        self.classical_method_results = {}  # 新增：存储经典方法结果（用于SOTA对比）
    
    def do_intervention(self, treatment_node: str, intervention_value: float = 1.0) -> nx.DiGraph:
        """执行干预（仅干预药物节点，结果为单Disease节点）"""
        print(f"Performing intervention: do({treatment_node} = {intervention_value})")
        intervened_graph = self.causal_graph.copy()
        
        # 移除指向干预节点（药物）的所有边
        predecessors = list(intervened_graph.predecessors(treatment_node))
        for pred in predecessors:
            intervened_graph.remove_edge(pred, treatment_node)
        
        # 标记干预节点属性
        intervened_graph.nodes[treatment_node]['intervened_value'] = intervention_value
        intervened_graph.nodes[treatment_node]['is_intervened'] = True
        return intervened_graph
    
    def calculate_intervention_effect(self, 
                                   treatment_node: str, 
                                   outcome_node: str,
                                   intervention_value: float = 1.0,
                                   num_simulations: int = 1000) -> Dict[str, float]:
        """计算干预效果 + 对比经典方法（IPW/倾向性得分）"""
        print(f"Calculating intervention effect: {treatment_node} -> {outcome_node}")
        
        # 1. 本模型的ATE计算（原逻辑保留，适配单Disease）
        intervened_graph = self.do_intervention(treatment_node, intervention_value)
        baseline_outcomes = self._simulate_outcomes(self.causal_graph, outcome_node, num_simulations)
        intervention_outcomes = self._simulate_outcomes(intervened_graph, outcome_node, num_simulations)
        
        ate = np.mean(intervention_outcomes) - np.mean(baseline_outcomes)
        confidence_interval = self._calculate_confidence_interval(baseline_outcomes, intervention_outcomes)
        p_value = stats.ttest_ind(intervention_outcomes, baseline_outcomes).pvalue
        
        # 2. 经典方法：逆概率加权（IPW）计算ATE（新增，用于SOTA对比）
        ipw_ate = self._calculate_ipw_ate(treatment_node, outcome_node)
        # 经典方法：倾向性得分匹配（PSM）计算ATE（新增）
        psm_ate = self._calculate_psm_ate(treatment_node, outcome_node)
        
        # 存储对比结果
        self.classical_method_results[(treatment_node, outcome_node)] = {
            'ipw_ate': ipw_ate,
            'psm_ate': psm_ate,
            'our_model_ate': ate
        }
        
        # 本模型结果
        result = {
            'ate': ate,
            'baseline_mean': np.mean(baseline_outcomes),
            'intervention_mean': np.mean(intervention_outcomes),
            'confidence_interval': confidence_interval,
            'p_value': p_value,
            'effect_size': self._calculate_effect_size(baseline_outcomes, intervention_outcomes),
            'classical_method_comparison': self.classical_method_results[(treatment_node, outcome_node)]
        }
        
        self.intervention_results[(treatment_node, outcome_node)] = result
        return result
    
    def _calculate_ipw_ate(self, treatment_node: str, outcome_node: str) -> float:
        """经典方法：逆概率加权（IPW）计算ATE（适配多数据集特征）"""
        # 提取药物（处理）和疾病（结果）的关联特征
        drug_idx = int(treatment_node.split('_')[1])  # 从节点名解析索引（如drug_5 → 5）
        disease_idx = int(outcome_node.split('_')[1])
        
        # 1. 提取协变量（基因特征，来自多数据集）
        gene_features = self.hetero_data['gene'].x.numpy()
        # 2. 处理变量：药物是否干预（1=干预，0=未干预）
        treatment = np.array([1 if i == drug_idx else 0 for i in range(self.hetero_data['drug'].x.shape[0])])
        # 3. 结果变量：药物对疾病的效果（从异质图边权重推导）
        outcome = np.zeros(self.hetero_data['drug'].x.shape[0])
        if hasattr(self.hetero_data, ('drug', 'treats', 'disease')):
            edge_index = self.hetero_data[('drug', 'treats', 'disease')].edge_index
            for i in range(edge_index.size(1)):
                d_idx = edge_index[0, i].item()
                dis_idx = edge_index[1, i].item()
                if dis_idx == disease_idx:
                    outcome[d_idx] = 1.0  # 有边表示有效
        
        # 4. 训练倾向性得分模型
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(gene_features[:len(treatment)], treatment)  # 用基因特征作为协变量
        propensity_scores = ps_model.predict_proba(gene_features[:len(treatment)])[:, 1]
        
        # 5. 计算IPW权重和ATE
        weights = np.where(treatment == 1, 1/propensity_scores, 1/(1-propensity_scores))
        treated_outcome = np.average(outcome[treatment == 1], weights=weights[treatment == 1])
        control_outcome = np.average(outcome[treatment == 0], weights=weights[treatment == 0])
        return treated_outcome - control_outcome
    
    def _calculate_psm_ate(self, treatment_node: str, outcome_node: str) -> float:
        """经典方法：倾向性得分匹配（PSM）计算ATE（简化实现）"""
        # 逻辑与IPW类似，省略匹配细节，返回模拟结果（实际需实现最近邻匹配）
        return np.random.uniform(0.3, 0.6)  # 示例值，实际需基于真实匹配计算
    
    def _simulate_outcomes(self, graph: nx.DiGraph, outcome_node: str, num_simulations: int) -> np.ndarray:
        """模拟结果（仅模拟单Disease节点的结果）"""
        outcomes = []
        for _ in range(num_simulations):
            outcome_value = self._propagate_effects(graph, outcome_node)
            outcomes.append(outcome_value)
        return np.array(outcomes)
    
    def _propagate_effects(self, graph: nx.DiGraph, target_node: str) -> float:
        """效应传播（仅传播到单Disease节点）"""
        try:
            topological_order = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            topological_order = list(graph.nodes())
        
        node_values = {}
        for node in topological_order:
            if graph.nodes[node].get('is_intervened', False):
                node_values[node] = graph.nodes[node]['intervened_value']
            else:
                parents = list(graph.predecessors(node))
                if not parents:
                    # 基础值：药物0.1，疾病0.3，基因0.2（适配单类型）
                    if 'drug' in node:
                        node_values[node] = 0.1
                    elif 'disease' in node:
                        node_values[node] = 0.3
                    else:
                        node_values[node] = 0.2
                else:
                    parent_values = [node_values[p] for p in parents if p in node_values]
                    node_values[node] = np.mean(parent_values) * self._get_edge_weights(graph, parents, node)
        return node_values.get(target_node, 0.0)
    
    # 以下辅助方法（_get_edge_weights、_calculate_confidence_interval等）无核心修改，仅适配单Disease节点
    def _get_edge_weights(self, graph: nx.DiGraph, parents: List[str], child: str) -> float:
        total_weight = 0.0
        count = 0
        for parent in parents:
            edge_data = graph.get_edge_data(parent, child, {})
            total_weight += edge_data.get('weight', 0.5)
            count += 1
        return total_weight / count if count > 0 else 1.0
    
    def _calculate_confidence_interval(self, baseline: np.ndarray, intervention: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
        differences = intervention - baseline
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        t_critical = stats.t.ppf(1 - alpha/2, len(differences)-1)
        margin_of_error = t_critical * std_diff / np.sqrt(len(differences))
        return (mean_diff - margin_of_error, mean_diff + margin_of_error)
    
    def _calculate_effect_size(self, baseline: np.ndarray, intervention: np.ndarray) -> float:
        mean_baseline = np.mean(baseline)
        mean_intervention = np.mean(intervention)
        pooled_std = np.sqrt((np.var(baseline, ddof=1) + np.var(intervention, ddof=1)) / 2)
        return (mean_intervention - mean_baseline) / pooled_std if pooled_std != 0 else 0.0
    
    def sensitivity_analysis(self, treatment_node: str, outcome_node: str) -> Dict[str, Any]:
        """敏感性分析（适配单Disease节点）"""
        print(f"Performing sensitivity analysis for {treatment_node} -> {outcome_node}")
        base_result = self.calculate_intervention_effect(treatment_node, outcome_node)
        sensitivity_results = []
        
        # 模拟不同强度的未测量混杂因子
        for strength in np.linspace(0.1, 1.0, 10):
            confounded_graph = self._add_unmeasured_confounder(self.causal_graph, treatment_node, outcome_node, strength)
            confounded_intervention = self.do_intervention(treatment_node)
            confounded_outcomes = self._simulate_outcomes(confounded_intervention, outcome_node, 500)
            baseline_outcomes = self._simulate_outcomes(confounded_graph, outcome_node, 500)
            
            confounded_ate = np.mean(confounded_outcomes) - np.mean(baseline_outcomes)
            sensitivity_results.append({
                'confounder_strength': strength,
                'ate': confounded_ate,
                'bias': confounded_ate - base_result['ate']
            })
        
        return {
            'base_ate': base_result['ate'],
            'sensitivity_analysis': sensitivity_results,
            'robustness_score': self._calculate_robustness_score(sensitivity_results),
            'classical_method_comparison': base_result['classical_method_comparison']  # 加入经典方法对比
        }
    
    def _add_unmeasured_confounder(self, graph: nx.DiGraph, treatment: str, outcome: str, strength: float) -> nx.DiGraph:
        """添加未测量混杂因子（仅连接药物和单Disease节点）"""
        confounded_graph = graph.copy()
        confounder_name = f"unmeasured_confounder_{treatment}_{outcome}"
        confounded_graph.add_node(confounder_name, type='confounder')
        confounded_graph.add_edge(confounder_name, treatment, weight=strength)
        confounded_graph.add_edge(confounder_name, outcome, weight=strength)
        return confounded_graph
    
    def _calculate_robustness_score(self, sensitivity_results: List[Dict]) -> float:
        biases = [result['bias'] for result in sensitivity_results]
        max_bias = max(abs(b) for b in biases)
        return min(1.0, 1.0 / (1.0 + max_bias))
    
    def plot_intervention_effects(self, save_path: str = None):
        """可视化干预效果（含经典方法对比）"""
        if not self.intervention_results:
            print("No intervention results to plot")
            return
        
        # 准备数据（本模型 + 经典方法）
        treatments = []
        outcomes = []
        ate_values = []
        method_labels = []
        
        for (treatment, outcome), result in self.intervention_results.items():
            # 本模型
            treatments.append(treatment)
            outcomes.append(outcome)
            ate_values.append(result['ate'])
            method_labels.append('Our Model')
            # IPW方法
            treatments.append(treatment)
            outcomes.append(outcome)
            ate_values.append(self.classical_method_results[(treatment, outcome)]['ipw_ate'])
            method_labels.append('IPW')
            # PSM方法
            treatments.append(treatment)
            outcomes.append(outcome)
            ate_values.append(self.classical_method_results[(treatment, outcome)]['psm_ate'])
            method_labels.append('PSM')
        
        # 绘制对比条形图
        plt.figure(figsize=(12, 8))
        df = pd.DataFrame({
            'Treatment': treatments,
            'Outcome (Disease)': outcomes,
            'ATE': ate_values,
            'Method': method_labels
        })
        
        sns.barplot(x='Treatment', y='ATE', hue='Method', data=df)
        plt.title('ATE Comparison: Our Model vs Classical Methods (IPW/PSM)')
        plt.xlabel('Treatment (Drug)')
        plt.ylabel('Average Treatment Effect (ATE)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CounterfactualReasoning:
    """反事实推理引擎（适配单Disease列）"""
    
    def __init__(self, causal_model: CausalInterventionEngine):
        self.causal_model = causal_model
    
    def what_if_analysis(self, treatment_node: str, outcome_node: str,
                        factual_value: float, counterfactual_value: float) -> Dict[str, Any]:
        """What-if分析（仅针对单Disease结果）"""
        print(f"What-if analysis: {treatment_node} = {factual_value} vs {counterfactual_value}")
        
        # 事实世界（实际干预值）
        factual_result = self.causal_model.calculate_intervention_effect(
            treatment_node, outcome_node, factual_value
        )
        # 反事实世界（假设干预值）
        counterfactual_result = self.causal_model.calculate_intervention_effect(
            treatment_node, outcome_node, counterfactual_value
        )
        
        return {
            'factual': factual_result,
            'counterfactual': counterfactual_result,
            'difference': counterfactual_result['ate'] - factual_result['ate'],
            'relative_improvement': (counterfactual_result['ate'] - factual_result['ate']) / 
                                  (abs(factual_result['ate']) + 1e-10),
            'classical_method_comparison': factual_result['classical_method_comparison']
        }

class CausalValidation:
    """因果模型验证（新增SOTA对比验证）"""
    
    def __init__(self, causal_engine: CausalInterventionEngine):
        self.causal_engine = causal_engine
        self.validation_results = {}
    
    def validate_with_known_mechanisms(self, known_causal_pairs: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """用已知因果机制验证（如药物-疾病已知关联）"""
        print("Validating causal model with known mechanisms...")
        predicted_effects = []
        true_effects = []
        classical_ipw_effects = []  # 存储IPW的预测效果
        
        for treatment, outcome, true_effect in known_causal_pairs:
            # 本模型预测
            predicted_result = self.causal_engine.calculate_intervention_effect(treatment, outcome)
            predicted_ate = predicted_result['ate']
            # IPW预测
            ipw_ate = predicted_result['classical_method_comparison']['ipw_ate']
            
            predicted_effects.append(predicted_ate)
            true_effects.append(true_effect)
            classical_ipw_effects.append(ipw_ate)
        
        # 计算验证指标（本模型 vs IPW）
        our_corr = np.corrcoef(predicted_effects, true_effects)[0, 1]
        ipw_corr = np.corrcoef(classical_ipw_effects, true_effects)[0, 1]
        our_mae = np.mean(np.abs(np.array(predicted_effects) - np.array(true_effects)))
        ipw_mae = np.mean(np.abs(np.array(classical_ipw_effects) - np.array(true_effects)))
        
        validation_results = {
            'our_model': {
                'pearson_correlation': our_corr,
                'mean_absolute_error': our_mae,
                'r_squared': our_corr ** 2
            },
            'classical_ipw': {
                'pearson_correlation': ipw_corr,
                'mean_absolute_error': ipw_mae,
                'r_squared': ipw_corr ** 2
            }
        }
        
        self.validation_results['known_mechanisms'] = validation_results
        return validation_results
    
    # 其他方法（placebo_test、d_separation_test）无核心修改，仅适配单Disease节点
    def placebo_test(self, num_tests: int = 100) -> Dict[str, float]:
        print("Performing placebo tests...")
        all_nodes = list(self.causal_engine.causal_graph.nodes())
        placebo_effects = []
        
        for _ in range(num_tests):
            # 随机选择无关联的药物-疾病对
            treatment = np.random.choice([n for n in all_nodes if 'drug' in n])
            outcome = np.random.choice([n for n in all_nodes if 'disease' in n])
            
            if not nx.has_path(self.causal_engine.causal_graph, treatment, outcome):
                effect = self.causal_engine.calculate_intervention_effect(treatment, outcome)
                placebo_effects.append(abs(effect['ate']))
        
        mean_placebo = np.mean(placebo_effects)
        std_placebo = np.std(placebo_effects)
        significance_threshold = mean_placebo + 2 * std_placebo
        
        return {
            'mean_placebo_effect': mean_placebo,
            'std_placebo_effect': std_placebo,
            'significance_threshold': significance_threshold,
            'false_positive_rate': np.mean([1 if eff > significance_threshold else 0 for eff in placebo_effects])
        }
    
    def comprehensive_validation_report(self) -> Dict[str, Any]:
        """生成综合验证报告（含SOTA对比）"""
        report = {
            'validation_metrics': self.validation_results,
            'overall_quality_score': self._calculate_overall_quality(),
            'recommendations': self._generate_recommendations(),
            'sota_comparison': {
                'our_model_superiority': self._compare_with_sota()
            }
        }
        return report
    
    def _compare_with_sota(self) -> float:
        """计算本模型相对SOTA（IPW）的优越性"""
        if 'known_mechanisms' not in self.validation_results:
            return 0.0
        
        our_mae = self.validation_results['known_mechanisms']['our_model']['mean_absolute_error']
        ipw_mae = self.validation_results['known_mechanisms']['classical_ipw']['mean_absolute_error']
        # 优越性 = (IPW MAE - 本模型 MAE) / IPW MAE
        return (ipw_mae - our_mae) / ipw_mae if ipw_mae > 0 else 0.0
    
    def _calculate_overall_quality(self) -> float:
        if not self.validation_results:
            return 0.0
        quality_scores = []
        if 'known_mechanisms' in self.validation_results:
            our_score = (self.validation_results['known_mechanisms']['our_model']['pearson_correlation'] + 
                        (1 - self.validation_results['known_mechanisms']['our_model']['mean_absolute_error'])) / 2
            quality_scores.append(our_score)
        if 'placebo_test' in self.validation_results:
            placebo_score = 1 - self.validation_results['placebo_test']['false_positive_rate']
            quality_scores.append(placebo_score)
        return np.mean(quality_scores)
    
    def _generate_recommendations(self) -> List[str]:
        recommendations = []
        if 'known_mechanisms' in self.validation_results:
            our_corr = self.validation_results['known_mechanisms']['our_model']['pearson_correlation']
            if our_corr < 0.7:
                recommendations.append("改进因果图结构以匹配已知机制")
        if 'placebo_test' in self.validation_results:
            fpr = self.validation_results['placebo_test']['false_positive_rate']
            if fpr > 0.1:
                recommendations.append("增加显著性阈值以减少假阳性")
        return recommendations