import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import requests
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class DynamicEvidenceIntegrator:
    """动态证据整合器 - 支持多数据源（PubMed/CTD/DrugBank）+ 适配单Disease列"""
    
    def __init__(self, 
                 model: nn.Module,
                 evidence_sources: List[str],  # 多数据源：['pubmed', 'ctd', 'drugbank']
                 integration_strategy: str = 'weighted_ensemble',
                 hetero_data: Any = None):  # 接收多数据集异质图
        
        self.model = model
        self.evidence_sources = evidence_sources
        self.integration_strategy = integration_strategy
        self.hetero_data = hetero_data  # 用于提取药物/疾病编码
        
        # 证据存储（按药物-单疾病对分组）
        self.evidence_db = defaultdict(list)  # key: "drug_disease"
        self.evidence_weights = self._initialize_evidence_weights()  # 按数据源设置权重
        
        # 模型版本与性能跟踪（加入SOTA对比）
        self.model_versions = {}
        self.performance_history = []
        self.sota_comparison = {}  # 存储与经典方法的性能差异
        
        # 证据质量评估器
        self.quality_assessor = EvidenceQualityAssessor()
    
    def _initialize_evidence_weights(self) -> Dict[str, float]:
        """初始化证据源权重（按数据源可靠性：PubMed>CTD>DrugBank>其他）"""
        source_weights = {
            'pubmed': 0.95,    # 随机对照试验/临床研究
            'ctd': 0.85,       # 药物-疾病关联数据库
            'drugbank': 0.9,   # 已批准药物适应症
            'bioarxiv': 0.6,   # 预印本
            'expert_opinion': 0.3
        }
        # 为未定义的数据源设置默认权重
        return {source: source_weights.get(source, 0.5) for source in self.evidence_sources}
    
    def collect_evidence_stream(self, 
                              drug_disease_pairs: List[Tuple[str, str]],  # (drug, disease) 单Disease
                              collection_interval: int = 3600) -> List[Dict]:
        """收集多数据源证据流（适配单Disease列）"""
        print("Starting evidence stream collection (multi-source)...")
        all_new_evidence = []
        
        for drug, disease in drug_disease_pairs:
            print(f"Collecting evidence for {drug} - {disease}")
            
            # 从每个数据源收集证据
            for source in self.evidence_sources:
                source_evidence = self._collect_from_source(source, drug, disease)
                # 标记数据源并添加权重
                for evidence in source_evidence:
                    evidence['source_weight'] = self.evidence_weights[source]
                    evidence['drug_disease_key'] = f"{drug}_{disease}"  # 单Disease键
                all_new_evidence.extend(source_evidence)
            
            time.sleep(1)  # 避免API限流
        
        # 质量过滤（多数据源共同过滤）
        filtered_evidence = self.quality_assessor.filter_evidence(all_new_evidence)
        
        # 存储证据（按药物-单疾病对分组）
        for evidence in filtered_evidence:
            self._store_evidence(evidence)
        
        print(f"Collected {len(filtered_evidence)} new evidence items (sources: {set(e['source'] for e in filtered_evidence)})")
        return filtered_evidence
    
    def _collect_from_source(self, source: str, drug: str, disease: str) -> List[Dict]:
        """从特定数据源收集证据（适配单Disease）"""
        evidence_items = []
        try:
            if source == 'pubmed':
                evidence_items = self._collect_pubmed_evidence(drug, disease)
            elif source == 'ctd':
                evidence_items = self._collect_ctd_evidence(drug, disease)
            elif source == 'drugbank':
                evidence_items = self._collect_drugbank_evidence(drug, disease)
            elif source == 'bioarxiv':
                evidence_items = self._collect_bioarxiv_evidence(drug, disease)
        
        except Exception as e:
            print(f"Error collecting from {source}: {e}")
        return evidence_items
    
    def _collect_ctd_evidence(self, drug: str, disease: str) -> List[Dict]:
        """新增：从CTD（Comparative Toxicogenomics Database）收集药物-疾病关联证据"""
        # 实际应用需调用CTD API，此处为模拟数据
        mock_evidence = [
            {
                'source': 'ctd',
                'drug': drug,
                'disease': disease,
                'title': f'CTD: {drug} associated with {disease}',
                'interaction_type': 'therapeutic',  # 治疗关联
                'evidence_type': 'observational_study',
                'confidence_score': 0.85,
                'publication_count': np.random.randint(5, 20),  # 支持该关联的文献数
                'last_updated': '2024-04-15'
            }
        ]
        return mock_evidence
    
    def _collect_drugbank_evidence(self, drug: str, disease: str) -> List[Dict]:
        """新增：从DrugBank收集已批准适应症证据"""
        # 实际应用需调用DrugBank API，此处为模拟数据
        is_approved = np.random.choice([True, False], p=[0.3, 0.7])  # 30%概率为已批准
        mock_evidence = [
            {
                'source': 'drugbank',
                'drug': drug,
                'disease': disease,
                'title': f'DrugBank: {drug} indication for {disease}',
                'evidence_type': 'rct' if is_approved else 'experimental',
                'approved': is_approved,
                'confidence_score': 0.95 if is_approved else 0.6,
                'approval_date': '2023-11-05' if is_approved else None
            }
        ]
        return mock_evidence
    
    # 以下为原有数据源收集方法（适配单Disease列，无核心修改）
    def _collect_pubmed_evidence(self, drug: str, disease: str) -> List[Dict]:
        mock_evidence = [
            {
                'source': 'pubmed',
                'drug': drug,
                'disease': disease,
                'title': f'Effect of {drug} on {disease} (Randomized Controlled Trial)',
                'publication_date': '2024-03-20',
                'journal': 'New England Journal of Medicine',
                'evidence_type': 'rct',
                'sample_size': np.random.randint(80, 500),
                'p_value': np.random.uniform(0.01, 0.05),
                'effect_size': np.random.uniform(0.3, 0.8),
                'confidence_score': np.random.uniform(0.8, 1.0)
            }
        ]
        return mock_evidence
    
    def _collect_bioarxiv_evidence(self, drug: str, disease: str) -> List[Dict]:
        mock_evidence = [
            {
                'source': 'bioarxiv',
                'drug': drug,
                'disease': disease,
                'title': f'Mechanistic study of {drug} in {disease} models',
                'preprint_date': '2024-04-01',
                'evidence_type': 'preclinical',
                'confidence_score': np.random.uniform(0.5, 0.7)
            }
        ]
        return mock_evidence
    
    def _store_evidence(self, evidence: Dict):
        """按药物-单疾病对存储证据（限制数量避免冗余）"""
        key = evidence['drug_disease_key']
        self.evidence_db[key].append(evidence)
        # 限制每个药物-疾病对最多存储100条证据
        if len(self.evidence_db[key]) > 100:
            self.evidence_db[key] = self.evidence_db[key][-100:]
    
    def calculate_evidence_coherence(self, drug: str, disease: str) -> Dict[str, float]:
        """计算证据一致性（考虑多数据源权重）"""
        key = f"{drug}_{disease}"
        evidence_list = self.evidence_db.get(key, [])
        if not evidence_list:
            return {'coherence_score': 0.0, 'evidence_direction': 'insufficient'}
        
        # 加权计算支持/反对证据（按数据源权重）
        supporting = 0.0
        opposing = 0.0
        neutral = 0.0
        
        for evidence in evidence_list:
            effect_size = evidence.get('effect_size', 0)
            p_value = evidence.get('p_value', 1.0)
            confidence = evidence['confidence_score'] * evidence['source_weight']  # 数据源加权
            
            if p_value < 0.05 and effect_size > 0:
                supporting += confidence
            elif p_value < 0.05 and effect_size < 0:
                opposing += confidence
            else:
                neutral += confidence
        
        total = supporting + opposing + neutral
        if total == 0:
            return {'coherence_score': 0.0, 'evidence_direction': 'insufficient'}
        
        # 一致性得分 = 支持比例 - 反对比例
        supporting_ratio = supporting / total
        opposing_ratio = opposing / total
        coherence_score = supporting_ratio - opposing_ratio
        
        # 确定证据方向（适配单Disease）
        if coherence_score > 0.3:
            direction = 'supporting'
        elif coherence_score < -0.3:
            direction = 'opposing'
        else:
            direction = 'conflicting'
        
        return {
            'coherence_score': coherence_score,
            'evidence_direction': direction,
            'supporting_evidence': supporting_ratio,
            'opposing_evidence': opposing_ratio,
            'neutral_evidence': neutral / total,
            'source_distribution': self._get_source_distribution(evidence_list)  # 新增：数据源分布
        }
    
    def _get_source_distribution(self, evidence_list: List[Dict]) -> Dict[str, int]:
        """统计证据的数据源分布"""
        source_counts = defaultdict(int)
        for evidence in evidence_list:
            source_counts[evidence['source']] += 1
        return dict(source_counts)
    
    def incremental_model_update(self, new_evidence: List[Dict], 
                               learning_rate: float = 0.001,
                               sota_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """增量模型更新 + 跟踪与SOTA方法的性能差异"""
        print("Performing incremental model update (with SOTA comparison)...")
        if not new_evidence:
            print("No new evidence for update")
            return
        
        # 准备训练数据（适配单Disease的药物-疾病对）
        training_data = self._prepare_incremental_training_data(new_evidence)
        if not training_data:
            print("No valid training data")
            return
        
        # 执行增量学习
        loss = self._perform_incremental_learning(training_data, learning_rate)
        
        # 记录性能（加入SOTA对比）
        performance_record = self._record_performance(loss, len(new_evidence), sota_metrics)
        
        # 创建模型版本（包含数据源和SOTA信息）
        version_info = self._create_model_version(loss, new_evidence, sota_metrics)
        
        print(f"Incremental update completed. Loss: {loss:.4f} | SOTA gap: {performance_record.get('sota_gap', 0.0):.4f}")
        return version_info
    
    def _prepare_incremental_training_data(self, new_evidence: List[Dict]) -> Optional[Dict]:
        """准备增量训练数据（从异质图提取真实嵌入，适配单Disease）"""
        drug_embeddings = []
        disease_embeddings = []
        labels = []
        weights = []
        
        # 从异质图获取药物/疾病编码
        drug_mapping = {name: idx for idx, name in enumerate(self.hetero_data['drug'].x)}
        disease_mapping = {name: idx for idx, name in enumerate(self.hetero_data['disease'].x)}
        
        for evidence in new_evidence:
            drug = evidence['drug']
            disease = evidence['disease']
            confidence = evidence['confidence_score'] * evidence['source_weight']  # 加权置信度
            
            # 检查药物/疾病是否在异质图中
            if drug not in drug_mapping or disease not in disease_mapping:
                continue
            
            # 提取真实嵌入（替换原随机生成）
            drug_idx = drug_mapping[drug]
            disease_idx = disease_mapping[disease]
            drug_emb = self.hetero_data['drug'].x[drug_idx]
            disease_emb = self.hetero_data['disease'].x[disease_idx]
            
            # 基于证据方向确定标签
            label = self._get_evidence_direction(evidence)
            
            drug_embeddings.append(drug_emb)
            disease_embeddings.append(disease_emb)
            labels.append(label)
            weights.append(confidence)
        
        if not drug_embeddings:
            return None
        
        return {
            'drug_embeddings': torch.stack(drug_embeddings),
            'disease_embeddings': torch.stack(disease_embeddings),
            'labels': torch.tensor(labels, dtype=torch.float),
            'weights': torch.tensor(weights, dtype=torch.float)
        }
    
    def _get_evidence_direction(self, evidence: Dict) -> float:
        """基于多数据源证据确定标签方向（适配单Disease）"""
        effect_size = evidence.get('effect_size', 0)
        p_value = evidence.get('p_value', 1.0)
        approved = evidence.get('approved', False)
        
        if approved or (p_value < 0.05 and effect_size > 0):
            return 1.0  # 正向关联（药物有效）
        elif p_value < 0.05 and effect_size < 0:
            return 0.0  # 负向关联（药物无效）
        else:
            return 0.5  # 不确定
    
    def _record_performance(self, loss: float, evidence_count: int, sota_metrics: Optional[Dict]) -> Dict[str, Any]:
        """记录性能 + 计算与SOTA的差距（如AUC差距）"""
        performance_record = {
            'timestamp': datetime.now(),
            'loss': loss,
            'evidence_count': evidence_count,
            'evidence_sources': self._get_evidence_sources_from_count(evidence_count),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        # 加入SOTA对比（如本模型AUC - SOTA模型AUC）
        if sota_metrics and 'our_auc' in sota_metrics and 'sota_auc' in sota_metrics:
            performance_record['sota_gap'] = sota_metrics['our_auc'] - sota_metrics['sota_auc']
            self.sota_comparison[datetime.now()] = performance_record['sota_gap']
        
        self.performance_history.append(performance_record)
        # 限制历史记录长度
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        return performance_record
    
    def _create_model_version(self, loss: float, new_evidence: List[Dict], sota_metrics: Optional[Dict]) -> Dict[str, Any]:
        """创建模型版本（包含数据源和SOTA信息）"""
        version_id = f"v{len(self.model_versions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        source_dist = self._get_source_distribution(new_evidence)
        
        version_info = {
            'version_id': version_id,
            'timestamp': datetime.now(),
            'loss': loss,
            'evidence_count': len(new_evidence),
            'evidence_sources': source_dist,
            'performance_metrics': self._calculate_performance_metrics(),
            'sota_comparison': sota_metrics  # 存储SOTA对比数据
        }
        
        self.model_versions[version_id] = version_info
        return version_info

class EvidenceQualityAssessor:
    """证据质量评估器（适配多数据源）"""
    
    def __init__(self):
        self.quality_metrics = {
            'sample_size_threshold': 50,
            'p_value_threshold': 0.05,
            'min_confidence': 0.3,
            'recent_threshold_days': 365  # 1年内证据视为近期
        }
    
    def filter_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        """基于多维度过滤低质量证据"""
        filtered = []
        for evidence in evidence_list:
            if self._assess_evidence_quality(evidence) >= 0.5:  # 质量阈值0.5
                filtered.append(evidence)
        return filtered
    
    def _assess_evidence_quality(self, evidence: Dict) -> float:
        """多维度评估证据质量（加入数据源权重）"""
        quality_score = 0.0
        
        # 1. 研究类型权重（RCT>临床研究>预印本）
        type_weights = {
            'rct': 1.0, 'clinical_trial': 0.9, 'observational_study': 0.7,
            'preclinical': 0.6, 'preprint': 0.5, 'expert_opinion': 0.3
        }
        evidence_type = evidence.get('evidence_type', 'unknown')
        quality_score += type_weights.get(evidence_type, 0.5) * 0.3
        
        # 2. 样本量（仅适用于临床研究）
        if evidence_type in ['rct', 'clinical_trial']:
            sample_size = evidence.get('sample_size', 0)
            if sample_size > 1000:
                quality_score += 0.3
            elif sample_size > 100:
                quality_score += 0.2
            elif sample_size > 50:
                quality_score += 0.1
        
        # 3. 统计显著性
        p_value = evidence.get('p_value', 1.0)
        if p_value < 0.001:
            quality_score += 0.2
        elif p_value < 0.01:
            quality_score += 0.15
        elif p_value < 0.05:
            quality_score += 0.1
        
        # 4. 数据源权重（已在整合时处理，此处仅补充）
        source_weight = evidence.get('source_weight', 0.5)
        quality_score += source_weight * 0.2
        
        # 5. 时效性
        pub_date = evidence.get('publication_date') or evidence.get('preprint_date')
        if pub_date:
            try:
                pub_dt = datetime.strptime(pub_date, '%Y-%m-%d')
                days_ago = (datetime.now() - pub_dt).days
                if days_ago <= 365:
                    quality_score += (1 - days_ago / 365) * 0.1
            except:
                pass
        
        return min(quality_score, 1.0)  # 限制在0-1之间

class ModelEvolutionTracker:
    """模型演化跟踪器（加入SOTA对比分析）"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.evolution_history = []
        self.performance_trends = defaultdict(list)
        self.sota_trends = []  # 跟踪与SOTA的差距趋势
    
    def track_evolution(self, 
                      update_type: str,
                      performance_metrics: Dict[str, float],
                      evidence_impact: float,
                      sota_gap: Optional[float] = None):
        """跟踪模型演化 + SOTA差距"""
        evolution_record = {
            'timestamp': datetime.now(),
            'update_type': update_type,
            'performance_metrics': performance_metrics,
            'evidence_impact': evidence_impact,
            'model_complexity': sum(p.numel() for p in self.model.parameters())
        }
        
        # 加入SOTA差距
        if sota_gap is not None:
            evolution_record['sota_gap'] = sota_gap
            self.sota_trends.append(sota_gap)
        
        self.evolution_history.append(evolution_record)
        
        # 更新性能趋势
        for metric, value in performance_metrics.items():
            self.performance_trends[metric].append(value)
    
    def analyze_evolution_trends(self) -> Dict[str, Any]:
        """分析演化趋势（包含SOTA差距趋势）"""
        if len(self.evolution_history) < 2:
            return {'status': 'insufficient_data'}
        
        trend_analysis = {}
        
        # 1. 性能指标趋势（如loss、AUC）
        for metric, values in self.performance_trends.items():
            if len(values) >= 2:
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                trend_analysis[metric] = {
                    'current_value': values[-1],
                    'trend_slope': slope,
                    'trend_direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                    'volatility': np.std(values[-5:]) if len(values) >= 5 else np.std(values)
                }
        
        # 2. SOTA差距趋势
        if self.sota_trends:
            x_sota = np.arange(len(self.sota_trends))
            slope_sota, _, _, _, _ = stats.linregress(x_sota, self.sota_trends)
            trend_analysis['sota_gap'] = {
                'current_gap': self.sota_trends[-1],
                'trend_slope': slope_sota,
                'trend_direction': 'closing' if slope_sota > 0 else 'widening' if slope_sota < 0 else 'stable'
            }
        
        # 3. 整体演化状态
        improving_metrics = sum(1 for analysis in trend_analysis.values() 
                              if analysis.get('trend_direction') == 'improving')
        total_metrics = len(trend_analysis)
        evolution_status = 'healthy' if improving_metrics / total_metrics > 0.6 else 'concerning'
        
        return {
            'trend_analysis': trend_analysis,
            'evolution_status': evolution_status,
            'improvement_ratio': improving_metrics / total_metrics,
            'total_updates': len(self.evolution_history)
        }
    
    def generate_evolution_report(self) -> Dict[str, Any]:
        """生成演化报告（包含SOTA对比建议）"""
        trend_analysis = self.analyze_evolution_trends()
        concept_drift = self.detect_concept_drift()
        
        report = {
            'health_score': self._calculate_evolution_health(),
            'trend_analysis': trend_analysis,
            'concept_drift': concept_drift,
            'total_evolution_steps': len(self.evolution_history),
            'recommendations': self._generate_recommendations(trend_analysis, concept_drift),
            'sota_comparison_summary': self._summarize_sota_comparison()
        }
        
        return report
    
    def _summarize_sota_comparison(self) -> Dict[str, Any]:
        """总结与SOTA的对比"""
        if not self.sota_trends:
            return {'status': 'no_sota_data'}
        
        return {
            'mean_sota_gap': np.mean(self.sota_trends),
            'current_sota_gap': self.sota_trends[-1],
            'gap_trend': 'closing' if np.mean(self.sota_trends[-5:]) > np.mean(self.sota_trends[:5]) else 'widening',
            'days_to_sota': self._estimate_days_to_sota()
        }
    
    def _estimate_days_to_sota(self) -> Optional[float]:
        """估计追上SOTA所需时间（简化）"""
        if len(self.sota_trends) < 5 or self.sota_trends[-1] < 0:
            return None  # 尚未落后或数据不足
        # 假设差距以当前斜率缩小
        slope = np.polyfit(range(len(self.sota_trends)), self.sota_trends, 1)[0]
        if slope <= 0:
            return float('inf')  # 差距未缩小
        days_per_update = 1  # 假设每天更新一次
        days_needed = self.sota_trends[-1] / slope * days_per_update
        return min(days_needed, 365)  # 上限1年

    # 其他方法（detect_concept_drift、_calculate_evolution_health等）无核心修改，仅适配SOTA字段
    def detect_concept_drift(self, window_size: int = 10) -> Dict[str, Any]:
        if len(self.performance_trends.get('mean_loss', [])) < window_size * 2:
            return {'detected': False, 'confidence': 0.0}
        
        losses = self.performance_trends['mean_loss']
        recent_losses = losses[-window_size:]
        previous_losses = losses[-window_size*2:-window_size]
        
        try:
            t_stat, p_value = stats.ttest_ind(previous_losses, recent_losses)
            drift_detected = p_value < 0.05 and np.mean(recent_losses) > np.mean(previous_losses)
            return {
                'detected': drift_detected,
                'confidence': 1 - p_value if drift_detected else p_value,
                'p_value': p_value,
                'performance_change': np.mean(recent_losses) - np.mean(previous_losses)
            }
        except:
            return {'detected': False, 'confidence': 0.0}