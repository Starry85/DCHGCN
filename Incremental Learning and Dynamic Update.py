import torch
import torch.nn as nn
import numpy as np
import requests
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict, deque
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EvidenceCollector:
    """证据收集器 - 支持多数据集（PubMed+公开数据库）证据收集"""
    
    def __init__(self, 
                 db_path: str = "evidence_database.db",
                 pubmed_api_key: Optional[str] = None,
                 public_data_sources: Optional[Dict] = None):  # 新增：公开数据源配置
        self.db_path = db_path
        self.pubmed_api_key = pubmed_api_key
        self.public_data_sources = public_data_sources  # {'ctd': 'https://ctdbase.org/api', 'drugbank': 'https://drugbank.com/api'}
        self.setup_database()
    
    def setup_database(self):
        """设置证据数据库（新增多数据源标记字段）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_name TEXT NOT NULL,
                disease_name TEXT NOT NULL,  -- 适配单Disease列
                evidence_type TEXT NOT NULL,
                evidence_text TEXT,
                publication_date TEXT,
                source_name TEXT NOT NULL,  -- 新增：标记数据源（PubMed/CTD/DrugBank）
                journal_name TEXT,
                impact_factor REAL,
                sample_size INTEGER,
                study_type TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(drug_name, disease_name, evidence_type, publication_date, source_name)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_drug_disease_source ON evidence(drug_name, disease_name, source_name)')
        conn.commit()
        conn.close()
    
    def fetch_public_db_evidence(self, drug: str, disease: str) -> List[Dict]:
        """从公开数据库（CTD/DrugBank）获取证据（新增）"""
        public_evidence = []
        if not self.public_data_sources:
            return public_evidence
        
        # 1. 从CTD获取药物-疾病关联证据
        if 'ctd' in self.public_data_sources:
            try:
                ctd_url = f"{self.public_data_sources['ctd']}/interaction?drug={drug}&disease={disease}&format=json"
                response = requests.get(ctd_url, timeout=20)
                response.raise_for_status()
                ctd_data = response.json()
                
                for item in ctd_data.get('interactions', []):
                    evidence = {
                        'drug_name': drug,
                        'disease_name': disease,
                        'evidence_type': 'public_database',
                        'evidence_text': item.get('description', 'CTD drug-disease interaction'),
                        'publication_date': item.get('pubDate', ''),
                        'source_name': 'CTD',
                        'study_type': item.get('interactionType', 'association'),
                        'confidence_score': 0.7 if item.get('confidence', 'low') == 'high' else 0.4
                    }
                    public_evidence.append(evidence)
            except Exception as e:
                print(f"Error fetching CTD evidence: {e}")
        
        # 2. 从DrugBank获取药物适应症证据
        if 'drugbank' in self.public_data_sources:
            try:
                drugbank_url = f"{self.public_data_sources['drugbank']}/drug/{drug}?format=json"
                response = requests.get(drugbank_url, timeout=20)
                response.raise_for_status()
                drugbank_data = response.json()
                
                for indication in drugbank_data.get('indications', []):
                    if disease.lower() in indication.get('disease', '').lower():
                        evidence = {
                            'drug_name': drug,
                            'disease_name': disease,
                            'evidence_type': 'public_database',
                            'evidence_text': indication.get('description', 'DrugBank indication'),
                            'publication_date': drugbank_data.get('approvalDate', ''),
                            'source_name': 'DrugBank',
                            'study_type': 'approved_indication',
                            'confidence_score': 0.9  # DrugBank批准适应症置信度高
                        }
                        public_evidence.append(evidence)
            except Exception as e:
                print(f"Error fetching DrugBank evidence: {e}")
        
        return public_evidence
    
    def fetch_pubmed_evidence(self, drug: str, disease: str, max_results: int = 10) -> List[Dict]:
        """从PubMed获取证据（适配单Disease列）"""
        print(f"Fetching PubMed evidence for {drug} - {disease}")
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        query = f'("{drug}"[Title/Abstract]) AND ("{disease}"[Title/Abstract])'
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        if self.pubmed_api_key:
            params['api_key'] = self.pubmed_api_key
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            article_ids = data.get('esearchresult', {}).get('idlist', [])
            articles = self._fetch_article_details(article_ids)
            
            # 标记数据源为PubMed
            for article in articles:
                article.update({
                    'drug_name': drug,
                    'disease_name': disease,
                    'source_name': 'PubMed'
                })
            return articles
        except Exception as e:
            print(f"Error fetching PubMed evidence: {e}")
            return []
    
    def _fetch_article_details(self, article_ids: List[str]) -> List[Dict]:
        """获取文章详细信息（无核心修改）"""
        if not article_ids:
            return []
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {'db': 'pubmed', 'id': ','.join(article_ids), 'retmode': 'json'}
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            articles = []
            
            for article_id in article_ids:
                article_data = data.get('result', {}).get(article_id, {})
                article = {
                    'pubmed_id': article_id,
                    'title': article_data.get('title', ''),
                    'authors': [auth.get('name', '') for auth in article_data.get('authors', [])],
                    'journal': article_data.get('fulljournalname', ''),
                    'publication_date': article_data.get('pubdate', ''),
                    'doi': article_data.get('elocationid', ''),
                    'evidence_type': 'publication'
                }
                articles.append(article)
            return articles
        except Exception as e:
            print(f"Error fetching article details: {e}")
            return []
    
    def calculate_evidence_confidence(self, evidence: Dict) -> float:
        """计算证据置信度（新增数据源权重）"""
        confidence = 0.5  # 基础置信度
        
        # 1. 基于数据源（PubMed>CTD>DrugBank>其他）
        source_weights = {'PubMed': 0.2, 'CTD': 0.15, 'DrugBank': 0.15, 'other': 0.05}
        confidence += source_weights.get(evidence.get('source_name', 'other'), 0.05)
        
        # 2. 基于期刊影响因子/研究类型（原逻辑保留）
        impact_factor = evidence.get('impact_factor', 0)
        if impact_factor > 10:
            confidence += 0.3
        elif impact_factor > 5:
            confidence += 0.2
        
        study_type = evidence.get('study_type', '')
        if study_type == 'randomized_controlled_trial':
            confidence += 0.3
        elif study_type == 'approved_indication':
            confidence += 0.25
        
        return min(confidence, 1.0)
    
    def store_evidence(self, evidence_list: List[Dict]):
        """存储证据（适配多数据源）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for evidence in evidence_list:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO evidence 
                    (drug_name, disease_name, evidence_type, evidence_text, publication_date, 
                     source_name, journal_name, impact_factor, sample_size, study_type, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    evidence.get('drug_name'),
                    evidence.get('disease_name'),
                    evidence.get('evidence_type'),
                    evidence.get('evidence_text', ''),
                    evidence.get('publication_date'),
                    evidence.get('source_name', 'other'),
                    evidence.get('journal_name', ''),
                    evidence.get('impact_factor', 0),
                    evidence.get('sample_size', 0),
                    evidence.get('study_type', ''),
                    evidence.get('confidence_score', self.calculate_evidence_confidence(evidence))
                ))
            except Exception as e:
                print(f"Error storing evidence: {e}")
        
        conn.commit()
        conn.close()
    
    def get_recent_evidence(self, days: int = 30) -> List[Dict]:
        """获取最近N天的证据（多数据源）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT * FROM evidence 
            WHERE publication_date >= ? OR created_at >= ?
            ORDER BY confidence_score DESC, source_name DESC
        ''', (cutoff_date, cutoff_date))
        
        columns = [col[0] for col in cursor.description]
        evidence_list = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return evidence_list

class IncrementalLearningManager:
    """增量学习管理器（适配多数据集和单Disease列）"""
    
    def __init__(self, 
                 model: nn.Module,
                 evidence_collector: EvidenceCollector,
                 encoding_mapping: Dict,  # 新增：实体编码映射
                 learning_rate: float = 0.001,
                 buffer_size: int = 1000):
        
        self.model = model
        self.evidence_collector = evidence_collector
        self.encoding_mapping = encoding_mapping  # 药物/疾病/基因的编码映射
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        
        self.experience_buffer = deque(maxlen=buffer_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # 学习统计
        self.learning_stats = {
            'updates': 0,
            'last_update': None,
            'evidence_processed': 0,
            'source_evidence_count': defaultdict(int)  # 新增：统计各数据源证据数量
        }
    
    def collect_new_evidence(self, drug_disease_pairs: List[Tuple] = None):
        """收集新证据（多数据源：PubMed+CTD+DrugBank）"""
        print("Collecting new evidence...")
        if drug_disease_pairs is None:
            drug_disease_pairs = self._get_all_drug_disease_pairs()
        
        new_evidence = []
        for drug, disease in drug_disease_pairs[:10]:  # 限制数量避免API限流
            # 1. 从PubMed收集
            pubmed_evidence = self.evidence_collector.fetch_pubmed_evidence(drug, disease)
            new_evidence.extend(pubmed_evidence)
            self.learning_stats['source_evidence_count']['PubMed'] += len(pubmed_evidence)
            
            # 2. 从公开数据库收集
            public_evidence = self.evidence_collector.fetch_public_db_evidence(drug, disease)
            new_evidence.extend(public_evidence)
            for ev in public_evidence:
                self.learning_stats['source_evidence_count'][ev['source_name']] += 1
            
            time.sleep(1)  # 避免API限流
        
        # 存储新证据
        if new_evidence:
            self.evidence_collector.store_evidence(new_evidence)
            print(f"Collected {len(new_evidence)} new evidence items (sources: {dict(self.learning_stats['source_evidence_count'])})")
        
        return new_evidence
    
    def _get_all_drug_disease_pairs(self) -> List[Tuple]:
        """获取所有药物-疾病对（从编码映射中提取，适配单Disease列）"""
        drugs = list(self.encoding_mapping['drugs'].keys())
        diseases = list(self.encoding_mapping['diseases'].keys())
        # 随机生成100对（实际可从数据库读取）
        return [(random.choice(drugs), random.choice(diseases)) for _ in range(100)]
    
    def prepare_incremental_data(self, new_evidence: List[Dict]) -> Dict:
        """准备增量学习数据（适配单Disease列）"""
        positive_pairs = []
        negative_pairs = []
        confidence_scores = []
        
        for evidence in new_evidence:
            drug = evidence['drug_name']
            disease = evidence['disease_name']
            confidence = evidence['confidence_score']
            
            # 过滤无效实体（不在编码映射中的药物/疾病）
            if drug not in self.encoding_mapping['drugs'] or disease not in self.encoding_mapping['diseases']:
                continue
            
            if confidence > 0.7:  # 高置信度=正样本
                positive_pairs.append((drug, disease))
                confidence_scores.append(confidence)
            elif confidence < 0.3:  # 低置信度=负样本
                negative_pairs.append((drug, disease))
                confidence_scores.append(1 - confidence)
        
        return {
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs,
            'confidence_scores': confidence_scores
        }
    
    def incremental_update(self, incremental_data: Dict):
        """执行增量更新（适配单Disease列的嵌入提取）"""
        print("Performing incremental update...")
        self.model.train()
        
        if not incremental_data['positive_pairs'] and not incremental_data['negative_pairs']:
            print("No new valid data for incremental learning")
            return
        
        # 准备训练数据（从模型中提取真实嵌入，替换原随机生成）
        drug_embeddings = []
        disease_embeddings = []
        labels = []
        weights = []
        
        # 正样本
        for (drug, disease), conf in zip(incremental_data['positive_pairs'], incremental_data['confidence_scores']):
            drug_idx = self.encoding_mapping['drugs'][drug]
            disease_idx = self.encoding_mapping['diseases'][disease]
            # 假设模型的 hetero_data 已保存节点嵌入（实际需从模型/数据中获取）
            drug_emb = self.model.hetero_data['drug'].x[drug_idx].detach()
            disease_emb = self.model.hetero_data['disease'].x[disease_idx].detach()
            
            drug_embeddings.append(drug_emb)
            disease_embeddings.append(disease_emb)
            labels.append(1.0)
            weights.append(conf)
        
        # 负样本
        for (drug, disease), conf in zip(incremental_data['negative_pairs'], incremental_data['confidence_scores']):
            drug_idx = self.encoding_mapping['drugs'][drug]
            disease_idx = self.encoding_mapping['diseases'][disease]
            drug_emb = self.model.hetero_data['drug'].x[drug_idx].detach()
            disease_emb = self.model.hetero_data['disease'].x[disease_idx].detach()
            
            drug_embeddings.append(drug_emb)
            disease_embeddings.append(disease_emb)
            labels.append(0.0)
            weights.append(conf)
        
        # 转换为Tensor并训练
        drug_tensor = torch.stack(drug_embeddings)
        disease_tensor = torch.stack(disease_embeddings)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        
        # 前向传播（调用模型的预测头）
        combined = torch.cat([drug_tensor, disease_tensor], dim=1)
        predictions = self.model.drug_disease_predictor(combined)
        
        # 加权损失
        loss = self.criterion(predictions.squeeze(), labels_tensor)
        weighted_loss = (loss * weights_tensor).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新统计
        self.learning_stats['updates'] += 1
        self.learning_stats['last_update'] = datetime.now()
        self.learning_stats['evidence_processed'] += len(drug_embeddings)
        
        print(f"Incremental update completed. Weighted Loss: {weighted_loss.item():.4f}")
        return weighted_loss.item()
    
    # 以下方法（experience_replay、adaptive_learning_rate等）无核心修改，仅适配数据格式
    def experience_replay(self, batch_size: int = 32):
        if len(self.experience_buffer) < batch_size:
            return
        print("Performing experience replay...")
        
        batch = random.sample(self.experience_buffer, batch_size)
        drug_embeddings = [exp['drug_embedding'] for exp in batch]
        disease_embeddings = [exp['disease_embedding'] for exp in batch]
        labels = [exp['label'] for exp in batch]
        
        drug_tensor = torch.stack(drug_embeddings)
        disease_tensor = torch.stack(disease_embeddings)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        
        combined = torch.cat([drug_tensor, disease_tensor], dim=1)
        predictions = self.model.drug_disease_predictor(combined)
        loss = self.criterion(predictions.squeeze(), labels_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"Experience replay completed. Loss: {loss.item():.4f}")
    
    def add_to_experience_buffer(self, drug_embedding, disease_embedding, label, confidence):
        self.experience_buffer.append({
            'drug_embedding': drug_embedding,
            'disease_embedding': disease_embedding,
            'label': label,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
    
    def adaptive_learning_rate(self, performance_metric: float):
        if performance_metric < 0.7:
            new_lr = self.learning_rate * 0.5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate adjusted to: {new_lr}")
    
    def continuous_learning_loop(self, update_interval_hours: int = 24):
        print("Starting continuous learning loop...")
        while True:
            try:
                new_evidence = self.collect_new_evidence()
                if new_evidence:
                    incremental_data = self.prepare_incremental_data(new_evidence)
                    loss = self.incremental_update(incremental_data)
                    self.experience_replay()
                    if loss and loss > 1.0:
                        self.adaptive_learning_rate(0.6)
                
                print(f"Waiting {update_interval_hours} hours for next update...")
                time.sleep(update_interval_hours * 3600)
            except KeyboardInterrupt:
                print("Continuous learning interrupted")
                break
            except Exception as e:
                print(f"Error in continuous learning: {e}")
                time.sleep(3600)

class ModelVersionManager:
    """模型版本管理器（无核心修改，仅适配多数据集版本标记）"""
    
    def __init__(self, model_save_dir: str = "model_versions"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.version_history = []
    
    def save_model_version(self, model: nn.Module, version_info: Dict):
        """新增：版本信息包含数据源统计"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v{len(self.version_history) + 1}_{timestamp}"
        model_path = self.model_save_dir / f"{version_id}.pth"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'version_info': version_info,
            'timestamp': timestamp,
            'data_sources': version_info.get('data_sources', {})  # 新增：记录数据源
        }, model_path)
        
        version_record = {
            'version_id': version_id,
            'timestamp': timestamp,
            'model_path': str(model_path),
            **version_info
        }
        self.version_history.append(version_record)
        self._save_version_history()
        print(f"Model version {version_id} saved (data sources: {version_info.get('data_sources', {})})")
        return version_id
    
    def _save_version_history(self):
        history_path = self.model_save_dir / "version_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.version_history, f, indent=2, default=str)
    
    def load_model_version(self, version_id: str, model: nn.Module) -> nn.Module:
        version_record = next((v for v in self.version_history if v['version_id'] == version_id), None)
        if not version_record:
            raise ValueError(f"Version {version_id} not found")
        
        checkpoint = torch.load(version_record['model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model version {version_id} loaded (data sources: {checkpoint.get('data_sources', {})})")
        return model
    
    def get_latest_version(self) -> Optional[Dict]:
        return self.version_history[-1] if self.version_history else None