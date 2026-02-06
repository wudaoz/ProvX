"""
NoDoze: Combatting Threat Alert Fatigue with Automated Provenance Triage
Implementation based on NDSS 2019 paper

Core Algorithm:
1. Transfer Probability (M_epsilon): M_epsilon = |Freq(epsilon)| / |Freq_src_rel(epsilon)|
2. Regularity Score (RS): RS = IN(src) × M_epsilon × OUT(dst)
3. Anomaly Score (AS): AS = 1 - RS
"""

import torch
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import pickle
import os
from pathlib import Path


class NoDozeAnalyzer:
    """
    NoDoze 异常检测分析器
    
    通过统计边缘出现的频率来量化异常度
    """
    
    NODE_TYPES = {
        'SUBJECT_PROCESS': 0,
        'FILE_OBJECT_FILE': 1,
        'FILE_OBJECT_UNIX_SOCKET': 2,
        'UnnamedPipeObject': 3,
        'NetFlowObject': 4,
        'FILE_OBJECT_DIR': 5
    }
    
    NODE_TYPE_NAMES = {v: k for k, v in NODE_TYPES.items()}
    
    def __init__(self, 
                 node_in_scores: Optional[Dict[int, float]] = None,
                 node_out_scores: Optional[Dict[int, float]] = None,
                 default_in_score: float = 0.5,
                 default_out_score: float = 0.5,
                 use_expert_weights: bool = False,
                 expert_weights_path: Optional[str] = None):
        """
        初始化 NoDoze 分析器
        
        Args:
            node_in_scores: 节点类型的 IN 得分字典 {node_type_id: score}
            node_out_scores: 节点类型的 OUT 得分字典 {node_type_id: score}
            default_in_score: 默认 IN 得分（当节点类型未定义时使用）
            default_out_score: 默认 OUT 得分（当节点类型未定义时使用）
            use_expert_weights: 是否使用专家知识权重表替代频率统计
            expert_weights_path: 专家权重表文件路径
        """
        self.node_in_scores = node_in_scores or self._get_default_in_scores()
        self.node_out_scores = node_out_scores or self._get_default_out_scores()
        self.default_in_score = default_in_score
        self.default_out_score = default_out_score
        
        self.epsilon_freq: Dict[Tuple[int, str, int], int] = defaultdict(int)  # (src_type, rel, dst_type) -> count
        self.src_rel_freq: Dict[Tuple[int, str], int] = defaultdict(int)  # (src_type, rel) -> count
        
        self.use_expert_weights = use_expert_weights
        self.expert_weights: Optional[Dict[Tuple[int, str, int], float]] = None
        if use_expert_weights and expert_weights_path:
            self._load_expert_weights(expert_weights_path)
        elif use_expert_weights:
            self.expert_weights = self._get_default_expert_weights()
    
    def _get_default_in_scores(self) -> Dict[int, float]:
        """获取默认的节点 IN 得分"""
        return {
            0: 0.1,  # SUBJECT_PROCESS (Executable)
            1: 0.5,  # FILE_OBJECT_FILE
            2: 0.1,  # FILE_OBJECT_UNIX_SOCKET (Socket)
            3: 0.5,  # UnnamedPipeObject
            4: 0.5,  # NetFlowObject
            5: 0.5   # FILE_OBJECT_DIR
        }
    
    def _get_default_out_scores(self) -> Dict[int, float]:
        """获取默认的节点 OUT 得分"""
        return {
            0: 0.1,  # SUBJECT_PROCESS
            1: 0.5,  # FILE_OBJECT_FILE
            2: 0.1,  # FILE_OBJECT_UNIX_SOCKET
            3: 0.5,  # UnnamedPipeObject
            4: 0.5,  # NetFlowObject
            5: 0.5   # FILE_OBJECT_DIR
        }
    
    def _get_default_expert_weights(self) -> Dict[Tuple[int, str, int], float]:
        """
        获取默认的专家知识权重表
        这是一个示例，实际使用时应该基于领域知识构建
        """
        weights = {}
        for src_type in range(6):
            for dst_type in range(6):
                weights[(src_type, 'default', dst_type)] = 0.5
        return weights
    
    def _load_expert_weights(self, weights_path: str):
        """从文件加载专家权重表"""
        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as f:
                self.expert_weights = pickle.load(f)
            print(f"已加载专家权重表: {len(self.expert_weights)} 条记录")
        else:
            print(f"警告: 专家权重表文件不存在: {weights_path}，使用默认权重")
            self.expert_weights = self._get_default_expert_weights()
    
    def build_frequency_dict(self, 
                            benign_subgraphs: List[Dict],
                            node_type_mapping: Optional[Dict] = None,
                            edge_type_mapping: Optional[Dict] = None):
        """
        从良性子图数据构建频率字典
        
        Args:
            benign_subgraphs: 良性子图列表，每个子图包含：
                - 'nodes': 节点ID列表
                - 'node_types': 节点类型列表（可选）
                - 'edges': 边列表 [(src_id, dst_id, ...)]
                - 'edge_types': 边类型列表（可选，对应每条边的关系类型）
            node_type_mapping: 节点ID到节点类型的映射 {node_id: node_type_id}
            edge_type_mapping: 边到边类型的映射 {(src_id, dst_id): edge_type}
        """
        print("开始构建频率字典...")
        total_edges = 0
        
        for subgraph in benign_subgraphs:
            nodes = subgraph.get('nodes', [])
            edges = subgraph.get('edges', [])
            node_types = subgraph.get('node_types', [])
            edge_types = subgraph.get('edge_types', [])
            
            if not node_types and node_type_mapping:
                node_types = [node_type_mapping.get(nid, 0) for nid in nodes]
            
            if not edge_types:
                edge_types = ['default'] * len(edges)
            
            node_id_to_idx = {nid: idx for idx, nid in enumerate(nodes)}
            
            for i, edge in enumerate(edges):
                if len(edge) >= 2:
                    src_id, dst_id = edge[0], edge[1]
                    
                    src_idx = node_id_to_idx.get(src_id, -1)
                    dst_idx = node_id_to_idx.get(dst_id, -1)
                    
                    if src_idx >= 0 and dst_idx >= 0:
                        src_type = node_types[src_idx] if src_idx < len(node_types) else 0
                        dst_type = node_types[dst_idx] if dst_idx < len(node_types) else 0
                        rel = edge_types[i] if i < len(edge_types) else 'default'
                        
                        epsilon = (src_type, rel, dst_type)
                        self.epsilon_freq[epsilon] += 1
                        self.src_rel_freq[(src_type, rel)] += 1
                        total_edges += 1
        
        print(f"频率字典构建完成: 统计了 {total_edges} 条边，{len(self.epsilon_freq)} 种不同的边类型组合")
        return self.epsilon_freq, self.src_rel_freq
    
    def compute_transfer_probability(self, src_type: int, rel: str, dst_type: int) -> float:
        """
        计算转移概率 M_epsilon = |Freq(epsilon)| / |Freq_src_rel(epsilon)|
        
        Args:
            src_type: 源节点类型
            rel: 关系类型
            dst_type: 目标节点类型
            
        Returns:
            转移概率值
        """
        epsilon = (src_type, rel, dst_type)
        src_rel = (src_type, rel)
        
        if self.use_expert_weights and self.expert_weights:
            if epsilon in self.expert_weights:
                return self.expert_weights[epsilon]
            else:
                epsilon_default = (src_type, 'default', dst_type)
                if epsilon_default in self.expert_weights:
                    return self.expert_weights[epsilon_default]
                return 0.5
        else:
            freq_epsilon = self.epsilon_freq.get(epsilon, 0)
            freq_src_rel = self.src_rel_freq.get(src_rel, 1)
            
            if freq_src_rel == 0:
                return 0.0
            
            return freq_epsilon / freq_src_rel
    
    def compute_regularity_score(self, src_type: int, rel: str, dst_type: int) -> float:
        """
        计算正则得分 RS = IN(src) × M_epsilon × OUT(dst)
        
        Args:
            src_type: 源节点类型
            rel: 关系类型
            dst_type: 目标节点类型
            
        Returns:
            正则得分
        """
        in_score = self.node_in_scores.get(src_type, self.default_in_score)
        out_score = self.node_out_scores.get(dst_type, self.default_out_score)
        m_epsilon = self.compute_transfer_probability(src_type, rel, dst_type)
        
        rs = in_score * m_epsilon * out_score
        return rs
    
    def compute_anomaly_score(self, src_type: int, rel: str, dst_type: int) -> float:
        """
        计算异常分 AS = 1 - RS
        
        Args:
            src_type: 源节点类型
            rel: 关系类型
            dst_type: 目标节点类型
            
        Returns:
            异常分（0-1之间，越高越异常）
        """
        rs = self.compute_regularity_score(src_type, rel, dst_type)
        as_score = 1.0 - rs
        return max(0.0, min(1.0, as_score))
    
    def analyze_subgraph(self, 
                        subgraph: Dict,
                        node_type_mapping: Optional[Dict] = None,
                        edge_type_mapping: Optional[Dict] = None) -> List[Tuple[Tuple, float]]:
        """
        分析子图，计算每条边的异常分
        
        Args:
            subgraph: 子图字典，包含 'nodes', 'edges', 'node_types', 'edge_types'
            node_type_mapping: 节点ID到节点类型的映射
            edge_type_mapping: 边到边类型的映射
            
        Returns:
            列表 [(edge_tuple, anomaly_score), ...]，按异常分降序排列
        """
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        node_types = subgraph.get('node_types', [])
        edge_types = subgraph.get('edge_types', [])
        
        if not node_types and node_type_mapping:
            node_types = [node_type_mapping.get(nid, 0) for nid in nodes]
        elif not node_types:
            node_types = [0] * len(nodes)
        
        if not edge_types:
            edge_types = ['default'] * len(edges)
        
        node_id_to_idx = {nid: idx for idx, nid in enumerate(nodes)}
        
        edge_scores = []
        
        for i, edge in enumerate(edges):
            if len(edge) >= 2:
                src_id, dst_id = edge[0], edge[1]
                
                src_idx = node_id_to_idx.get(src_id, -1)
                dst_idx = node_id_to_idx.get(dst_id, -1)
                
                if src_idx >= 0 and dst_idx >= 0:
                    src_type = node_types[src_idx] if src_idx < len(node_types) else 0
                    dst_type = node_types[dst_idx] if dst_idx < len(node_types) else 0
                    rel = edge_types[i] if i < len(edge_types) else 'default'
                    
                    as_score = self.compute_anomaly_score(src_type, rel, dst_type)
                    edge_scores.append(((src_id, dst_id), as_score))
        
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        
        return edge_scores
    
    def get_top_k_anomalous_edges(self, 
                                  subgraph: Dict,
                                  k: int = 10,
                                  node_type_mapping: Optional[Dict] = None,
                                  edge_type_mapping: Optional[Dict] = None) -> List[Tuple[Tuple, float]]:
        """
        获取 Top-K 条最异常的边
        
        Args:
            subgraph: 子图字典
            k: 返回的边数量
            node_type_mapping: 节点ID到节点类型的映射
            edge_type_mapping: 边到边类型的映射
            
        Returns:
            Top-K 条边的列表 [(edge_tuple, anomaly_score), ...]
        """
        edge_scores = self.analyze_subgraph(subgraph, node_type_mapping, edge_type_mapping)
        return edge_scores[:k]
    
    def save_frequency_dict(self, save_path: str):
        """保存频率字典到文件"""
        freq_data = {
            'epsilon_freq': dict(self.epsilon_freq),
            'src_rel_freq': dict(self.src_rel_freq)
        }
        with open(save_path, 'wb') as f:
            pickle.dump(freq_data, f)
        print(f"频率字典已保存到: {save_path}")
    
    def load_frequency_dict(self, load_path: str):
        """从文件加载频率字典"""
        with open(load_path, 'rb') as f:
            freq_data = pickle.load(f)
        self.epsilon_freq = defaultdict(int, freq_data['epsilon_freq'])
        self.src_rel_freq = defaultdict(int, freq_data['src_rel_freq'])
        print(f"频率字典已从 {load_path} 加载: {len(self.epsilon_freq)} 种边类型组合")
