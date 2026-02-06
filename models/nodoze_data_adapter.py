"""
NoDoze 数据适配器
用于从当前项目的数据格式中提取节点类型和边类型信息
"""

import torch
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class NoDozeDataAdapter:
    """
    数据适配器：将当前项目的数据格式转换为 NoDoze 所需的格式
    """
    
    NODE_TYPE_MAPPING = {
        'SUBJECT_PROCESS': 0,
        'FILE_OBJECT_FILE': 1,
        'FILE_OBJECT_UNIX_SOCKET': 2,
        'UnnamedPipeObject': 3,
        'NetFlowObject': 4,
        'FILE_OBJECT_DIR': 5
    }
    
    def __init__(self, 
                 dataset_dir: str,
                 node_type_attr: str = 'node_type',
                 edge_type_attr: str = 'edge_attr',
                 action_attr: str = 'action'):
        """
        初始化数据适配器
        
        Args:
            dataset_dir: 数据集目录路径
            node_type_attr: 节点类型属性名称（在 Data 对象中）
            edge_type_attr: 边类型属性名称（在 Data 对象中）
            action_attr: action 属性名称（在原始数据中）
        """
        self.dataset_dir = Path(dataset_dir)
        self.node_type_attr = node_type_attr
        self.edge_type_attr = edge_type_attr
        self.action_attr = action_attr
        
        self.node_id_to_type_cache: Optional[Dict] = None
        self.edge_to_action_cache: Optional[Dict] = None
    
    def load_pt_file(self, file_path: str) -> List:
        """
        加载 .pt 文件（可能包含单个 Data 对象或 Data 对象列表）
        
        Args:
            file_path: .pt 文件路径
            
        Returns:
            Data 对象列表
        """
        data = torch.load(file_path, map_location='cpu')
        
        if isinstance(data, list):
            return data
        elif hasattr(data, 'x'):
            return [data]
        else:
            raise ValueError(f"无法识别的数据格式: {file_path}")
    
    def extract_node_types_from_data(self, data_list: List) -> Dict[int, int]:
        """
        从 Data 对象列表中提取节点类型映射
        
        Args:
            data_list: Data 对象列表
            
        Returns:
            节点ID到节点类型的映射 {node_id: node_type}
        """
        node_id_to_type = {}
        
        for data in data_list:
            if hasattr(data, self.node_type_attr):
                node_types = getattr(data, self.node_type_attr)
                for idx, node_type in enumerate(node_types):
                    node_id_to_type[idx] = int(node_type.item() if torch.is_tensor(node_type) else node_type)
            elif hasattr(data, 'y') and len(data.y.shape) == 1:
                node_types = data.y
                for idx, node_type in enumerate(node_types):
                    node_id_to_type[idx] = int(node_type.item() if torch.is_tensor(node_type) else node_type)
        
        return node_id_to_type
    
    def extract_edge_types_from_data(self, data_list: List) -> Dict[Tuple[int, int], str]:
        """
        从 Data 对象列表中提取边类型映射
        
        Args:
            data_list: Data 对象列表
            
        Returns:
            边到边类型的映射 {(src_id, dst_id): edge_type}
        """
        edge_to_type = {}
        
        for data in data_list:
            if hasattr(data, self.edge_type_attr):
                edge_attr = getattr(data, self.edge_type_attr)
                edge_index = data.edge_index
                
                if edge_attr is not None and edge_index is not None:
                    num_edges = edge_index.shape[1]
                    for i in range(num_edges):
                        src_id = int(edge_index[0, i].item())
                        dst_id = int(edge_index[1, i].item())
                        edge_type = edge_attr[i] if i < len(edge_attr) else 'default'
                        edge_to_type[(src_id, dst_id)] = str(edge_type)
        
        return edge_to_type
    
    def load_original_dataframe(self, df_path: str) -> Optional[Dict]:
        """
        从原始 DataFrame (pkl) 文件中加载数据，提取节点类型和边类型信息
        
        Args:
            df_path: DataFrame 文件路径
            
        Returns:
            包含节点类型和边类型映射的字典
        """
        try:
            with open(df_path, 'rb') as f:
                df = pickle.load(f)
            
            node_id_to_type = {}
            edge_to_action = {}
            
            dummies = {
                'SUBJECT_PROCESS': 0,
                'FILE_OBJECT_FILE': 1,
                'FILE_OBJECT_UNIX_SOCKET': 2,
                'UnnamedPipeObject': 3,
                'NetFlowObject': 4,
                'FILE_OBJECT_DIR': 5
            }
            
            for _, row in df.iterrows():
                actor_id = row.get('actorID')
                object_id = row.get('objectID')
                action = row.get('action', 'default')
                
                if actor_id is not None:
                    actor_type = row.get('actor_type', 'SUBJECT_PROCESS')
                    node_id_to_type[actor_id] = dummies.get(actor_type, 0)
                
                if object_id is not None:
                    object_type = row.get('object', 'FILE_OBJECT_FILE')
                    node_id_to_type[object_id] = dummies.get(object_type, 1)
                
                if actor_id is not None and object_id is not None:
                    edge_to_action[(actor_id, object_id)] = str(action)
            
            self.node_id_to_type_cache = node_id_to_type
            self.edge_to_action_cache = edge_to_action
            
            return {
                'node_id_to_type': node_id_to_type,
                'edge_to_action': edge_to_action
            }
        
        except Exception as e:
            print(f"警告: 无法加载原始 DataFrame: {e}")
            return None
    
    def convert_data_to_nodoze_format(self, 
                                     data_path: str,
                                     use_original_df: bool = True,
                                     original_df_path: Optional[str] = None) -> List[Dict]:
        """
        将 .pt 文件中的数据转换为 NoDoze 所需的格式
        
        Args:
            data_path: .pt 文件路径
            use_original_df: 是否使用原始 DataFrame 来获取节点类型和边类型
            original_df_path: 原始 DataFrame 文件路径（如果 use_original_df=True）
            
        Returns:
            子图列表，每个子图包含：
            - 'nodes': 节点ID列表
            - 'node_types': 节点类型列表
            - 'edges': 边列表 [(src_id, dst_id)]
            - 'edge_types': 边类型列表
        """
        data_list = self.load_pt_file(data_path)
        
        if use_original_df and original_df_path:
            self.load_original_dataframe(original_df_path)
        
        subgraphs = []
        
        for data in data_list:
            num_nodes = data.x.shape[0] if hasattr(data, 'x') else 0
            nodes = list(range(num_nodes))
            
            node_types = []
            if hasattr(data, 'node_type'):
                node_type_tensor = getattr(data, 'node_type')
                if torch.is_tensor(node_type_tensor):
                    node_types = [int(t.item()) for t in node_type_tensor]
                else:
                    node_types = [int(t) for t in node_type_tensor]
            elif hasattr(data, self.node_type_attr):
                node_types = [int(t.item() if torch.is_tensor(t) else t) 
                             for t in getattr(data, self.node_type_attr)]
            elif self.node_id_to_type_cache:
                node_types = [self.node_id_to_type_cache.get(i, 0) for i in nodes]
            else:
                node_types = [0] * num_nodes
            
            edges = []
            edge_types = []
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                edge_index = data.edge_index
                num_edges = edge_index.shape[1]
                
                for i in range(num_edges):
                    src_id = int(edge_index[0, i].item())
                    dst_id = int(edge_index[1, i].item())
                    edges.append((src_id, dst_id))
                    
                    edge_type = 'default'
                    
                    if hasattr(data, 'edge_action'):
                        edge_action_list = getattr(data, 'edge_action')
                        if isinstance(edge_action_list, list) and i < len(edge_action_list):
                            edge_type = str(edge_action_list[i])
                    elif hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                        edge_attr = getattr(data, 'edge_attr')
                        if torch.is_tensor(edge_attr) and i < len(edge_attr):
                            edge_type_idx = int(edge_attr[i].item())
                            edge_type = str(edge_type_idx)
                    elif hasattr(data, self.edge_type_attr) and getattr(data, self.edge_type_attr) is not None:
                        edge_attr = getattr(data, self.edge_type_attr)
                        edge_type = edge_attr[i] if i < len(edge_attr) else 'default'
                        edge_type = str(edge_type.item() if torch.is_tensor(edge_type) else edge_type)
                    elif self.edge_to_action_cache:
                        edge_type = self.edge_to_action_cache.get((src_id, dst_id), 'default')
                    
                    edge_types.append(edge_type)
            
            subgraph = {
                'nodes': nodes,
                'node_types': node_types,
                'edges': edges,
                'edge_types': edge_types
            }
            subgraphs.append(subgraph)
        
        return subgraphs
    
    def load_benign_subgraphs_from_pt(self, 
                                     pt_file_path: str,
                                     use_original_df: bool = True,
                                     original_df_path: Optional[str] = None) -> List[Dict]:
        """
        从 .pt 文件中加载良性子图
        
        Args:
            pt_file_path: .pt 文件路径
            use_original_df: 是否使用原始 DataFrame
            original_df_path: 原始 DataFrame 文件路径
            
        Returns:
            良性子图列表
        """
        subgraphs = self.convert_data_to_nodoze_format(
            pt_file_path, 
            use_original_df=use_original_df,
            original_df_path=original_df_path
        )
        
        benign_subgraphs = []
        data_list = self.load_pt_file(pt_file_path)
        
        for i, (subgraph, data) in enumerate(zip(subgraphs, data_list)):
            is_benign = True
            if hasattr(data, '_VULN'):
                vuln = data._VULN
                if torch.is_tensor(vuln):
                    if vuln.dim() == 0 or (vuln.dim() == 1 and len(vuln) == 1):
                        is_benign = vuln.item() == 0
                    elif vuln.dim() == 1:
                        is_benign = (vuln == 0).all().item()
            
            if is_benign:
                benign_subgraphs.append(subgraph)
        
        return benign_subgraphs
    
    def load_malicious_subgraphs_from_pt(self, 
                                        pt_file_path: str,
                                        use_original_df: bool = True,
                                        original_df_path: Optional[str] = None) -> List[Tuple[int, Dict]]:
        """
        从 .pt 文件中加载恶意子图（包含攻击节点的子图）
        
        Args:
            pt_file_path: .pt 文件路径
            use_original_df: 是否使用原始 DataFrame
            original_df_path: 原始 DataFrame 文件路径
            
        Returns:
            恶意子图列表，每个元素是 (原始索引, 子图字典)
        """
        subgraphs = self.convert_data_to_nodoze_format(
            pt_file_path, 
            use_original_df=use_original_df,
            original_df_path=original_df_path
        )
        
        malicious_subgraphs = []
        data_list = self.load_pt_file(pt_file_path)
        
        for i, (subgraph, data) in enumerate(zip(subgraphs, data_list)):
            is_malicious = False
            if hasattr(data, '_VULN'):
                vuln = data._VULN
                if torch.is_tensor(vuln):
                    if vuln.dim() == 0 or (vuln.dim() == 1 and len(vuln) == 1):
                        is_malicious = vuln.item() != 0
                    elif vuln.dim() == 1:
                        is_malicious = (vuln != 0).any().item()
            
            if is_malicious:
                subgraph['malicious_nodes'] = []
                if hasattr(data, '_VULN') and torch.is_tensor(data._VULN) and data._VULN.dim() == 1:
                    subgraph['malicious_nodes'] = (data._VULN != 0).nonzero(as_tuple=True)[0].tolist()
                subgraph['original_idx'] = i
                malicious_subgraphs.append((i, subgraph))
        
        return malicious_subgraphs
    
    def suggest_data_reconstruction(self) -> str:
        """
        提供数据重构建议
        
        Returns:
            重构建议的字符串
        """
        suggestion = """
        ========================================
        NoDoze 数据重构建议
        ========================================
        
        当前数据格式分析：
        1. 您的数据保存在 .pt 文件中，包含 PyTorch Geometric 的 Data 对象
        2. 数据包含节点特征 (x) 和边索引 (edge_index)
        3. 但可能缺少节点类型和边类型信息
        
        建议的重构方案：
        
        方案1：从原始 DataFrame 恢复信息（推荐）
        ----------------------------------------
        1. 使用原始 test_df.pkl 文件构建节点类型和边类型映射
        2. 在保存 .pt 文件时，将节点类型和边类型信息也保存进去
        3. 修改数据预处理脚本，添加以下属性：
           - data.node_type: 节点类型张量
           - data.edge_attr: 边类型张量（action）
        
        方案2：从子图文本文件恢复信息
        ----------------------------------------
        1. 使用 com8_{N}nodes.txt 文件
        2. 结合原始的 test_df.pkl 来匹配节点ID和类型
        3. 构建映射表并保存为 pickle 文件
        
        方案3：使用专家知识权重表（临时方案）
        ----------------------------------------
        1. 基于领域知识构建 (src_type, rel, dst_type) -> weight 的映射
        2. 使用 NoDozeAnalyzer 的 use_expert_weights=True 选项
        3. 这样可以绕过频率统计，直接使用预定义的权重
        
        实施步骤（方案1）：
        ----------------------------------------
        1. 修改 com2data.ipynb 或相关数据预处理脚本
        2. 在构建 Data 对象时，添加：
           data.node_type = torch.tensor(node_types)
           data.edge_attr = torch.tensor(edge_types)
        3. 重新生成 .pt 文件
        
        临时解决方案：
        ----------------------------------------
        如果无法立即重构数据，可以使用：
        1. 从训练集中切分一部分作为背景数据
        2. 使用默认的节点类型（假设所有节点都是 SUBJECT_PROCESS）
        3. 使用默认的边类型（'default'）
        4. 使用专家权重表替代频率统计
        
        ========================================
        """
        return suggestion
