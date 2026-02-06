"""
NoDoze 工具函数
包括数据切分、专家权重表生成等辅助功能
"""

import pickle
import json
import os
from typing import Dict, List, Tuple
import torch
from pathlib import Path


def split_train_data_for_background(dataset_dir: str,
                                   train_file: str,
                                   background_file: str,
                                   background_ratio: float = 0.7):
    """
    从训练数据中切分出一部分作为背景数据
    
    Args:
        dataset_dir: 数据集目录
        train_file: 训练数据文件名
        background_file: 背景数据保存文件名
        background_ratio: 背景数据比例
    """
    train_path = os.path.join(dataset_dir, train_file)
    background_path = os.path.join(dataset_dir, background_file)
    
    if not os.path.exists(train_path):
        print(f"错误: 训练数据文件不存在: {train_path}")
        return
    
    train_data = torch.load(train_path, map_location='cpu')
    
    if isinstance(train_data, list):
        num_background = int(len(train_data) * background_ratio)
        background_data = train_data[:num_background]
        remaining_data = train_data[num_background:]
        
        torch.save(background_data, background_path)
        print(f"背景数据已保存: {background_path} ({len(background_data)} 个子图)")
        
        remaining_path = train_path.replace('.pt', '_remaining.pt')
        torch.save(remaining_data, remaining_path)
        print(f"剩余训练数据已保存: {remaining_path} ({len(remaining_data)} 个子图)")
    else:
        print("警告: 训练数据格式不是列表，无法切分")


def generate_expert_weights_template(output_path: str):
    """
    生成专家权重表模板
    
    Args:
        output_path: 输出文件路径
    """
    node_types = {
        'SUBJECT_PROCESS': 0,
        'FILE_OBJECT_FILE': 1,
        'FILE_OBJECT_UNIX_SOCKET': 2,
        'UnnamedPipeObject': 3,
        'NetFlowObject': 4,
        'FILE_OBJECT_DIR': 5
    }
    
    relation_types = [
        'read', 'write', 'execute', 'open', 'close',
        'create', 'delete', 'connect', 'accept', 'send', 'recv',
        'default'
    ]
    
    weights = {}
    
    
    weights[(0, 'execute', 1)] = 0.8  # SUBJECT_PROCESS -> execute -> FILE_OBJECT_FILE
    
    weights[(0, 'read', 1)] = 0.7  # SUBJECT_PROCESS -> read -> FILE_OBJECT_FILE
    
    weights[(0, 'write', 1)] = 0.7  # SUBJECT_PROCESS -> write -> FILE_OBJECT_FILE
    
    weights[(0, 'connect', 2)] = 0.6  # SUBJECT_PROCESS -> connect -> FILE_OBJECT_UNIX_SOCKET
    
    weights[(1, 'read', 1)] = 0.3  # FILE_OBJECT_FILE -> read -> FILE_OBJECT_FILE
    
    for src_type in node_types.values():
        for dst_type in node_types.values():
            for rel in relation_types:
                key = (src_type, rel, dst_type)
                if key not in weights:
                    weights[key] = 0.5
    
    with open(output_path, 'wb') as f:
        pickle.dump(weights, f)
    
    print(f"专家权重表模板已生成: {output_path}")
    print(f"包含 {len(weights)} 条权重记录")
    print("\n提示: 请根据实际的系统调用模式调整权重值")
    print("权重范围: 0.0 (非常异常) 到 1.0 (非常正常)")


def analyze_frequency_dict(freq_dict_path: str):
    """
    分析频率字典，输出统计信息
    
    Args:
        freq_dict_path: 频率字典文件路径
    """
    with open(freq_dict_path, 'rb') as f:
        freq_data = pickle.load(f)
    
    epsilon_freq = freq_data['epsilon_freq']
    src_rel_freq = freq_data['src_rel_freq']
    
    print("频率字典统计信息:")
    print(f"  不同的边类型组合数: {len(epsilon_freq)}")
    print(f"  不同的 (src_type, rel) 组合数: {len(src_rel_freq)}")
    print(f"  总边数: {sum(epsilon_freq.values())}")
    
    print("\nTop-10 最常见的边类型组合:")
    sorted_epsilon = sorted(epsilon_freq.items(), key=lambda x: x[1], reverse=True)
    for i, ((src_type, rel, dst_type), count) in enumerate(sorted_epsilon[:10], 1):
        print(f"  {i}. ({src_type}, {rel}, {dst_type}): {count} 次")
    
    print("\nTop-10 最常见的 (src_type, rel) 组合:")
    sorted_src_rel = sorted(src_rel_freq.items(), key=lambda x: x[1], reverse=True)
    for i, ((src_type, rel), count) in enumerate(sorted_src_rel[:10], 1):
        print(f"  {i}. ({src_type}, {rel}): {count} 次")


def create_node_score_config(output_path: str):
    """
    创建节点得分配置文件（JSON格式）
    
    Args:
        output_path: 输出文件路径
    """
    in_scores = {
        "0": 0.1,  # SUBJECT_PROCESS (Executable)
        "1": 0.5,  # FILE_OBJECT_FILE
        "2": 0.1,  # FILE_OBJECT_UNIX_SOCKET (Socket)
        "3": 0.5,  # UnnamedPipeObject
        "4": 0.5,  # NetFlowObject
        "5": 0.5   # FILE_OBJECT_DIR
    }
    
    out_scores = {
        "0": 0.1,  # SUBJECT_PROCESS
        "1": 0.5,  # FILE_OBJECT_FILE
        "2": 0.1,  # FILE_OBJECT_UNIX_SOCKET
        "3": 0.5,  # UnnamedPipeObject
        "4": 0.5,  # NetFlowObject
        "5": 0.5   # FILE_OBJECT_DIR
    }
    
    config = {
        "node_in_scores": in_scores,
        "node_out_scores": out_scores,
        "description": "节点类型得分配置",
        "node_types": {
            "0": "SUBJECT_PROCESS",
            "1": "FILE_OBJECT_FILE",
            "2": "FILE_OBJECT_UNIX_SOCKET",
            "3": "UnnamedPipeObject",
            "4": "NetFlowObject",
            "5": "FILE_OBJECT_DIR"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"节点得分配置文件已生成: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='NoDoze 工具函数')
    parser.add_argument('--action',
                       type=str,
                       choices=['split_background', 'generate_weights', 'analyze_freq', 'create_node_config'],
                       required=True,
                       help='要执行的操作')
    parser.add_argument('--dataset_dir',
                       type=str,
                       default='Datasets/Cadets_com',
                       help='数据集目录')
    parser.add_argument('--train_file',
                       type=str,
                       default='train_100nodes.pt',
                       help='训练数据文件名')
    parser.add_argument('--background_file',
                       type=str,
                       default='background_100nodes.pt',
                       help='背景数据文件名')
    parser.add_argument('--background_ratio',
                       type=float,
                       default=0.7,
                       help='背景数据比例')
    parser.add_argument('--output_path',
                       type=str,
                       required=True,
                       help='输出文件路径')
    parser.add_argument('--freq_dict_path',
                       type=str,
                       help='频率字典文件路径（用于分析）')
    
    args = parser.parse_args()
    
    if args.action == 'split_background':
        split_train_data_for_background(
            args.dataset_dir,
            args.train_file,
            args.background_file,
            args.background_ratio
        )
    elif args.action == 'generate_weights':
        generate_expert_weights_template(args.output_path)
    elif args.action == 'analyze_freq':
        if args.freq_dict_path:
            analyze_frequency_dict(args.freq_dict_path)
        else:
            print("错误: 需要指定 --freq_dict_path")
    elif args.action == 'create_node_config':
        create_node_score_config(args.output_path)
