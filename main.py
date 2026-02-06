import os
import gc
import json
import random
import argparse
import warnings
from torch_geometric.data import Data
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import pickle
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
import torch_scatter
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from models.vul_detector import Detector
from helpers import utils
from sklearn.model_selection import train_test_split
from line_extract import get_dep_add_lines_bigvul
from dataset import ContractGraphDataset


from models.gnnexplainer import XGNNExplainer
from models.pgexplainer import XPGExplainer, PGExplainer_edges
from models.subgraphx import SubgraphX
from models.gnn_lrp import GNN_LRP
from models.deeplift import DeepLIFT
from models.gradcam import GradCAM
import pickle as pkl
from pathlib import Path
from models.ProvX import ProvX
from torch_geometric.nn import MessagePassing

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_metrics(y_true, y_pred):
    results = {
        'binary_precision': round(precision_score(y_true, y_pred, average='binary'), 4),
        'binary_recall': round(recall_score(y_true, y_pred, average='binary'), 4),
        'binary_f1': round(f1_score(y_true, y_pred, average='binary'), 4),
    }
    return results

def calculate_metrics_detailed(y_true, y_pred, average='binary'):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    if len(y_true_np) == 0 or len(y_pred_np) == 0:
        return {
            f'{average}_precision': 0.0,
            f'{average}_recall': 0.0,
            f'{average}_f1': 0.0,
            f'{average}_accuracy': 0.0,
        }
    
    return {
        f'{average}_precision': round(precision_score(y_true_np, y_pred_np, average=average, zero_division=0), 4),
        f'{average}_recall': round(recall_score(y_true_np, y_pred_np, average=average, zero_division=0), 4),
        f'{average}_f1': round(f1_score(y_true_np, y_pred_np, average=average, zero_division=0), 4),
        f'{average}_accuracy': round(accuracy_score(y_true_np, y_pred_np), 4)
    }

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def train(args, train_dataloader, valid_dataloader, test_dataloader, model):
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.max_steps * 0.1),
        num_training_steps=args.max_steps
    )

    # checkpoint_last = os.path.join(args.model_checkpoint_dir, 'checkpoint-last')
    # optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')

    # if os.path.exists(optimizer_last):
    #     try:
    #         saved_optimizer = torch.load(optimizer_last, map_location=args.device)
    #         if len(saved_optimizer['param_groups']) != len(optimizer.param_groups):
    #         optimizer.load_state_dict(saved_optimizer)
    #     except Exception as e:

    # if os.path.exists(scheduler_last):
    #     try:
    #         scheduler.load_state_dict(torch.load(scheduler_last, map_location=args.device))

    #     except Exception as e:





    # print(f"Current optimizer param groups: {len(optimizer.param_groups)}")
    # print(f"Loaded optimizer param groups: {len(torch.load(optimizer_last)['param_groups'])}")

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    
    # if os.path.exists(scheduler_last):
    #     scheduler.load_state_dict(torch.load(scheduler_last, map_location=args.device), strict=False)
    # if os.path.exists(optimizer_last):
    #     optimizer.load_state_dict(torch.load(optimizer_last, map_location=args.device), strict=False)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader.dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Total optimization steps = {args.max_steps}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    global_step = args.start_step
    tr_loss = logging_loss = avg_loss = 0.0
    tr_nb = tr_num = train_loss = 0
    best_acc = 0.0

    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training epoch {idx}")
        tr_num = 0
        train_loss = 0
        


        for step, batch_data in enumerate(bar):
            try:
                torch.cuda.empty_cache()
                if batch_data is None or not hasattr(batch_data, 'x') or not hasattr(batch_data, 'edge_index'):
                    continue

                batch_data = batch_data.to(args.device)
                x = batch_data.x
                edge_index = batch_data.edge_index.long()
                batch = batch_data.batch


                num_nodes = x.size(0)
                if edge_index.max() >= num_nodes:
                    print(f"Warning: edge_index max value ({edge_index.max()}) >= num_nodes ({num_nodes})")
                    continue

                edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
                edge_index = coalesce(edge_index)


                # labels = torch_scatter.segment_csr(batch_data._VULN, batch_data.ptr).long()

                labels = global_max_pool(batch_data._VULN, batch).long()
                # with open("labels.txt", "a") as f:
                #     for label in labels:
                #         f.write(f"{label.item()}\n")


                model.train()

                logits = model(x, edge_index, batch)

                
                loss = F.cross_entropy(logits, labels)


                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                
                # for name, param in model.named_parameters():
                #     if param.grad is not None:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                tr_num += 1
                train_loss += loss.item()
                avg_loss = round(train_loss / tr_num, 5)
                bar.set_description(f"epoch {idx} loss {avg_loss}")

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    if tr_nb == 0:
                        avg_loss = tr_loss
                    else:
                        avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logging_loss = tr_loss
                        tr_nb = global_step

                    if args.save_steps > 0 and global_step % args.save_steps == 0:

                        results = evaluate(args, valid_dataloader, model)
                        print(f"  Valid acc: {results['eval_acc']:.4f}")



                        if results['eval_acc'] > best_acc:
                            best_acc = results['eval_acc']
                            print("  " + "*" * 20)
                            print(f"  Best acc: {best_acc:.4f}")
                            print("  " + "*" * 20)

                            checkpoint_prefix = 'checkpoint-best-acc'
                            output_dir = os.path.join(args.model_checkpoint_dir, checkpoint_prefix)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_path = os.path.join(output_dir, 'model.bin')
                            torch.save(model_to_save.state_dict(), output_path)
                            print(f"Saving model checkpoint to {output_path}")

                            test_result = evaluate(args, test_dataloader, model)
                            for key, value in test_result.items():
                                print(f"  {key} = {value:.4f}")

                if step % 100 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"Error in batch: {e}")
                print(f"Batch info: nodes={x.size(0)}, edges={edge_index.size(1)}")
                continue

        bar.close()
        
        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.model_checkpoint_dir, checkpoint_prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))

    return global_step, tr_loss / global_step


def train_node_level(args, train_dataloader, valid_dataloader, test_dataloader_for_eval, model):
    if args.max_steps <= 0 :
        args.max_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    if args.warmup_steps <= 0 and args.max_steps > 0 :
         args.warmup_steps = int(args.max_steps * 0.1)


    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    print("***** 开始节点级别训练 (Running Node-Level Training) *****")
    print(f"  图样本数量 (Num graphs) = {len(train_dataloader.dataset)}")
    print(f"  训练轮次 (Num Epochs) = {args.num_train_epochs}")
    print(f"  总优化步数 (Total optimization steps) = {args.max_steps}")
    print(f"  梯度累积步数 (Gradient Accumulation steps) = {args.gradient_accumulation_steps}")

    global_step = 0
    best_graph_f1_from_nodes = 0.0
    model.zero_grad()

    for epoch_idx in range(int(args.num_train_epochs)):
        epoch_loss = 0.0
        num_batches_processed = 0
        
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"训练 Epoch {epoch_idx + 1}/{int(args.num_train_epochs)}")
        for step, batch_data in enumerate(bar):
            model.train()
            if not hasattr(batch_data, 'x') or not hasattr(batch_data, 'edge_index') or not hasattr(batch_data, '_VULN') or not hasattr(batch_data, 'batch'):
                print("批次数据缺少必要属性 (x, edge_index, _VULN, or batch)，跳过此批次。")
                continue

            batch_data = batch_data.to(args.device)
            x = batch_data.x
            edge_index = batch_data.edge_index.long()
            batch_assignment = batch_data.batch
            
            node_labels_true = batch_data._VULN.long()

            if x is None or x.size(0) == 0:
                continue
            if node_labels_true is None or node_labels_true.size(0) == 0:
                continue
            if node_labels_true.size(0) != x.size(0):
                print(f"节点特征数量 ({x.size(0)}) 与节点标签数量 ({node_labels_true.size(0)}) 不匹配，跳过。")
                continue

            # edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
            # edge_index = coalesce(edge_index)

            node_logits = model(x, edge_index, batch_assignment)

            loss = F.cross_entropy(node_logits, node_labels_true)

            # if args.node_class_weights is not None:
            #     loss = F.cross_entropy(node_logits, node_labels_true, weight=args.node_class_weights)
            # else:
            #     loss = F.cross_entropy(node_logits, node_labels_true)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches_processed +=1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                current_lr = scheduler.get_last_lr()[0] if scheduler else args.learning_rate
                bar.set_postfix_str(f"loss: {loss.item():.4f}, avg_epoch_loss: {epoch_loss/num_batches_processed:.4f}, lr: {current_lr:.2e}")

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    pass

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    val_results = evaluate_node_and_graph_level(args, valid_dataloader, model)
                    print(f"\nEpoch {epoch_idx + 1}, Global Step {global_step}:")
                    print(f"  验证集节点级别: Acc: {val_results.get('node_binary_accuracy', 0):.4f}, F1: {val_results.get('node_binary_f1', 0):.4f}")
                    print(f"  验证集图级别 (来自节点): Acc: {val_results.get('graph_binary_accuracy', 0):.4f}, F1: {val_results.get('graph_binary_f1', 0):.4f}")

                    current_graph_f1 = val_results.get('graph_binary_f1', 0)
                    if current_graph_f1 > best_graph_f1_from_nodes:
                        best_graph_f1_from_nodes = current_graph_f1
                        print(f"  **** 新的最佳验证集图 F1 分数: {best_graph_f1_from_nodes:.4f} ****")
                        
                        output_dir = os.path.join(args.model_checkpoint_dir, 'checkpoint-best-graph-f1')
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.bin'))
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        print(f"  最佳模型已保存至 {output_dir}")
                        
                        if test_dataloader_for_eval:
                           test_results_best = evaluate_node_and_graph_level(args, test_dataloader_for_eval, model)
                           print("  --- 使用最佳模型在测试集上的结果 ---")
                           print(f"    测试集节点级别: Acc: {test_results_best.get('node_binary_accuracy',0):.4f}, F1: {test_results_best.get('node_binary_f1',0):.4f}")
                           print(f"    测试集图级别 (来自节点): Acc: {test_results_best.get('graph_binary_accuracy',0):.4f}, F1: {test_results_best.get('graph_binary_f1',0):.4f}")
        
        avg_epoch_tr_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        print(f"Epoch {epoch_idx + 1} 结束. 平均训练损失: {avg_epoch_tr_loss:.4f}")
        
        output_dir_last = os.path.join(args.model_checkpoint_dir, 'checkpoint-last')
        os.makedirs(output_dir_last, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir_last, 'model.bin'))
        torch.save(args, os.path.join(output_dir_last, 'training_args.bin'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir_last, 'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(output_dir_last, 'scheduler.pt'))
        print(f"最新模型已保存至 {output_dir_last}")

    return global_step, avg_epoch_tr_loss

import os
import gc
import json
import random
import argparse
import warnings
from torch_geometric.data import Data
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import pickle
from sklearn.metrics import * # Ensure confusion_matrix is available
import torch
import torch.nn.functional as F
# from transformers import BertModel, BertTokenizer
from torch_geometric.nn import global_max_pool
from torch_geometric.data import DataLoader # Corrected: Was Batch, should be DataLoader
from torch_geometric.data import Batch # For the collate_fn example, if it creates Batch objects
from torch_geometric.utils import *
import torch_scatter
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from models.vul_detector import Detector
from helpers import utils
from sklearn.model_selection import train_test_split
from line_extract import get_dep_add_lines_bigvul
# from graph_dataset import VulGraphDataset, collate
from dataset import ContractGraphDataset
from models.gnnexplainer import XGNNExplainer
from models.pgexplainer import XPGExplainer, PGExplainer_edges
from models.subgraphx import SubgraphX
from models.gnn_lrp import GNN_LRP
from models.deeplift import DeepLIFT
from models.gradcam import GradCAM
import pickle as pkl
from pathlib import Path
from torch_geometric.nn import MessagePassing

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_metrics(y_true, y_pred):
    results = {
        'binary_precision': round(precision_score(y_true, y_pred, average='binary', zero_division=0), 4),
        'binary_recall': round(recall_score(y_true, y_pred, average='binary', zero_division=0), 4), # TPR is the same as recall
        'binary_f1': round(f1_score(y_true, y_pred, average='binary', zero_division=0), 4),
    }
    return results

def calculate_metrics_detailed(y_true, y_pred, average='binary'):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    if len(y_true_np) == 0 or len(y_pred_np) == 0:
        base_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
        }
        if average == 'binary': # Add TP/TN/FP/FN for binary case
            base_metrics.update({
                'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                'FPR': 0.0
            })
        return {f'{average}_{k}': v for k, v in base_metrics.items()}
    
    metrics_dict = {
        f'{average}_precision': round(precision_score(y_true_np, y_pred_np, average=average, zero_division=0), 4),
        f'{average}_recall': round(recall_score(y_true_np, y_pred_np, average=average, zero_division=0), 4), # TPR
        f'{average}_f1': round(f1_score(y_true_np, y_pred_np, average=average, zero_division=0), 4),
        f'{average}_accuracy': round(accuracy_score(y_true_np, y_pred_np), 4)
    }

    if average == 'binary':
        # Always use labels=[0, 1] for binary confusion matrix to ensure consistent 2x2 shape
        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        fpr = 0.0
        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
        
        metrics_dict.update({
            f'{average}_TP': int(tp),
            f'{average}_TN': int(tn),
            f'{average}_FP': int(fp),
            f'{average}_FN': int(fn),
            f'{average}_FPR': round(fpr, 4),
            f'{average}_TPR': metrics_dict[f'{average}_recall'] # TPR is recall
        })
    return metrics_dict

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def train(args, train_dataloader, valid_dataloader, test_dataloader, model):
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.max_steps * 0.1),
        num_training_steps=args.max_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader.dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Total optimization steps = {args.max_steps}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    global_step = args.start_step
    tr_loss = logging_loss = avg_loss = 0.0
    tr_nb = tr_num = train_loss = 0
    best_acc = 0.0 # This seems to be graph accuracy from 'evaluate'

    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training epoch {idx}")
        tr_num = 0
        train_loss = 0
        


        for step, batch_data in enumerate(bar):
            try:
                torch.cuda.empty_cache()
                if batch_data is None or not hasattr(batch_data, 'x') or not hasattr(batch_data, 'edge_index'):
                    continue

                batch_data = batch_data.to(args.device)
                x = batch_data.x
                edge_index = batch_data.edge_index.long()
                batch = batch_data.batch


                num_nodes = x.size(0)
                if edge_index.max() >= num_nodes:
                    print(f"Warning: edge_index max value ({edge_index.max()}) >= num_nodes ({num_nodes})")
                    continue

                edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
                edge_index = coalesce(edge_index)
                
                labels = global_max_pool(batch_data._VULN, batch).long()

                model.train()

                logits, _ = model(x, edge_index, batch)
                loss = F.cross_entropy(logits, labels)


                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                tr_num += 1
                train_loss += loss.item()
                avg_loss = round(train_loss / tr_num, 5)
                bar.set_description(f"epoch {idx} loss {avg_loss}")

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    if tr_nb == 0:
                        avg_loss = tr_loss
                    else:
                        avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logging_loss = tr_loss
                        tr_nb = global_step

                    if args.save_steps > 0 and global_step % args.save_steps == 0:

                        results = evaluate(args, valid_dataloader, model)
                        print(f"\n   Valid Results (Graph Level) at Global Step {global_step}:")
                        print(f"     Acc: {results.get('eval_acc', 0):.4f}, AUC: {results.get('eval_auc', 0):.4f}, F1: {results.get('binary_f1', 0):.4f}, Recall (TPR): {results.get('binary_recall', 0):.4f}, Precision: {results.get('binary_precision', 0):.4f}")
                        print(f"     TP: {results.get('TP',0)}, TN: {results.get('TN',0)}, FP: {results.get('FP',0)}, FN: {results.get('FN',0)}")
                        print(f"     FPR: {results.get('FPR',0):.4f}")


                        current_eval_metric_for_best = results['eval_acc'] # Or change to F1: results['binary_f1']
                        if current_eval_metric_for_best > best_acc:
                            best_acc = current_eval_metric_for_best
                            print("  " + "*" * 20)
                            print(f"  Best acc (graph): {best_acc:.4f}") # Or other metric like F1
                            print("  " + "*" * 20)

                            checkpoint_prefix = 'checkpoint-best-acc'
                            output_dir = os.path.join(args.model_checkpoint_dir, checkpoint_prefix)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_path = os.path.join(output_dir, 'model.bin')
                            torch.save(model_to_save.state_dict(), output_path)
                            print(f"Saving model checkpoint to {output_path}")

                            test_result = evaluate(args, test_dataloader, model)
                            print("  --- Test results with best validation model (Graph Level) ---")
                            for key, value in sorted(test_result.items()):
                                if isinstance(value, float):
                                    print(f"    {key} = {value:.4f}")
                                else:
                                    print(f"    {key} = {value}")


                if step % 100 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"Error in batch: {e}")
                if 'x' in locals() and 'edge_index' in locals() :
                    print(f"Batch info: nodes={x.size(0)}, edges={edge_index.size(1)}")
                continue

        bar.close()
        
        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.model_checkpoint_dir, checkpoint_prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.bin')) # Save model
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))

    return global_step, tr_loss / global_step if global_step > 0 else 0


def evaluate(args, eval_dataloader, model):
    model.eval()
    all_preds_list = []
    all_labels_list = []
    all_probs_list = []

    with torch.no_grad():
        for step, batch_data in enumerate(eval_dataloader):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            labels = global_max_pool(batch_data._VULN, batch).long()
            
            logits, _ = model(x, edge_index, batch)
            probs = F.softmax(logits, dim=1) # shape: [batch_size, num_classes]
            y_pred = probs.argmax(dim=1)  

            all_preds_list.append(y_pred.cpu().numpy())
            all_labels_list.append(labels.cpu().numpy())
            
            all_probs_list.append(probs[:, 1].cpu().numpy())

    all_preds_np = np.concatenate(all_preds_list, 0)
    all_labels_np = np.concatenate(all_labels_list, 0)
    all_probs_np = np.concatenate(all_probs_list, 0)
    
    eval_acc = np.mean(all_labels_np == all_preds_np)
    
    result = {
        "eval_acc": round(eval_acc, 4),
    }

    if len(np.unique(all_labels_np)) > 1:
        eval_auc = roc_auc_score(all_labels_np, all_probs_np)
        result['eval_auc'] = round(eval_auc, 4)
    else:
        result['eval_auc'] = 0.0

    base_eval_results = calculate_metrics(all_labels_np, all_preds_np)
    result.update(base_eval_results)

    if len(all_labels_np) > 0 and len(all_preds_np) > 0:
        cm = confusion_matrix(all_labels_np, all_preds_np, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        result['TP'] = int(tp)
        result['TN'] = int(tn)
        result['FP'] = int(fp)
        result['FN'] = int(fn)

        fpr = 0.0
        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
        result['FPR'] = round(fpr, 4)
        result['TPR'] = result['binary_recall']
    else:
        result.update({'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'FPR': 0.0, 'TPR': 0.0})

    return result

def evaluate_node_and_graph_level(args, eval_dataloader, model):
    model.eval()
    
    all_node_labels_true_list = []
    all_node_preds_list = []
    
    all_graph_labels_true_list = []
    all_graph_preds_from_nodes_list = []

    with torch.no_grad():
        for batch_data in tqdm(eval_dataloader, desc="评估中 (Evaluating)", leave=False):
            if not hasattr(batch_data, 'x') or not hasattr(batch_data, 'edge_index') or not hasattr(batch_data, '_VULN') or not hasattr(batch_data, 'batch'):
                continue
            batch_data = batch_data.to(args.device)
            x, edge_index, batch_assignment = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            
            node_labels_true_batch = batch_data._VULN.long()

            if x is None or x.size(0) == 0 or node_labels_true_batch is None or node_labels_true_batch.size(0) == 0 or node_labels_true_batch.size(0) != x.size(0):
                continue

            node_logits = model(x, edge_index, batch_assignment)
            node_preds_batch = torch.argmax(node_logits, dim=1)
            
            all_node_labels_true_list.extend(node_labels_true_batch.cpu().tolist())
            all_node_preds_list.extend(node_preds_batch.cpu().tolist())
            
            true_graph_labels_batch = global_max_pool(node_labels_true_batch, batch_assignment)
            
            graph_preds_from_nodes_batch = global_max_pool(node_preds_batch, batch_assignment)
            
            all_graph_labels_true_list.extend(true_graph_labels_batch.cpu().tolist())
            all_graph_preds_from_nodes_list.extend(graph_preds_from_nodes_batch.cpu().tolist())

    results = {}
    if all_node_labels_true_list:
        node_metrics = calculate_metrics_detailed(all_node_labels_true_list, all_node_preds_list, average='binary')
        results.update({f"node_{k}": v for k, v in node_metrics.items()})
    else:
        results.update({f"node_{k}": 0.0 for k in ['binary_precision', 'binary_recall', 'binary_f1', 'binary_accuracy']})

    if all_graph_labels_true_list:
        graph_metrics_from_nodes = calculate_metrics_detailed(all_graph_labels_true_list, all_graph_preds_from_nodes_list, average='binary')
        results.update({f"graph_{k}": v for k, v in graph_metrics_from_nodes.items()})
    else:
        results.update({f"graph_{k}": 0.0 for k in ['binary_precision', 'binary_recall', 'binary_f1', 'binary_accuracy']})
        
    return results

def gen_exp_lines(edge_index, edge_weight, index, num_nodes, lines):
    temp = torch.zeros_like(edge_weight).to(edge_index.device)
    temp[index] = edge_weight[index]

    adj_mask = torch.sparse_coo_tensor(edge_index, temp, [num_nodes, num_nodes])
    adj_mask_binary = to_dense_adj(edge_index[:, temp != 0], max_num_nodes=num_nodes).squeeze(0)

    out_degree = torch.sum(adj_mask_binary, dim=1)
    out_degree[out_degree == 0] = 1e-8
    in_degree = torch.sum(adj_mask_binary, dim=0)
    in_degree[in_degree == 0] = 1e-8

    line_importance_init = torch.ones(num_nodes).unsqueeze(-1).to(edge_index.device)
    line_importance_out = torch.spmm(adj_mask, line_importance_init) / out_degree.unsqueeze(-1)
    line_importance_in = torch.spmm(adj_mask.T, line_importance_init) / in_degree.unsqueeze(-1)
    line_importance = line_importance_out + line_importance_in

    ret = sorted(
        list(
            zip(
                line_importance.squeeze(-1).cpu().numpy(),
                lines,
            )
        ),
        reverse=True,
    )

    filtered_ret = []
    for i in ret:
        if i[0] > 0:
            filtered_ret.append(int(i[1]))

    return filtered_ret

def eval_exp(exp_saved_path, model, correct_lines, args):
    graph_exp_list = torch.load(exp_saved_path, map_location=args.device)
    print("Number of explanations:", len(graph_exp_list))

    accuracy = 0
    precisions = []
    recalls = []
    F1s = []
    pn = []

    
    for graph in graph_exp_list:
        graph.to(args.device)
        x, edge_index, edge_weight, pred, batch = graph.x, graph.edge_index.long(), graph.edge_weight, graph.pred, graph.batch

        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        exp_label_lines = correct_lines[int(sampleid)]
        # exp_label_lines = list(exp_label_lines["removed"]) + list(exp_label_lines["depadd"])
        exp_label_lines = list(exp_label_lines["removed"])
        if len(edge_weight) > args.KM:
            value, index = torch.topk(edge_weight, k=args.KM)
        else:
            index = torch.arange(edge_weight.shape[0])
        temp = torch.ones_like(edge_weight)
        temp[index] = 0
        cf_index = temp != 0

        lines = graph._LINE.cpu().numpy()
        exp_lines = gen_exp_lines(edge_index, edge_weight, index, x.shape[0], lines)

        for i, l in enumerate(exp_lines):
            if l in exp_label_lines:
                accuracy += 1
                break

        hit = 0
        for i, l in enumerate(exp_lines):
            if l in exp_label_lines:
                hit += 1
        if hit != 0:
            precision = hit / len(exp_lines)
            recall = hit / len(exp_label_lines)
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(f1)

        fac_edge_index = edge_index[:, index]
        fac_edge_index, _ = add_self_loops(fac_edge_index, num_nodes=x.shape[0])  # add self-loop
        fac_logits, _ = model(x, fac_edge_index, batch)
        fac_pred = F.one_hot(torch.argmax(fac_logits, dim=-1), 2)[0][0]

        cf_edge_index = edge_index[:, cf_index]
        cf_edge_index, _ = add_self_loops(cf_edge_index, num_nodes=x.shape[0])  # add self-loop


        important_nodes = torch.unique(edge_index[:, index])
        cf_x = x.clone()
        cf_x[important_nodes] = 0.0

        cf_logits, _ = model(cf_x, cf_edge_index, batch)
        # cf_pred = F.one_hot(torch.argmax(cf_logits, dim=-1), 2)[0][0]
        cf_probs = F.softmax(cf_logits, dim=1)
        cf_pred = cf_probs.argmax(dim=1)
        # print("cf_pred:", cf_pred, "pred:", pred)

        pn.append(int(cf_pred != pred))

        if args.case_sample_ids and str(sampleid) in args.case_sample_ids:
            case_saving_dir = str(utils.cache_dir() / f"cases")
            case_graph_saving_path = f"{case_saving_dir}/{args.gnn_model}_{args.ipt_method}_{sampleid}.pt"
            torch.save(graph, case_graph_saving_path)
            print(f"Saving {str(sampleid)} in {case_graph_saving_path}!")

    accuracy = round(accuracy / len(graph_exp_list), 4)
    print("Accuracy:", accuracy)
    precision = round(np.mean(precisions), 4)
    print("Precision:", precision)
    recall = round(np.mean(recalls), 4)
    print("Recall:", recall)
    f1 = round(np.mean(F1s), 4)
    print("F1:", f1)
    PN = round(sum(pn) / len(pn), 4)
    print("Probability of Necessity:", PN)

        # KM_index_map = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6, 16: 7, 18: 8, 20: 9}
        # if os.path.isfile(para_saving_path):
        #     result = json.load(open(para_saving_path, "r"))
        # else:
        #     GNN_models = ["GCNConv", "GatedGraphConv", "GINConv", "GraphConv"]
        #     metrics = [r"Accuracy", r"Precision", r"Recall", r"$F_1$", r"PN"]
        #     result = {}
        #     for GNN_model in GNN_models:
        #         result[GNN_model] = {}
        #         for metric in metrics:
        #             result[GNN_model][metric] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        # result[args.gnn_model][r"Accuracy"][KM_index_map[args.KM]] = accuracy
        # result[args.gnn_model][r"Precision"][KM_index_map[args.KM]] = precision
        # result[args.gnn_model][r"Recall"][KM_index_map[args.KM]] = recall
        # result[args.gnn_model][r"$F_1$"][KM_index_map[args.KM]] = f1
        # result[args.gnn_model][r"PN"][KM_index_map[args.KM]] = PN
        # json.dump(result, open(para_saving_path, "w"))

    # results_saving_dir = str(utils.cache_dir() / f"results")
    # results_saving_dir = str(utils.cache_dir() / args.dataset_name / f"results")
    # if not os.path.exists(results_saving_dir):
    #     os.makedirs(results_saving_dir)
    # results_saving_path = os.path.join(results_saving_dir, f"{args.ipt_method}.res")
    # KM_index_map = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6, 16: 7, 18: 8, 20: 9, 25: 10}
    # if os.path.isfile(results_saving_path):
    #     result = json.load(open(results_saving_path, "r"))
    # else:
    #     GNN_models = ["GCNConv", "GatedGraphConv", "GINConv", "GraphConv"]
    #     metrics = [r"Accuracy", r"Precision", r"Recall", r"$F_1$", r"PN"]
    #     result = {}
    #     for GNN_model in GNN_models:
    #         result[GNN_model] = {}
    #         for metric in metrics:
    #             result[GNN_model][metric] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # result[args.gnn_model][r"Accuracy"][KM_index_map[args.KM]] = accuracy
    # result[args.gnn_model][r"Precision"][KM_index_map[args.KM]] = precision
    # result[args.gnn_model][r"Recall"][KM_index_map[args.KM]] = recall
    # result[args.gnn_model][r"$F_1$"][KM_index_map[args.KM]] = f1
    # result[args.gnn_model][r"PN"][KM_index_map[args.KM]] = PN
    # json.dump(result, open(results_saving_path, "w"))

# ==============================================================================
# ==============================================================================
def eval_exp_node(exp_saved_path, model, args):
    """
    通过移除Top-K重要节点来评估解释的必要性 (PN_Node)。
    """
    graph_exp_list = torch.load(exp_saved_path, map_location=args.device)
    print("***** Running Node-based Explanation Evaluation *****")
    print("Number of explanations:", len(graph_exp_list))

    K_N = args.KN 

    pn_node = []

    for graph in tqdm(graph_exp_list, desc=f"Evaluating PN_Node (K_N={K_N})"):
        graph.to(args.device)
        x, edge_index, edge_weight, pred, batch = graph.x, graph.edge_index.long(), graph.edge_weight, graph.pred, graph.batch
        num_nodes = x.shape[0]


        node_importance = torch.zeros(num_nodes, device=args.device)
        node_importance = torch_scatter.scatter_add(edge_weight, edge_index[0], out=node_importance)
        node_importance = torch_scatter.scatter_add(edge_weight, edge_index[1], out=node_importance)

        if len(node_importance) > K_N:
            _, top_k_node_indices = torch.topk(node_importance, k=K_N)
        else:
            top_k_node_indices = torch.arange(num_nodes)

        node_mask = torch.ones(num_nodes, dtype=torch.bool, device=args.device)
        node_mask[top_k_node_indices] = False

        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]

        cf_edge_index_node = edge_index[:, edge_mask]
        

        # cf_logits_node = model(x, cf_edge_index_node, batch)
        # cf_pred_node = cf_logits_node.argmax(dim=-1)
        cf_x = x.clone()
        cf_x[top_k_node_indices] = 0.0
        cf_logits = model(cf_x, cf_edge_index_node, batch)
        cf_probs = F.softmax(cf_logits, dim=1)
        cf_pred = cf_probs.argmax(dim=1)
        # print("cf_pred:", cf_pred, "pred:", pred)


        if cf_pred != pred:
            pn_node.append(1)
        else:
            pn_node.append(0)

    PN_Node = round(sum(pn_node) / len(pn_node), 4) if pn_node else 0
    print(f"Probability of Necessity (Node-based) with K_N={K_N}: {PN_Node}")
    
    return PN_Node

# def eval_exp(exp_saved_path, model, correct_lines, args):
#     graph_exp_list = torch.load(exp_saved_path, map_location=args.device)

#     precisions_for_tps = []
#     recalls_for_tps = []
#     f1s_for_tps = []
    

#     pn_values = []

#         graph_explanation_data.to(args.device)
#         x, edge_index, edge_weight, pred_by_gnn, batch = \
#             graph_explanation_data.x, graph_explanation_data.edge_index.long(), \
#             graph_explanation_data.edge_weight, graph_explanation_data.pred, graph_explanation_data.batch
        
#         if not hasattr(graph_explanation_data, '_VULN') or graph_explanation_data._VULN is None:
#             continue
#         true_graph_label = 1 if (graph_explanation_data._VULN == 1).any() else 0
        
#         sampleid = graph_explanation_data._SAMPLE.max().int().item()

#         if len(edge_weight) > args.KM:
#             _, top_k_edge_indices = torch.topk(edge_weight, k=args.KM)
#         else:
#             top_k_edge_indices = torch.arange(edge_weight.shape[0])
        
#         node_original_lines = graph_explanation_data._LINE.cpu().numpy()
#         exp_lines = gen_exp_lines(edge_index, edge_weight, top_k_edge_indices, x.shape[0], node_original_lines)

#             num_true_positives_explained += 1
            
#             if int(sampleid) not in correct_lines:
#             else:
#                 exp_label_lines_removed = list(correct_lines[int(sampleid)].get("removed", []))

#             hit_count = 0
#                 for line in exp_lines:
#                     if line in exp_label_lines_removed:
#                         hit_count += 1
            
#             if hit_count > 0:
#                 accuracy_hits_for_tps += 1
#                 current_accuracy = 1
#             else:
#                 current_accuracy = 0
            
#             current_precision = 0.0
#             if len(exp_lines) > 0:
#                 current_precision = hit_count / len(exp_lines)
#                 current_precision = 1.0

#             current_recall = 0.0
#             if len(exp_label_lines_removed) > 0:
#                 current_recall = hit_count / len(exp_label_lines_removed)
#                 current_recall = 1.0

#             current_f1 = 0.0
#             if (current_precision + current_recall) > 0:
#                 current_f1 = (2 * current_precision * current_recall) / (current_precision + current_recall)
#                  current_f1 = 1.0
            

#             precisions_for_tps.append(current_precision)
#             recalls_for_tps.append(current_recall)
#             f1s_for_tps.append(current_f1)

        


#         # cf_edge_index_with_loops, _ = add_self_loops(cf_edge_index, num_nodes=x.shape[0])
#         # cf_logits = model(x, cf_edge_index_with_loops, batch)

#         pn_values.append(int(cf_predicted_class != original_predicted_class))



#     final_accuracy = 0.0
#     if num_true_positives_explained > 0:
#         final_accuracy = round(accuracy_hits_for_tps / num_true_positives_explained, 4)
    
#     final_precision = round(np.mean(precisions_for_tps) if precisions_for_tps else 0.0, 4)
#     final_recall = round(np.mean(recalls_for_tps) if recalls_for_tps else 0.0, 4)
#     final_f1 = round(np.mean(f1s_for_tps) if f1s_for_tps else 0.0, 4)
    

    
    
#     return {
#         "Accuracy_TP": final_accuracy,
#         "Precision_TP": final_precision,
#         "Recall_TP": final_recall,
#         "F1_TP": final_f1,
#         "Probability_of_Necessity": final_pn,
#         "Num_TP_Explained": num_true_positives_explained,
#         "Total_Graphs_Explained": len(graph_exp_list)
#     }



# def eval_exp(exp_saved_path, model, correct_lines, args):
#     graph_exp_list = torch.load(exp_saved_path, map_location=args.device)

#     accuracy = 0
#     precisions = []
#     recalls = []
#     F1s = []
#     pn = []

#     tp_fac = 0  # True Positives on factual subgraph
#     fn_fac = 0  # False Negatives on factual subgraph
#     # fp_fac and tn_fac will be 0 because all graphs in graph_exp_list have true label = 1

#     for graph in graph_exp_list:
#         graph.to(args.device)
#         x, edge_index, edge_weight, pred_original_graph, batch = graph.x, graph.edge_index.long(), graph.edge_weight, graph.pred, graph.batch
        
        
#         sampleid = graph._SAMPLE.max().int().item()
#         exp_label_lines = correct_lines[int(sampleid)]
#         exp_label_lines = list(exp_label_lines["removed"])

#         if len(edge_weight) > args.KM:
#             value, index = torch.topk(edge_weight, k=args.KM)
#         else:
#             index = torch.arange(edge_weight.shape[0])
        
#         temp = torch.ones_like(edge_weight)
#         temp[index] = 0
#         cf_index = temp != 0

#         lines = graph._LINE.cpu().numpy()
#         exp_lines = gen_exp_lines(edge_index, edge_weight, index, x.shape[0], lines)

#         for i, l in enumerate(exp_lines):
#             if l in exp_label_lines:
#                 accuracy += 1
#                 break

#         hit = 0
#         for i, l in enumerate(exp_lines):
#             if l in exp_label_lines:
#                 hit += 1
        
#         if len(exp_lines) > 0 and len(exp_label_lines) > 0 and hit > 0:
#             precision = hit / len(exp_lines)
#             recall = hit / len(exp_label_lines)
#             f1 = (2 * precision * recall) / (precision + recall)
#             recall = 1.0
#             f1 = 1.0
#             precision = 0.0
#             recall = 0.0
#             f1 = 0.0
            
#         precisions.append(precision)
#         recalls.append(recall)
#         F1s.append(f1)

#         fac_edge_index = edge_index[:, index]
#         fac_edge_index, _ = add_self_loops(fac_edge_index, num_nodes=x.shape[0])
#             fac_logits = model(x, fac_edge_index, batch)
        
#         predicted_class_on_fac = torch.argmax(fac_logits, dim=-1).item()

#             tp_fac += 1
#             fn_fac += 1

#         cf_edge_index = edge_index[:, cf_index]
#         cf_edge_index, _ = add_self_loops(cf_edge_index, num_nodes=x.shape[0])
#         with torch.no_grad():
#             cf_logits = model(x, cf_edge_index, batch)
        
#         predicted_class_on_cf = torch.argmax(cf_logits, dim=-1).item()
        
#         pn.append(int(predicted_class_on_cf != pred_original_graph))


#         if args.case_sample_ids and str(sampleid) in args.case_sample_ids:
#             case_saving_dir = str(utils.cache_dir() / f"cases")
#             case_graph_saving_path = f"{case_saving_dir}/{args.gnn_model}_{args.ipt_method}_{sampleid}.pt"
#             torch.save(graph, case_graph_saving_path)
#             print(f"Saving {str(sampleid)} in {case_graph_saving_path}!")

#     num_evaluated_graphs = len(graph_exp_list)
#     if num_evaluated_graphs == 0:
#         print("Warning: No graphs were evaluated for explanations. Skipping metric calculation.")

#     print("\n--- Explanation Line-Level Metrics (Averaged) ---")
#     print("Accuracy (at least 1 line correct):", accuracy_avg)
#     precision_avg = round(np.mean(precisions), 4)
#     print("Precision (line-level):", precision_avg)
#     recall_avg = round(np.mean(recalls), 4)
#     print("Recall (line-level):", recall_avg)
#     f1_avg = round(np.mean(F1s), 4)
#     print("F1-score (line-level):", f1_avg)
#     print("Probability of Necessity (PN):", PN_avg)

#     print("\n--- Factual Subgraph Prediction Metrics ---")
#     print(f"Total Explained Graphs (True Label is Malicious, Original Model Predicted Malicious): {num_evaluated_graphs}")
#     print(f"  TP_fac (Factual Subgraph Predicted Malicious): {tp_fac} ({tp_fac/num_evaluated_graphs:.2%})")
#     print(f"  FN_fac (Factual Subgraph Predicted Benign):    {fn_fac} ({fn_fac/num_evaluated_graphs:.2%})")

#     if args.hyper_para:
#     else:

def gnnexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explainer = XGNNExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05,
        coff_edge_size=0.001, coff_edge_ent=0.001
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        logits = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        probs = F.softmax(logits, dim=1)
        print("probs shape:", probs.shape)
        predicted_class = probs.argmax(dim=1)
        print("predicted_class shape:", predicted_class.shape)
        exp_prob_label = predicted_class.item()
        print("exp_prob_label shape:", exp_prob_label)
        # prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        # exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or exp_prob_label != 1:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None,
                                                                                     num_classes=args.num_classes)
        edge_weight = edge_masks[exp_prob_label]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label)
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def provx_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explainer = ProvX(
        model=model,
        epochs=200,  # Example
        lr=0.05,
        alpha=args.cfexp_alpha,   # Example
        L1_dist=args.cfexp_L1, # Example
        solidification_factor=args.solidification_factor,
        solidification_stage_start_ratio=args.solidification_stage_start_ratio,
        confident_threshold_low=0.05,
        confident_threshold_high=0.95,
        explain_graph=True
    )
    explainer.device = args.device



    for graph in test_dataset:
        # if not hasattr(graph, '_VULN') or graph._VULN is None:
        #     print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) due to missing or None _VULN attribute.")
        #     continue
        # if not isinstance(graph._VULN, torch.Tensor):
        #     print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) _VULN is not a tensor.")
        #     continue
        # if not (graph._VULN == 1).any():
        #     # print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) as _VULN contains no 1s. _VULN: {graph._VULN}")
        #     continue
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        logits = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        probs = F.softmax(logits, dim=1)
        # exp_prob_label = probs.argmax(dim=1).item() 
        predicted_class = probs.argmax(dim=1)
        exp_prob_label = predicted_class.item()


        if label != 1 or exp_prob_label != 1:
            continue

        # if label != 1:
        #     continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None,
                                                                                     num_classes=args.num_classes)
        print("edge_masks shape:", type(edge_masks))
        if isinstance(edge_masks, list):
            edge_masks = torch.stack(edge_masks)
        edge_weight = 1 - edge_masks[exp_prob_label]
        # edge_weight = 1 - edge_masks[:, exp_prob_label]


        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label)
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list



def pgexplainer_run(args, model, eval_model, train_dataset, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    input_dim = args.gnn_hidden_size * 2

    pgexplainer = XPGExplainer(model=model, in_channels=input_dim, device=args.device, explain_graph=True, epochs=10,
                               lr=0.005,
                               coff_size=0.01, coff_ent=5e-4, sample_bias=0.0, t0=5.0, t1=1.0)
    pgexplainer_saving_path = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/pgexplainer.bin")
    if os.path.isfile(pgexplainer_saving_path) and not args.ipt_update:
        print("Load saved PGExplainer model...")
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))
    else:
        if hasattr(train_dataset, 'dataset'):
            actual_train_dataset = train_dataset.dataset
        else:
            actual_train_dataset = train_dataset

        print(f"Type of object passed to train_explanation_network: {type(actual_train_dataset)}")
        pgexplainer.train_explanation_network(actual_train_dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))

    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer, model=eval_model)
    pgexplainer_edges.device = pgexplainer.device

    for graph in test_dataset:
        if not hasattr(graph, '_VULN') or graph._VULN is None:
            print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) due to missing or None _VULN attribute.")
            continue
        if not isinstance(graph._VULN, torch.Tensor):
            print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) _VULN is not a tensor.")
            continue
        if not (graph._VULN == 1).any():
            # print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) as _VULN contains no 1s. _VULN: {graph._VULN}")
            continue
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        logits = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        probs = F.softmax(logits, dim=1)
        print("probs shape:", probs.shape)
        # exp_prob_label = probs.argmax(dim=1).item()
        predicted_class = probs.argmax(dim=1)
        print("predicted_class shape:", predicted_class.shape)
        exp_prob_label = probs.argmax(dim=1).item()
        print("exp_prob_label shape:", exp_prob_label)
        # prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        # exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        # if label != 1 or prob[0][0] < prob[0][1]:
        #     continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = pgexplainer_edges(x, edge_index,
                                                                                             num_classes=args.num_classes,
                                                                                             sparsity=0.5)
        edge_weight = edge_masks[exp_prob_label]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label)
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def subgraphx_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()

    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/subgraphx")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    subgraphx = SubgraphX(model, args.num_classes, args.device, explain_graph=True,
                          verbose=True, c_puct=10.0, rollout=5, high2low=False, min_atoms=5, expand_atoms=14,
                          reward_method='gnn_score', subgraph_building_method='zero_filling',
                          save_dir=explanation_saving_dir)

    for graph in test_dataset:
        if not hasattr(graph, '_VULN') or graph._VULN is None:
            print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) due to missing or None _VULN attribute.")
            continue
        if not isinstance(graph._VULN, torch.Tensor):
            print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) _VULN is not a tensor.")
            continue
        if not (graph._VULN == 1).any():
            # print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) as _VULN contains no 1s. _VULN: {graph._VULN}")
            continue
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        logits = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        probs = F.softmax(logits, dim=1)
        print("probs shape:", probs.shape)
        # exp_prob_label = probs.argmax(dim=1).item()
        predicted_class = probs.argmax(dim=1)
        print("predicted_class shape:", predicted_class.shape)
        exp_prob_label = predicted_class.item()
        print("exp_prob_label shape:", exp_prob_label)

        if label != 1 or exp_prob_label != 1:
            continue
        # prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        # exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        # if label != 1 or prob[0][0] < prob[0][1]:
        #     continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        saved_MCTSInfo_list = None
        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'),
                                             map_location=args.device)
            print(f"load example {sampleid}.") 
        explain_result = subgraphx.explain(x, edge_index, label=exp_prob_label, node_idx=0,
                                           saved_MCTSInfo_list=saved_MCTSInfo_list)
        torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))
        node_weight = torch.zeros(x.shape[0])
        for item in explain_result:
            node_weight[item['coalition']] += item['P']
        node_weight = node_weight / len(explain_result)
        edge_index, _ = remove_self_loops(edge_index.detach().cpu())
        edge_weight = node_weight[edge_index] + node_weight[edge_index]
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label)
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def gnn_lrp_run(args, model, test_dataset, correct_lines):
    # for name, parameter in model.named_parameters():
    #     print(name)

    graph_exp_list = []
    visited_sampleids = set()

    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/gnn_lrp")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    gnnlrp_explainer = GNN_LRP(model, explain_graph=True)

    for graph in test_dataset:
        if not hasattr(graph, '_VULN') or graph._VULN is None:
            print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) due to missing or None _VULN attribute.")
            continue
        if not isinstance(graph._VULN, torch.Tensor):
            print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) _VULN is not a tensor.")
            continue
        if not (graph._VULN == 1).any():
            # print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) as _VULN contains no 1s. _VULN: {graph._VULN}")
            continue
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        logits = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        probs = F.softmax(logits, dim=1)
        print("probs shape:", probs.shape)
        # exp_prob_label = probs.argmax(dim=1).item()
        predicted_class = probs.argmax(dim=1)
        print("predicted_class shape:", predicted_class.shape)
        exp_prob_label = predicted_class.item()
        print("exp_prob_label shape:", exp_prob_label)
        # prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        # exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        # if label != 1 or prob[0][0] < prob[0][1]:
        #     continue
        # print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])

        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            edge_masks, self_loop_edge_index = torch.load(
                os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'), map_location=args.device)
            print(f"load example {sampleid}.")
        else:
            walks, edge_masks, related_preds, self_loop_edge_index = gnnlrp_explainer(x, edge_index, sparsity=0.5,
                                                                                      num_classes=args.num_classes)
            torch.save((edge_masks, self_loop_edge_index),
                       os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))

        edge_weight = edge_masks[exp_prob_label]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label)
        graph_exp_list.append(graph.detach().clone().cpu())
        visited_sampleids.add(sampleid)

        del graph
        gc.collect()

    return graph_exp_list


def deeplift_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    deep_lift = DeepLIFT(model, explain_graph=True)

    for graph in test_dataset:
        # if not hasattr(graph, '_VULN') or graph._VULN is None:
        #     print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) due to missing or None _VULN attribute.")
        #     continue
        # if not isinstance(graph._VULN, torch.Tensor):
        #     print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) _VULN is not a tensor.")
        #     continue
        # if not (graph._VULN == 1).any():
        #     # print(f"Skipping graph (sampleid: {graph._SAMPLE.max().int().item() if hasattr(graph, '_SAMPLE') else 'Unknown'}) as _VULN contains no 1s. _VULN: {graph._VULN}")
        #     continue
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        logits = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        probs = F.softmax(logits, dim=1)
        print("probs shape:", probs.shape)
        # exp_prob_label = probs.argmax(dim=1).item()
        predicted_class = probs.argmax(dim=1)
        print("predicted_class shape:", predicted_class.shape)
        exp_prob_label = predicted_class.item()
        # prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        # exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or exp_prob_label != 1:
            continue
        # print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = deep_lift(x, edge_index, sparsity=0.5,
                                                                                     num_classes=args.num_classes)
        edge_weight = edge_masks[exp_prob_label]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label)
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def gradcam_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    gc_explainer = GradCAM(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        logits = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        probs = F.softmax(logits, dim=1)
        print("probs shape:", probs.shape)
        # exp_prob_label = probs.argmax(dim=1).item()
        predicted_class = probs.argmax(dim=1)
        print("predicted_class shape:", predicted_class.shape)
        exp_prob_label = predicted_class.item()
        print("exp_prob_label shape:", exp_prob_label)
        # prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        # exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or exp_prob_label != 1:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = gc_explainer(x, edge_index, sparsity=0.5,
                                                                                        num_classes=args.num_classes)
        edge_weight = edge_masks[exp_prob_label]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label)
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def collate(batch):
    """
    自定义批处理函数
    """
    max_nodes = max(data.x.size(0) for data in batch)

    batch_x = []
    batch_edge_index = []
    batch_mask = []
    batch_y = []

    cum_nodes = 0
    for i, data in enumerate(batch):
        num_nodes = data.x.size(0)

        padded_x = torch.zeros((max_nodes, data.x.size(1)))
        padded_x[:num_nodes] = data.x
        batch_x.append(padded_x)

        if data.edge_index.size(1) > 0:
            edge_index = data.edge_index + cum_nodes
            batch_edge_index.append(edge_index)

        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:num_nodes] = True
        batch_mask.append(mask)

        if hasattr(data, '_VULN'):
            batch_y.append(data._VULN)

        cum_nodes += num_nodes

    batch_x = torch.stack(batch_x)
    batch_edge_index = torch.cat(batch_edge_index, dim=1) if batch_edge_index else torch.empty((2, 0))
    batch_mask = torch.stack(batch_mask)
    if batch_y:
        batch_y = torch.stack(batch_y)

    return Batch(
        x=batch_x,
        edge_index=batch_edge_index,
        mask=batch_mask,
        y=batch_y if batch_y else None,
        batch=torch.arange(len(batch)).repeat_interleave(max_nodes)
    )


def load_datasets(args):
    train_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='train'
    )

    valid_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='val'
    )

    test_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='test'
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate
    )

    return train_dataloader, valid_dataloader, test_dataloader



# def label_to_tensor(label):
#     label_mapping = {
#         'reentrancy': 0,
#         'arithmetic': 1,
#         'time_manipulation': 2,
#     }
#     return torch.tensor([label['reentrancy'], label['arithmetic'], label['time_manipulation']], dtype=torch.long)
def label_to_tensor(label_dict):
    """将标签字典转换为0/1标量"""
    return 1 if any(label_dict.values()) else 0

class FeatureExtractor:
    def __init__(self, bert_path="/root/autodl-tmp/bert-tiny", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path).to(device)
        self.transform = nn.Linear(128, 4).to(device)
        
    def get_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]
        
    def get_transformed_features(self, text):
        features = self.get_features(text)
        features = features.squeeze(0)
        return self.transform(features)
    
    def batch_get_features(self, texts, batch_size=32):
        features_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", max_length=512, 
                                  truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, 128]
                transformed_features = self.transform(batch_features)  # [batch_size, 4]
                features_list.append(transformed_features)
                
        return torch.cat(features_list, dim=0)

def get_bert_features(text, feature_extractor):
    features = feature_extractor.get_features(text)
    features = features.squeeze(0)
    transform = nn.Linear(features.shape[-1], 4)
    return transform(features)
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_partition_data(data_dir, partition):
    """直接加载指定分区的数据集"""
    # return torch.load(os.path.join(data_dir, partition, "data.pt"))
    
    return torch.load(os.path.join(data_dir, f"{partition}.pt"))

def load_existing_labels():
    saved_path = Path("statement_labels.pkl")
    with open(saved_path, "rb") as f:
        labels = pkl.load(f)
    return labels


def save_embeddings_for_plotting(args, model, dataloader):
    """
    运行模型，提取所有图的高维嵌入和真实标签，并将其保存到.npz文件中。
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    print("正在为后续绘图提取高维嵌入...")
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="提取嵌入和标签"):
            batch_data = batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            
            try:
                logits, embeddings = model(x, edge_index, batch)
            except (TypeError, ValueError):
                print("\n\n[错误] 您的模型 'forward' 方法似乎没有返回两个值(logits, embedding)。")
                print("请确认您已按要求修改了 Detector 类的 forward 方法。")
                return

            labels = global_max_pool(batch_data._VULN, batch).long()
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    output_filename = f"embeddings_for_plotting_{args.dataset_name}_{args.gnn_model}.npz"
    
    np.savez_compressed(
        output_filename,
        embeddings=all_embeddings,
        labels=all_labels
    )
    
    print("\n" + "="*60)
    print(f"数据提取完成！")
    print(f"共提取了 {len(all_labels)} 个样本。")
    print(f"高维嵌入 (维度: {all_embeddings.shape[1]}) 和标签已保存至:")
    print(f"==> {output_filename}")
    print("="*60)

import time
import threading
import psutil
import os

# ==============================================================================
# ==============================================================================
class PerformanceMonitor:
    """
    一个使用 'with' 语句来监控代码块执行时间和峰值内存的上下文管理器。
    结果会同时打印到控制台并追加到指定的日志文件中。
    """
    def __init__(self, name="Task", log_file=None):
        self.name = name
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0
        self.duration_s = 0
        self._stop_thread = threading.Event()
        self._memory_thread = None

    def _poll_memory(self):
        """在后台线程中运行，定期检查内存使用情况。"""
        try:
            self.peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            while not self._stop_thread.is_set():
                current_memory_mb = self.process.memory_info().rss / (1024 * 1024)
                if current_memory_mb > self.peak_memory_mb:
                    self.peak_memory_mb = current_memory_mb
                time.sleep(0.05)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def __enter__(self):
        """当进入 'with' 块时调用。"""
        print(f"\n{'='*20} [开始监控: {self.name}] {'='*20}")
        self.start_time = time.perf_counter()
        self._stop_thread.clear()
        self._memory_thread = threading.Thread(target=self._poll_memory)
        self._memory_thread.daemon = True
        self._memory_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """当离开 'with' 块时调用。"""
        self.end_time = time.perf_counter()
        self.duration_s = self.end_time - self.start_time
        self._stop_thread.set()
        self._memory_thread.join()
        
        result_string = (
            f"{'='*20} [监控结果: {self.name}] {'='*20}\n"
            f"  -> 执行耗时: {self.duration_s:.4f} 秒\n"
            f"  -> CPU 峰值内存占用: {self.peak_memory_mb:.4f} MB\n"
            f"{'='*50}\n\n"
        )

        print(result_string, end='')

        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(result_string)
            except Exception as e:
                print(f"[Warning] 无法写入日志文件 {self.log_file}: {e}")
        
        return False

#cadets 64/256；theia 128/512; trace 64/256;optc trace 64/256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")

    # GNN Model
    parser.add_argument("--model_checkpoint_dir", default="saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--gnn_model", default="GINConv", type=str,
                        help="GNN core.")
    parser.add_argument("--gnn_hidden_size", default=64, type=int,  #236
                        help="hidden size of gnn.")
    parser.add_argument("--gnn_feature_dim_size", default=256, type=int,
                        help="feature dim size of gnn.")
    parser.add_argument("--residual", action='store_true',
                        help="Whether to obtain residual representations.")
    parser.add_argument("--graph_pooling", default="mean", type=str,
                        help="The operator of graph pooling.")
    parser.add_argument("--num_gnn_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_ggnn_steps", default=3, type=int,
                        help="The sequence length for GGNN.")
    parser.add_argument("--ggnn_aggr", default="add", type=str,
                        help="The aggregation scheme to use for GGNN.")
    parser.add_argument("--gin_eps", default=0., type=float,
                        help="Eps value for GIN.")
    parser.add_argument("--gin_train_eps", action='store_true',
                        help="If set to True, eps will be a trainable parameter.")
    parser.add_argument("--gconv_aggr", default="mean", type=str,
                        help="The aggregation scheme to use.")
    parser.add_argument("--dropout_rate", default=0.1, type=float,
                        help="Dropout rate.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    parser.add_argument('--gat_heads', type=int, default=8, 
                        help='Number of multi-head-attentions for GAT.')
    parser.add_argument('--graphsage_aggr', type=str, default='mean',
                        choices=['mean', 'max', 'lstm'],
                        help='The aggregation scheme to use for GraphSAGE.')

#GAT head:cadets 8;theia 4;trace 8

    # Training
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size.")
    parser.add_argument("--learning_rate", default=5e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_explain", action='store_true',
                        help="Whether to run explaining.")

    # Explainer
    parser.add_argument("--ipt_method", default="gnnexplainer", type=str,
                        help="The save path of interpretations.")
    parser.add_argument("--ipt_update", action='store_true',
                        help="Whether to update interpretations.")
    parser.add_argument("--KM", default=8, type=int,
                        help="The size of explanation subgraph.")
    parser.add_argument("--cfexp_L1", action='store_true',
                        help="Whether to use L1 distance item.")
    parser.add_argument("--cfexp_alpha", default=0.9, type=float,
                        help="ProvX alpha.")
    parser.add_argument("--hyper_para", action='store_true',
                        help="Whether to tune the hyper-parameters.")
    parser.add_argument("--case_sample_ids", nargs='+',
                        help="Ids of samples to extract for case study.")

    parser.add_argument('--dataset_name', default='Cadets_com', type=str, 
                        help='The unique name for the dataset (e.g., "Trace_com", "bigvul"). Used for creating saving directories.')
    parser.add_argument('--KN', type=int, default=5, help='Number of top nodes to remove for node-based explanation eval.')

    parser.add_argument('--solidification_factor', type=int, default=0.6,
                        help='Solidification penalty strength.')
    parser.add_argument('--solidification_stage_start_ratio', type=int, default=0.6,
                        help='When to start the solidification stage (ratio of epochs).')
    parser.add_argument("--do_profile", action='store_true',
                        help="Whether to run performance profiling on an explainer.")
    

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    print(device)
    args.device = device
    # args.model_checkpoint_dir = str(utils.cache_dir() / f"{args.model_checkpoint_dir}" / args.gnn_model)
    args.model_checkpoint_dir = str(utils.cache_dir() / args.dataset_name / f"{args.model_checkpoint_dir}" / args.gnn_model)
    print(args.model_checkpoint_dir)
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    model = Detector(args, input_feature_dim=30)
    model.to(args.device)

    log_filepath = f"{args.dataset_name}_performance_log.txt"
    
    # if os.path.exists(log_filepath):
    #    os.remove(log_filepath)
    print(f"Performance log will be saved to: {log_filepath}")


    # ------------------------- Dataset loading -------------------------
    
    # args.data_dir = "./Datasets/Trace_com"
    args.data_dir = args.dataset_name


    train_data = load_partition_data(os.path.join('./Datasets', args.dataset_name), "train_100nodes")
    valid_data = load_partition_data(os.path.join('./Datasets', args.dataset_name), "val_100nodes")
    test_data = load_partition_data(os.path.join('./Datasets', args.dataset_name), "test_100nodes")


    # args.data_dir = "./Datasets/bigvul"

    # train_data = load_partition_data(args.data_dir, "processed_traindata")
    # valid_data = load_partition_data(args.data_dir, "processed_valdata")
    # test_data = load_partition_data(args.data_dir, "processed_testdata")

    # labels = [data.y[2].item() for data in all_data]
    # balanced_data, balanced_labels = balance_classes(all_data, labels)
    # train_data, temp_data, train_labels, temp_labels = train_test_split(...)
    # valid_data, test_data, valid_labels, test_labels = train_test_split(...)

    print("训练集大小:", len(train_data))
    print("验证集大小:", len(valid_data))
    print("测试集大小:", len(test_data))



    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    print("测试集Loader大小:", len(test_dataloader))

    def get_label_distribution(data):
        """统计数据集标签分布"""
        labels = [torch.any(data_point._VULN.bool()).int().item() for data_point in data]
        
        return dict(Counter(labels))

    print(f"训练集类别分布: {get_label_distribution(train_data)}")
    print(f"验证集类别分布: {get_label_distribution(valid_data)}")
    print(f"测试集类别分布: {get_label_distribution(test_data)}")

    train_sampler = None
    if args.do_train:
        print("正在为训练集中的【图】计算采样权重...")
        graph_actual_labels_for_sampler = []
        for graph_data_item in train_data:
            if hasattr(graph_data_item, '_VULN') and graph_data_item._VULN is not None and len(graph_data_item._VULN) > 0:
                is_malicious_graph = 1 if (graph_data_item._VULN == 1).any() else 0
                graph_actual_labels_for_sampler.append(is_malicious_graph)
            else:
                graph_actual_labels_for_sampler.append(0)

        if graph_actual_labels_for_sampler:
            graph_label_counts = Counter(graph_actual_labels_for_sampler)
            count_g0 = graph_label_counts.get(0,0); count_g1 = graph_label_counts.get(1,0)
            print(f"训练集【图】类别统计: 良性(0)={count_g0}, 恶性(1)={count_g1}")
            if count_g1 > 0 and count_g0 > 0:
                weight_g0 = 1.0 / count_g0; weight_g1 = 1.0 / count_g1
                sample_weights = [weight_g1 if lbl == 1 else weight_g0 for lbl in graph_actual_labels_for_sampler]
                from torch.utils.data import WeightedRandomSampler
                train_sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), 
                                                      int(len(sample_weights) * 3), replacement=True)
                print("已创建 WeightedRandomSampler 用于图级别过采样。")
            else:
                print("训练集中某一类别的图数量为0，不进行过采样。")
        else:
            print("未能计算图的采样权重。")

    if train_sampler and args.do_train:
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    else:
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.do_train and train_sampler is not None:
        print("\n--- 检查 DataLoader 输出批次的图类别分布 (前3个批次) ---")
        temp_batches_checked = 0
        try:
            for i_check, temp_batch_data in enumerate(train_dataloader):
                if temp_batches_checked >= 3:
                    break
                
                temp_batch_data = temp_batch_data.to(args.device)
                graph_labels_in_batch = global_max_pool(temp_batch_data._VULN.float(), temp_batch_data.batch).long()
                counts = Counter(graph_labels_in_batch.cpu().tolist())
                num_graphs_in_this_batch = len(graph_labels_in_batch)

                print(f"  DataLoader 输出批次 {i_check+1} (含 {num_graphs_in_this_batch} 个图):")
                print(f"    良性图(0) = {counts.get(0,0)}, 恶性图(1) = {counts.get(1,0)}")
                if num_graphs_in_this_batch > 0:
                    print(f"    此批次中恶性图比例: {counts.get(1,0)/num_graphs_in_this_batch:.2%}")
                temp_batches_checked += 1
        except Exception as e:
            print(f"  检查 DataLoader 输出时发生错误: {e}")
        print("--- DataLoader 输出批次检查结束 ---\n")
    elif args.do_train:
        print("\n提示: 未使用 WeightedRandomSampler，DataLoader 将按原始数据分布随机打乱。\n")


    if args.do_train:
        with PerformanceMonitor("Model Training", log_file=log_filepath):
            train(args, train_dataloader, valid_dataloader, test_dataloader, model)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        model_checkpoint_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
        model.to(args.device)
        test_result = evaluate(args, test_dataloader, model)

        print("\n" + "="*50)
        print("准备提取并保存嵌入数据...")
        save_embeddings_for_plotting(args, model, test_dataloader)
        print("="*50 + "\n")

        print("***** Test results *****")
        for key in sorted(test_result.keys()):
            print("  {} = {}".format(key, str(round(test_result[key], 4))))

        if args.do_explain:
            correct_lines = get_dep_add_lines_bigvul()

            # print("correct lines:", correct_lines)
            # correct_lines = load_existing_labels()

            # ipt_save_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}")
            ipt_save_dir = str(utils.cache_dir() / args.dataset_name / f"explainer_cache" / f"{args.gnn_model}")
            if not os.path.exists(ipt_save_dir):
                os.makedirs(ipt_save_dir)
            if args.hyper_para:
                if args.ipt_method == "provx":
                    if args.cfexp_L1:
                        ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}_L1_{args.cfexp_alpha}.pt")
                    else:
                        ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}_{args.cfexp_alpha}.pt")
            else:
                ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}.pt")
            print("Size of test dataset:", len(test_data))

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

            if not os.path.exists(ipt_save) or args.ipt_update:
                graph_exp_list = []
                if args.ipt_method == "pgexplainer":
                    eval_model = Detector(args, input_feature_dim=17)
                    eval_model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
                    eval_model.to(args.device)
                    graph_exp_list = pgexplainer_run(args, model, eval_model, train_dataloader, test_dataloader,
                                                     correct_lines)
                elif args.ipt_method == "subgraphx":
                    graph_exp_list = subgraphx_run(args, model, test_dataloader, correct_lines)
                elif args.ipt_method == "gnn_lrp":
                    graph_exp_list = gnn_lrp_run(args, model, test_dataloader, correct_lines)
                elif args.ipt_method == "deeplift":
                    graph_exp_list = deeplift_run(args, model, test_dataloader, correct_lines)
                elif args.ipt_method == "gradcam":
                    graph_exp_list = gradcam_run(args, model, test_dataloader, correct_lines)
                elif args.ipt_method == "gnnexplainer":
                    graph_exp_list = gnnexplainer_run(args, model, test_dataloader, correct_lines)
                elif args.ipt_method == "provx":
                    with PerformanceMonitor("ProvX Run", log_file=log_filepath):
                        graph_exp_list = provx_run(args, model, test_dataloader, correct_lines)

                torch.save(graph_exp_list, ipt_save)

            with PerformanceMonitor("Explanation Evaluation (Line-based)", log_file=log_filepath):
                eval_exp(ipt_save, model, correct_lines, args)   

            # args.KN = 15
            # with PerformanceMonitor("Explanation Evaluation (Node-based)", log_file=log_filepath):
                # eval_exp_node(ipt_save, model, args)


if __name__ == "__main__":
    main()

