#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import tensorflow.compat.v1 as tf
import scipy.sparse as sp
import time
import argparse

def load_test_users(test_file):
    """加载测试集中的用户"""
    test_users = set()
    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                user = int(parts[0])
                test_users.add(user)
    return list(test_users)

def load_adj_mat(data_path, dataset):
    """加载邻接矩阵"""
    path = f"{data_path}/{dataset}/adj_mat.npz"
    
    try:
        # 尝试直接加载预计算的邻接矩阵
        adj_mat = sp.load_npz(path)
        print(f"从 {path} 加载邻接矩阵")
        return adj_mat
    except:
        print(f"无法从 {path} 加载邻接矩阵，将创建新的邻接矩阵")
        # 从训练数据创建邻接矩阵
        train_file = f"{data_path}/{dataset}/train.txt"
        
        n_users = 0
        n_items = 0
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) > 1:
                    u = int(parts[0])
                    n_users = max(n_users, u + 1)
                    for i in parts[1:]:
                        i = int(i)
                        n_items = max(n_items, i + 1)
        
        print(f"从训练数据中计算得到: n_users={n_users}, n_items={n_items}")
        
        # 创建R矩阵
        R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) > 1:
                    u = int(parts[0])
                    for i in parts[1:]:
                        R[u, int(i)] = 1.0
        
        # 计算邻接矩阵 [[0, R], [R^T, 0]]
        adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()
        adj_mat[:n_users, n_users:] = R
        adj_mat[n_users:, :n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # 计算拉普拉斯归一化版本 (D^-0.5 * A * D^-0.5)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv).tocsr()
        
        print(f"已创建归一化的邻接矩阵: shape={norm_adj.shape}")
        
        # 保存邻接矩阵以备后用
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sp.save_npz(path, norm_adj)
        
        return norm_adj

def extract_macrlightgcn_recommendations(model_path, test_users, dataset, data_path, c_value=40.0, topk=20):
    """直接从检查点文件中提取嵌入向量并计算MACR推荐"""
    print(f"从路径 {model_path} 直接提取MACR-LightGCN推荐结果，c值={c_value}...")
    
    # 从检查点读取
    reader = tf.train.NewCheckpointReader(model_path)
    
    # 获取所有变量名
    var_to_shape_map = reader.get_variable_to_shape_map()
    print(f"检查点中的所有变量：{list(var_to_shape_map.keys())}")
    
    # 尝试提取嵌入变量
    try:
        # 尝试各种可能的嵌入变量名称
        embedding_var_names = [
            "weights/embedding", 
            "embedding",
            "weights",
            "embedding_P",
            "weights/embedding/embedding_P", 
            "embedding_Q", 
            "weights/embedding/embedding_Q"
        ]
        
        user_embeddings = None
        item_embeddings = None
        
        for var_name in embedding_var_names:
            if var_name in var_to_shape_map:
                embeddings = reader.get_tensor(var_name)
                shape = embeddings.shape
                print(f"发现嵌入变量 {var_name}: shape={shape}")
                
                # 判断该变量是用户嵌入还是物品嵌入
                if user_embeddings is None and "embedding_P" in var_name:
                    user_embeddings = embeddings
                    print(f"已加载用户嵌入: shape={user_embeddings.shape}")
                elif item_embeddings is None and "embedding_Q" in var_name:
                    item_embeddings = embeddings
                    print(f"已加载物品嵌入: shape={item_embeddings.shape}")
        
        # 如果没有找到明确标记的用户和物品嵌入，尝试根据变量形状判断
        if user_embeddings is None or item_embeddings is None:
            # 找具有相同最后一个维度的变量，这通常是嵌入维度
            embed_vars = []
            for var_name, shape in var_to_shape_map.items():
                if len(shape) == 2:  # 只考虑二维变量
                    embed_vars.append((var_name, shape))
            
            if len(embed_vars) >= 2:
                # 如果有至少两个不同大小的嵌入矩阵
                # 按第一维从大到小排序，通常用户数量大于物品数量
                embed_vars.sort(key=lambda x: x[1][0], reverse=True)
                
                if user_embeddings is None:
                    user_var_name, _ = embed_vars[0]
                    user_embeddings = reader.get_tensor(user_var_name)
                    print(f"已根据形状判断加载用户嵌入 {user_var_name}: shape={user_embeddings.shape}")
                
                if item_embeddings is None and len(embed_vars) > 1:
                    item_var_name, _ = embed_vars[1]
                    item_embeddings = reader.get_tensor(item_var_name)
                    print(f"已根据形状判断加载物品嵌入 {item_var_name}: shape={item_embeddings.shape}")
        
        # 如果仍然没有找到，则尝试加载总的嵌入并拆分
        if user_embeddings is None or item_embeddings is None:
            if "weights/embedding" in var_to_shape_map:
                all_embeddings = reader.get_tensor("weights/embedding")
                # 尝试判断用户和物品的数量
                # 加载数据集信息
                adj_mat = load_adj_mat(data_path, dataset)
                n_users = adj_mat.shape[0] - adj_mat.shape[1]
                n_items = adj_mat.shape[1]
                
                if user_embeddings is None:
                    user_embeddings = all_embeddings[:n_users]
                    print(f"从全局嵌入拆分出用户嵌入: shape={user_embeddings.shape}")
                
                if item_embeddings is None:
                    item_embeddings = all_embeddings[n_users:]
                    print(f"从全局嵌入拆分出物品嵌入: shape={item_embeddings.shape}")
        
        # 尝试提取MACR相关的变量
        rubi_c = None
        if "rubi_c" in var_to_shape_map:
            rubi_c = reader.get_tensor("rubi_c")[0]
            print(f"已从检查点加载c值: {rubi_c}")
        else:
            # 使用传入的c值
            rubi_c = c_value
            print(f"使用指定的c值: {c_value}")
        
        # 尝试加载分支权重
        w_item = None
        w_user = None
        
        # 尝试各种可能的分支权重变量名称
        branch_var_names = [
            "item_branch", "weights/item_branch", "w", "weights/w",
            "user_branch", "weights/user_branch", "w_user", "weights/w_user"
        ]
        
        for var_name in branch_var_names:
            if var_name in var_to_shape_map:
                weight = reader.get_tensor(var_name)
                if any(item_name in var_name for item_name in ["item_branch", "w"]) and w_item is None:
                    w_item = weight
                    print(f"已加载物品分支权重 {var_name}: shape={w_item.shape}")
                elif any(user_name in var_name for user_name in ["user_branch", "w_user"]) and w_user is None:
                    w_user = weight
                    print(f"已加载用户分支权重 {var_name}: shape={w_user.shape}")
    
    except Exception as e:
        print(f"提取嵌入向量失败: {e}")
        return {}
    
    # 确保我们有用户和物品嵌入
    if user_embeddings is None or item_embeddings is None:
        print("无法找到用户或物品嵌入向量，提取推荐失败")
        return {}
    
    # 创建默认的分支权重，如果未能从检查点中加载
    embed_dim = user_embeddings.shape[1]
    if w_item is None:
        w_item = np.ones((embed_dim, 1), dtype=np.float32)
        print(f"创建默认物品分支权重: shape={w_item.shape}")
    
    if w_user is None:
        w_user = np.ones((embed_dim, 1), dtype=np.float32)
        print(f"创建默认用户分支权重: shape={w_user.shape}")
    
    # 根据嵌入向量计算推荐
    recommendations = {}
    batch_size = 128
    n_batches = (len(test_users) + batch_size - 1) // batch_size
    
    # 使用批处理方式计算评分
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(test_users))
        user_batch = test_users[start:end]
        
        # 获取这批用户的嵌入
        user_embed_batch = user_embeddings[user_batch]
        
        # 计算用户分支得分
        user_branch_scores = np.matmul(user_embed_batch, w_user)
        user_branch_probs = 1.0 / (1.0 + np.exp(-user_branch_scores))  # sigmoid
        
        # 计算与所有物品的基本评分 (batch_size, embed_dim) x (embed_dim, n_items)
        base_scores = np.matmul(user_embed_batch, item_embeddings.T)
        
        # 计算物品分支得分 (为每个物品计算sigmoid分数)
        item_branch_scores = np.matmul(item_embeddings, w_item)
        item_branch_probs = 1.0 / (1.0 + np.exp(-item_branch_scores))  # sigmoid
        
        # 应用MACR去偏置：减去c值和应用两个分支的sigmoid概率
        # MACR公式: (base_score - c) * sigmoid(user_branch) * sigmoid(item_branch)
        # 对每个用户-物品对应用MACR公式
        for idx, user in enumerate(user_batch):
            user_scores = base_scores[idx]
            
            # 对于每个物品应用MACR公式
            # 从基础分数中减去c值
            unbiased_scores = user_scores - rubi_c
            
            # 应用用户和物品分支的sigmoid概率
            # 注意：这里我们要乘以物品分支概率的转置，以匹配维度
            final_scores = unbiased_scores * user_branch_probs[idx, 0] * item_branch_probs.flatten()
            
            # 获取评分最高的K个物品
            top_indices = np.argsort(-final_scores)[:topk].tolist()
            recommendations[user] = top_indices
        
        print(f"批次 {batch_idx+1}/{n_batches} 已完成")
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='从MACR-LightGCN检查点中直接提取推荐结果')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型检查点路径，例如 weights/addressa/LightGCN/64-64/l0.0001_r1e-05-1e-05-0.01/weights_-20')
    parser.add_argument('--data_path', type=str, default='data', 
                        help='数据集根目录路径')
    parser.add_argument('--dataset', type=str, default='addressa',
                        help='数据集名称')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出文件路径，如 results/macrlightgcn_recommendations.json')
    parser.add_argument('--c_value', type=float, default=40.0,
                        help='MACR的c值，用于去偏置')
    parser.add_argument('--topk', type=int, default=20, 
                        help='推荐列表长度')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载测试用户
    test_file = os.path.join(args.data_path, args.dataset, 'test.txt')
    test_users = load_test_users(test_file)
    print(f"共加载了{len(test_users)}个测试用户")
    
    # 提取推荐结果
    start_time = time.time()
    recommendations = extract_macrlightgcn_recommendations(
        args.model_path, test_users, args.dataset, args.data_path, args.c_value, args.topk
    )
    elapsed = time.time() - start_time
    
    # 保存推荐结果
    if recommendations:
        # 将用户ID转为字符串，以便JSON序列化
        json_recommendations = {str(u): r for u, r in recommendations.items()}
        with open(args.output_file, 'w') as f:
            json.dump(json_recommendations, f)
        print(f"MACR-LightGCN模型的推荐结果已保存到 {args.output_file}，耗时: {elapsed:.2f}秒")
    else:
        print("MACR-LightGCN模型的推荐结果提取失败")

if __name__ == "__main__":
    main() 