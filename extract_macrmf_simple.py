#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import tensorflow.compat.v1 as tf
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

def get_data_stats(data_path, dataset):
    """获取数据集的统计信息（用户数和物品数）"""
    print(f"从数据集 {dataset} 获取用户和物品数量")
    
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
    
    print(f"统计得到：n_users={n_users}, n_items={n_items}")
    return n_users, n_items

def extract_macrmf_recommendations(model_path, test_users, n_users, n_items, c_value=40.0, topk=20):
    """直接从检查点文件中提取嵌入向量并计算MACR-MF推荐"""
    print(f"从路径 {model_path} 直接提取MACR-MF推荐结果，c值={c_value}...")
    
    # 从检查点读取
    reader = tf.train.NewCheckpointReader(model_path)
    
    # 获取所有变量名
    var_to_shape_map = reader.get_variable_to_shape_map()
    print(f"检查点中的所有变量：{list(var_to_shape_map.keys())}")
    
    # 尝试提取嵌入变量
    try:
        # 尝试各种可能的嵌入变量名称
        user_embedding_names = [
            "parameter/user_embedding", 
            "user_embedding",
            "weights/user_embedding"
        ]
        
        item_embedding_names = [
            "parameter/item_embedding",
            "item_embedding",
            "weights/item_embedding"
        ]
        
        user_embeddings = None
        item_embeddings = None
        
        # 尝试加载用户嵌入
        for var_name in user_embedding_names:
            if var_name in var_to_shape_map:
                user_embeddings = reader.get_tensor(var_name)
                print(f"已加载用户嵌入 {var_name}: shape={user_embeddings.shape}")
                break
        
        # 尝试加载物品嵌入
        for var_name in item_embedding_names:
            if var_name in var_to_shape_map:
                item_embeddings = reader.get_tensor(var_name)
                print(f"已加载物品嵌入 {var_name}: shape={item_embeddings.shape}")
                break
        
        # 如果仍然没有找到嵌入，尝试从变量形状判断
        if user_embeddings is None or item_embeddings is None:
            emb_vars = []
            for var_name, shape in var_to_shape_map.items():
                if len(shape) == 2 and shape[1] > 10:  # 嵌入通常是二维的，且第二维度大于10
                    emb_vars.append((var_name, shape))
            
            if len(emb_vars) >= 2:
                # 按第一维从大到小排序，通常用户数大于物品数
                emb_vars.sort(key=lambda x: x[1][0], reverse=True)
                
                if user_embeddings is None and emb_vars[0][1][0] >= n_users:
                    var_name, _ = emb_vars[0]
                    user_embeddings = reader.get_tensor(var_name)
                    print(f"根据形状加载用户嵌入 {var_name}: shape={user_embeddings.shape}")
                
                if item_embeddings is None and len(emb_vars) > 1 and emb_vars[1][1][0] >= n_items:
                    var_name, _ = emb_vars[1]
                    item_embeddings = reader.get_tensor(var_name)
                    print(f"根据形状加载物品嵌入 {var_name}: shape={item_embeddings.shape}")
        
        # 尝试提取MACR相关的变量
        rubi_c = None
        if "const_embedding/rubi_c" in var_to_shape_map:
            rubi_c = reader.get_tensor("const_embedding/rubi_c")[0]
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
            "item_branch", "w", 
            "user_branch", "w_user"
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
    parser = argparse.ArgumentParser(description='从MACR-MF检查点中直接提取推荐结果')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型检查点路径，例如 mf_addressa_checkpoint/wd_1e-05_lr_0.001_0/189_ckpt.ckpt')
    parser.add_argument('--data_path', type=str, default='data', 
                        help='数据集根目录路径')
    parser.add_argument('--dataset', type=str, default='addressa',
                        help='数据集名称')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出文件路径，如 results/macrmf_recommendations.json')
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
    
    # 获取数据集统计信息
    n_users, n_items = get_data_stats(args.data_path, args.dataset)
    
    # 提取推荐结果
    start_time = time.time()
    recommendations = extract_macrmf_recommendations(
        args.model_path, test_users, n_users, n_items, args.c_value, args.topk
    )
    elapsed = time.time() - start_time
    
    # 保存推荐结果
    if recommendations:
        # 将用户ID转为字符串，以便JSON序列化
        json_recommendations = {str(u): r for u, r in recommendations.items()}
        with open(args.output_file, 'w') as f:
            json.dump(json_recommendations, f)
        print(f"MACR-MF模型的推荐结果已保存到 {args.output_file}，耗时: {elapsed:.2f}秒")
    else:
        print("MACR-MF模型的推荐结果提取失败")

if __name__ == "__main__":
    main() 