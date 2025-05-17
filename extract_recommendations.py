#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
from collections import defaultdict
import time

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

def extract_mf_recommendations(model_path, test_users, topk=20):
    """从MF模型中提取推荐结果"""
    print(f"从路径 {model_path} 提取MF模型推荐结果...")
    
    # 导入必要的模块
    sys.path.append('./macr_mf')
    
    # 先导入这些，但在创建 args 对象时不使用 parse_args
    from macr_mf.model import BPRMF
    import macr_mf.parse as mf_parse
    
    # 手动创建 args 对象，不调用 parse_args
    args = mf_parse.parse_args.__defaults__[0]  # 获取默认参数对象
    # 或者创建一个简单对象并设置需要的属性
    class Args:
        pass
    args = Args()
    args.dataset = os.path.basename(os.path.dirname(model_path))
    
    # 加载模型配置
    config = {}
    # 从模型路径推断出必要的配置
    # 这里需要根据您的实际情况设置config
    # 例如：
    from macr_mf.batch_test import Data
    data = Data(args.dataset)
    config['n_users'] = data.n_users
    config['n_items'] = data.n_items
    
    # 创建模型
    with tf.Graph().as_default():
        model = BPRMF(args, config)
        
        # 创建会话
        sess = tf.compat.v1.Session()
        
        # 加载模型参数
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(sess, model_path)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            return {}
        
        # 提取推荐结果
        recommendations = {}
        batch_size = 128
        n_batches = (len(test_users) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(test_users))
            user_batch = test_users[start:end]
            
            # 对所有物品进行评分
            items = list(range(config['n_items']))
            ratings = sess.run(model.batch_ratings, 
                              {model.users: user_batch, model.pos_items: items})
            
            # 为每个用户获取Top-K推荐
            for idx, user in enumerate(user_batch):
                user_ratings = ratings[idx]
                # 获取评分最高的K个物品
                top_indices = np.argsort(-user_ratings)[:topk].tolist()
                recommendations[user] = top_indices
            
            print(f"批次 {batch_idx+1}/{n_batches} 已完成")
        
        sess.close()
    
    return recommendations

def extract_macrmf_recommendations(model_path, test_users, c_value, topk=20):
    """从MACR-MF模型中提取推荐结果"""
    print(f"从路径 {model_path} 提取MACR-MF模型推荐结果，c值={c_value}...")
    
    # 导入必要的模块
    sys.path.append('./macr_mf')
    from macr_mf.model import BPRMF
    from macr_mf.parse import parse_args
    
    # 设置参数
    args = parse_args()
    args.dataset = os.path.basename(os.path.dirname(model_path))
    
    # 加载模型配置
    config = {}
    # 从模型路径推断出必要的配置
    from macr_mf.batch_test import Data
    data = Data(args.dataset)
    config['n_users'] = data.n_users
    config['n_items'] = data.n_items
    
    # 创建模型
    with tf.Graph().as_default():
        model = BPRMF(args, config)
        
        # 创建会话
        sess = tf.compat.v1.Session()
        
        # 加载模型参数
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(sess, model_path)
            print(f"成功加载模型: {model_path}")
            
            # 设置MACR对应的c值
            model.update_c(sess, c_value)
            print(f"已设置c值: {c_value}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return {}
        
        # 提取推荐结果
        recommendations = {}
        batch_size = 128
        n_batches = (len(test_users) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(test_users))
            user_batch = test_users[start:end]
            
            # 对所有物品进行评分
            items = list(range(config['n_items']))
            
            # 使用MACR的评分函数（例如rubi_ratings）
            # 根据您的模型实际情况选择正确的评分函数
            ratings = sess.run(model.rubi_ratings, 
                              {model.users: user_batch, model.pos_items: items})
            
            # 为每个用户获取Top-K推荐
            for idx, user in enumerate(user_batch):
                user_ratings = ratings[idx]
                # 获取评分最高的K个物品
                top_indices = np.argsort(-user_ratings)[:topk].tolist()
                recommendations[user] = top_indices
            
            print(f"批次 {batch_idx+1}/{n_batches} 已完成")
        
        sess.close()
    
    return recommendations

def extract_lightgcn_recommendations(model_path, test_users, topk=20):
    """从LightGCN模型中提取推荐结果"""
    print(f"从路径 {model_path} 提取LightGCN模型推荐结果...")
    
    # 导入必要的模块
    sys.path.append('./macr_lightgcn')
    try:
        from macr_lightgcn.LightGCN import LightGCN
        from macr_lightgcn.utility.parser import parse_args
        from macr_lightgcn.utility.load_data import Data
    except ImportError as e:
        print(f"导入LightGCN模块失败: {e}")
        return {}
    
    # 设置参数
    args = parse_args()
    args.dataset = os.path.basename(os.path.dirname(model_path))
    
    # 加载数据配置
    data_generator = Data(path=f"data/{args.dataset}", batch_size=args.batch_size, args=args)
    config = {}
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    
    # 获取邻接矩阵
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
    elif args.adj_type == 'pre':
        config['norm_adj'] = pre_adj
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
    
    # 创建模型
    with tf.Graph().as_default():
        model = LightGCN(data_config=config, pretrain_data=None)
        
        # 创建会话
        sess = tf.compat.v1.Session()
        
        # 加载模型参数
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(sess, model_path)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            return {}
        
        # 提取推荐结果
        recommendations = {}
        batch_size = 128
        n_batches = (len(test_users) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(test_users))
            user_batch = test_users[start:end]
            
            # 对所有物品进行评分
            items = list(range(config['n_items']))
            ratings = sess.run(model.batch_ratings, 
                             {model.users: user_batch, model.pos_items: items,
                              model.node_dropout: [0.] * len(eval(args.layer_size)),
                              model.mess_dropout: [0.] * len(eval(args.layer_size))})
            
            # 为每个用户获取Top-K推荐
            for idx, user in enumerate(user_batch):
                user_ratings = ratings[idx]
                # 获取评分最高的K个物品
                top_indices = np.argsort(-user_ratings)[:topk].tolist()
                recommendations[user] = top_indices
            
            print(f"批次 {batch_idx+1}/{n_batches} 已完成")
        
        sess.close()
    
    return recommendations

def extract_macrlightgcn_recommendations(model_path, test_users, c_value, topk=20):
    """从MACR-LightGCN模型中提取推荐结果"""
    print(f"从路径 {model_path} 提取MACR-LightGCN模型推荐结果，c值={c_value}...")
    
    # 导入必要的模块
    sys.path.append('./macr_lightgcn')
    try:
        from macr_lightgcn.LightGCN import LightGCN
        from macr_lightgcn.utility.parser import parse_args
        from macr_lightgcn.utility.load_data import Data
    except ImportError as e:
        print(f"导入LightGCN模块失败: {e}")
        return {}
    
    # 设置参数
    args = parse_args()
    args.dataset = os.path.basename(os.path.dirname(model_path))
    
    # 加载数据配置
    data_generator = Data(path=f"data/{args.dataset}", batch_size=args.batch_size, args=args)
    config = {}
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    
    # 获取邻接矩阵
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
    elif args.adj_type == 'pre':
        config['norm_adj'] = pre_adj
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
    
    # 创建模型
    with tf.Graph().as_default():
        model = LightGCN(data_config=config, pretrain_data=None)
        
        # 创建会话
        sess = tf.compat.v1.Session()
        
        # 加载模型参数
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(sess, model_path)
            print(f"成功加载模型: {model_path}")
            
            # 设置MACR对应的c值
            model.update_c(sess, c_value)
            print(f"已设置c值: {c_value}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return {}
        
        # 提取推荐结果
        recommendations = {}
        batch_size = 128
        n_batches = (len(test_users) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(test_users))
            user_batch = test_users[start:end]
            
            # 对所有物品进行评分
            items = list(range(config['n_items']))
            
            # 使用MACR的评分函数（例如rubi_ratings_both）
            # 根据您的模型实际情况选择正确的评分函数
            ratings = sess.run(model.rubi_ratings_both, 
                             {model.users: user_batch, model.pos_items: items,
                              model.node_dropout: [0.] * len(eval(args.layer_size)),
                              model.mess_dropout: [0.] * len(eval(args.layer_size))})
            
            # 为每个用户获取Top-K推荐
            for idx, user in enumerate(user_batch):
                user_ratings = ratings[idx]
                # 获取评分最高的K个物品
                top_indices = np.argsort(-user_ratings)[:topk].tolist()
                recommendations[user] = top_indices
            
            print(f"批次 {batch_idx+1}/{n_batches} 已完成")
        
        sess.close()
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='从模型中提取Top-K推荐结果')
    parser.add_argument('--data_path', type=str, default='data/addressa', help='数据集路径')
    parser.add_argument('--results_path', type=str, default='results', help='结果保存路径')
    parser.add_argument('--model_paths', type=str, required=True, help='模型路径，格式：model_name:path,model_name:path')
    parser.add_argument('--c_values', type=str, default='macrmf:45,macrlightgcn:45', help='MACR模型的c值，格式：model_name:c_value')
    parser.add_argument('--topk', type=int, default=20, help='推荐列表长度')
    args = parser.parse_args()
    
    # 解析模型路径
    model_paths = {}
    for item in args.model_paths.split(','):
        name, path = item.split(':')
        model_paths[name] = path
    
    # 解析c值
    c_values = {}
    for item in args.c_values.split(','):
        name, value = item.split(':')
        c_values[name] = float(value)
    
    # 创建结果目录
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    # 加载测试用户
    test_file = os.path.join(args.data_path, 'test.txt')
    test_users = load_test_users(test_file)
    print(f"共加载了{len(test_users)}个测试用户")
    
    # 提取推荐结果
    for model_name, model_path in model_paths.items():
        start_time = time.time()
        
        if model_name == 'mf':
            recommendations = extract_mf_recommendations(model_path, test_users, args.topk)
        elif model_name == 'macrmf':
            c_value = c_values.get('macrmf', 45)
            recommendations = extract_macrmf_recommendations(model_path, test_users, c_value, args.topk)
        elif model_name == 'lightgcn':
            recommendations = extract_lightgcn_recommendations(model_path, test_users, args.topk)
        elif model_name == 'macrlightgcn':
            c_value = c_values.get('macrlightgcn', 45)
            recommendations = extract_macrlightgcn_recommendations(model_path, test_users, c_value, args.topk)
        else:
            print(f"不支持的模型类型: {model_name}")
            continue
        
        elapsed = time.time() - start_time
        
        # 保存推荐结果
        if recommendations:
            # 将用户ID转为字符串，以便JSON序列化
            json_recommendations = {str(u): r for u, r in recommendations.items()}
            output_file = os.path.join(args.results_path, f"{model_name}_recommendations.json")
            with open(output_file, 'w') as f:
                json.dump(json_recommendations, f)
            print(f"{model_name}模型的推荐结果已保存到 {output_file}，耗时: {elapsed:.2f}秒")
        else:
            print(f"{model_name}模型的推荐结果提取失败")

if __name__ == "__main__":
    main() 