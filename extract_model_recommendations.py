#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 设置环境变量，阻止macr_mf和macr_lightgcn中的参数解析器自动执行
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
# 备份sys.argv，防止被其他模块的参数解析污染
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  # 只保留脚本名称

import argparse
from collections import defaultdict
import time
import scipy.sparse as sp

# 在导入其他模块后恢复原来的参数
def restore_args():
    sys.argv = original_argv.copy()

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
    
    # 保存当前argv
    tmp_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    
    # 修复导入方式
    try:
        from macr_mf.model import BPRMF
        from macr_mf.load_data import Data
        # 不再直接从parse导入parse_args
    except ImportError as e:
        print(f"导入MF模块失败: {e}")
        return {}
    
    # 恢复argv，以防导入过程中被修改
    sys.argv = tmp_argv
    
    # 创建参数对象
    class Args:
        def __init__(self):
            # 正确设置数据集名称为addressa而不是从模型路径中提取
            self.dataset = "addressa"
            self.data_path = './data/'  # 添加data_path属性
            self.regs = 1e-5
            self.embed_size = 64
            self.batch_size = 1024
            self.layer_size = '[64]'
            self.verbose = 1
            self.epoch = 1000
            self.pretrain = 0
            self.valid_set = "test"  # 使用test数据集进行评估
            self.model = "mf"  # 模型类型
            self.source = "normal"  # 数据源类型
            self.data_type = "ori"  # 数据类型，使用原始数据
            self.lr = 0.001  # 学习率
            self.c = 0.0    # 默认c值
            self.alpha = 0.5 # 默认alpha值
            self.beta = 0.5  # 默认beta值
            
    args = Args()
    
    # 加载模型配置
    config = {}
    data = Data(args)  # 创建Data对象
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
    
    # 保存当前argv
    tmp_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    
    # 修复导入方式
    try:
        from macr_mf.model import BPRMF
        from macr_mf.load_data import Data
        # 不再直接从parse导入parse_args
    except ImportError as e:
        print(f"导入MACR-MF模块失败: {e}")
        return {}
    
    # 恢复argv，以防导入过程中被修改
    sys.argv = tmp_argv
    
    # 创建参数对象
    class Args:
        def __init__(self):
            # 正确设置数据集名称为addressa而不是从模型路径中提取
            self.dataset = "addressa"
            self.data_path = './data/'  # 添加data_path属性
            self.regs = 1e-5
            self.embed_size = 64
            self.batch_size = 1024
            self.layer_size = '[64]'
            self.verbose = 1
            self.epoch = 1000
            self.pretrain = 0
            self.valid_set = "test"  # 使用test数据集进行评估
            self.model = "mf"  # 模型类型
            self.source = "normal"  # 数据源类型
            self.data_type = "ori"  # 数据类型，使用原始数据
            self.lr = 0.001  # 学习率
            self.c = 0.0    # 默认c值
            self.alpha = 0.5 # 默认alpha值
            self.beta = 0.5  # 默认beta值
            
    args = Args()
    
    # 加载模型配置
    config = {}
    data = Data(args)  # 创建Data对象
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
            
            # 使用MACR的评分函数（例如rubi_ratings_both）
            # 根据您的模型实际情况选择正确的评分函数
            ratings = sess.run(model.rubi_ratings_both, 
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
    
    # 保存当前argv
    tmp_argv = sys.argv.copy()
    # 创建一个包含正确数据集路径的参数列表
    sys.argv = [sys.argv[0], '--dataset', 'addressa', '--data_path', './data/']
    
    try:
        # 直接导入需要的类，避免执行batch_test.py中的代码
        from macr_lightgcn.utility.load_data import Data
        # 然后再导入LightGCN模型
        from macr_lightgcn.LightGCN import LightGCN
        
        # 恢复argv，以防导入过程中被修改
        sys.argv = tmp_argv
        
        # 创建参数对象
        class Args:
            def __init__(self):
                # 正确设置数据集名称为addressa而不是从模型路径中提取
                self.dataset = "addressa"
                self.adj_type = 'pre'
                self.alg_type = 'lightgcn'
                self.layer_size = '[64,64]'
                self.embed_size = 64
                self.batch_size = 1024
                self.node_dropout = '[0.1]'
                self.mess_dropout = '[0.1]'
                self.regs = '[1e-5,1e-5,1e-2]'
                self.lr = 0.001
                self.model_type = 'LightGCN'
                self.valid_set = "test"  # 使用test数据集进行评估
                self.test_flag = "part"  # 测试类型
                self.save_flag = 0  # 是否保存模型
                self.early_stop = 1  # 是否早停
                self.pretrain = 0  # 是否预训练
                self.verbose = 1  # 输出详细程度
                
        args = Args()
        
    except ImportError as e:
        print(f"导入LightGCN模块失败: {e}")
        sys.argv = tmp_argv  # 确保在异常情况下也恢复argv
        return {}
    
    # 加载数据配置
    try:
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
    except Exception as e:
        print(f"加载数据配置失败: {e}")
        return {}
    
    # 创建模型
    with tf.Graph().as_default():
        model = LightGCN(data_config=config, pretrain_data=None)
        
        # 创建会话
        sess = tf.compat.v1.Session()
        
        # 初始化变量
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # 创建一个只包含现有变量的saver
        var_list = {}
        reader = tf.compat.v1.train.NewCheckpointReader(model_path)
        available_vars = reader.get_variable_to_shape_map()
        
        for var in tf.compat.v1.global_variables():
            # 获取变量名称（去掉前缀":")
            var_name = var.name.split(':')[0]
            if var_name in available_vars:
                var_list[var_name] = var
        
        saver = tf.compat.v1.train.Saver(var_list=var_list)
        
        try:
            saver.restore(sess, model_path)
            print(f"成功加载模型(部分变量): {model_path}")
            print(f"加载了 {len(var_list)} 个变量，模型总变量数 {len(tf.compat.v1.global_variables())}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            sess.close()
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
            try:
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
            except Exception as e:
                print(f"批次 {batch_idx+1}/{n_batches} 评分失败: {e}")
                continue
        
        sess.close()
    
    return recommendations

def extract_macrlightgcn_recommendations(model_path, test_users, c_value, topk=20):
    """从MACR-LightGCN模型中提取推荐结果"""
    print(f"从路径 {model_path} 提取MACR-LightGCN模型推荐结果，c值={c_value}...")
    
    # 导入必要的模块
    sys.path.append('./macr_lightgcn')
    
    # 保存当前argv
    tmp_argv = sys.argv.copy()
    # 创建一个包含正确数据集路径的参数列表
    sys.argv = [sys.argv[0], '--dataset', 'addressa', '--data_path', './data/']
    
    try:
        # 直接导入需要的类，避免执行batch_test.py中的代码
        from macr_lightgcn.utility.load_data import Data
        # 然后再导入LightGCN模型
        from macr_lightgcn.LightGCN import LightGCN
        
        # 恢复argv，以防导入过程中被修改
        sys.argv = tmp_argv
        
        # 创建参数对象
        class Args:
            def __init__(self):
                # 正确设置数据集名称为addressa而不是从模型路径中提取
                self.dataset = "addressa"
                self.adj_type = 'pre'
                self.alg_type = 'lightgcn'
                self.layer_size = '[64,64]'
                self.embed_size = 64
                self.batch_size = 1024
                self.node_dropout = '[0.1]'
                self.mess_dropout = '[0.1]'
                self.regs = '[1e-5,1e-5,1e-2]'
                self.lr = 0.0001
                self.model_type = 'LightGCN'
                self.valid_set = "test"  # 使用test数据集进行评估
                self.test_flag = "part"  # 测试类型
                self.save_flag = 0  # 是否保存模型
                self.early_stop = 1  # 是否早停
                self.pretrain = 0  # 是否预训练
                self.verbose = 1  # 输出详细程度
                
        args = Args()
        
    except ImportError as e:
        print(f"导入LightGCN模块失败: {e}")
        sys.argv = tmp_argv  # 确保在异常情况下也恢复argv
        return {}
    
    # 加载数据配置
    try:
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
    except Exception as e:
        print(f"加载数据配置失败: {e}")
        return {}
    
    # 创建模型
    with tf.Graph().as_default():
        model = LightGCN(data_config=config, pretrain_data=None)
        
        # 创建会话
        sess = tf.compat.v1.Session()
        
        # 初始化变量
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # 创建一个只包含现有变量的saver
        var_list = {}
        reader = tf.compat.v1.train.NewCheckpointReader(model_path)
        available_vars = reader.get_variable_to_shape_map()
        
        for var in tf.compat.v1.global_variables():
            # 获取变量名称（去掉前缀":")
            var_name = var.name.split(':')[0]
            if var_name in available_vars:
                var_list[var_name] = var
        
        saver = tf.compat.v1.train.Saver(var_list=var_list)
        
        try:
            saver.restore(sess, model_path)
            print(f"成功加载模型(部分变量): {model_path}")
            print(f"加载了 {len(var_list)} 个变量，模型总变量数 {len(tf.compat.v1.global_variables())}")
            
            # 设置MACR对应的c值
            try:
                model.update_c(sess, c_value)
                print(f"已设置c值: {c_value}")
            except Exception as e:
                print(f"设置c值失败: {e}，将使用默认值")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            sess.close()
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
            
            try:
                # 使用不同的评分函数尝试
                try:
                    # 首先尝试MACR的评分函数
                    ratings = sess.run(model.rubi_ratings_both, 
                                    {model.users: user_batch, model.pos_items: items,
                                     model.node_dropout: [0.] * len(eval(args.layer_size)),
                                     model.mess_dropout: [0.] * len(eval(args.layer_size))})
                except Exception as e:
                    print(f"使用rubi_ratings_both失败: {e}，尝试使用batch_ratings")
                    # 如果MACR评分函数失败，使用普通评分函数
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
            except Exception as e:
                print(f"批次 {batch_idx+1}/{n_batches} 评分失败: {e}")
                continue
        
        sess.close()
    
    return recommendations

def main():
    # 恢复原始命令行参数
    restore_args()
    
    parser = argparse.ArgumentParser(description='从模型中提取Top-K推荐结果')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['mf', 'macrmf', 'lightgcn', 'macrlightgcn'],
                        help='模型类型：mf, macrmf, lightgcn, macrlightgcn')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型检查点路径')
    parser.add_argument('--data_path', type=str, default='data/addressa', 
                        help='数据集路径')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出文件路径，如 results/mf_recommendations.json')
    parser.add_argument('--c_value', type=float, default=40.0,
                        help='MACR模型的c值，仅适用于macrmf和macrlightgcn模型')
    parser.add_argument('--topk', type=int, default=20, 
                        help='推荐列表长度')
    parser.add_argument('--dataset', type=str, default='addressa',
                        help='数据集名称')
    
    args = parser.parse_args()
    
    # 提取数据集名称，去除路径中的"data/"前缀
    dataset_name = args.dataset
    if args.data_path.startswith('data/'):
        dataset_name = args.data_path.split('data/')[1].split('/')[0]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载测试用户
    test_file = os.path.join(args.data_path, 'test.txt')
    test_users = load_test_users(test_file)
    print(f"共加载了{len(test_users)}个测试用户")
    
    # 根据模型类型提取推荐结果
    start_time = time.time()
    
    if args.model == 'mf':
        recommendations = extract_mf_recommendations(args.model_path, test_users, args.topk)
    elif args.model == 'macrmf':
        recommendations = extract_macrmf_recommendations(args.model_path, test_users, args.c_value, args.topk)
    elif args.model == 'lightgcn':
        recommendations = extract_lightgcn_recommendations(args.model_path, test_users, args.topk)
    elif args.model == 'macrlightgcn':
        recommendations = extract_macrlightgcn_recommendations(args.model_path, test_users, args.c_value, args.topk)
    
    elapsed = time.time() - start_time
    
    # 保存推荐结果
    if recommendations:
        # 将用户ID转为字符串，以便JSON序列化
        json_recommendations = {str(u): r for u, r in recommendations.items()}
        with open(args.output_file, 'w') as f:
            json.dump(json_recommendations, f)
        print(f"{args.model}模型的推荐结果已保存到 {args.output_file}，耗时: {elapsed:.2f}秒")
    else:
        print(f"{args.model}模型的推荐结果提取失败")

if __name__ == "__main__":
    main() 