#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import json
import argparse
import matplotlib as mpl

# 设置英文字体而非中文，避免字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义英文标签，替换中文标签
GROUP_NAMES_EN = ['Most Popular', 'Popular', 'Less Popular', 'Unpopular']
GROUP_NAMES_ZH = ['最流行', '较流行', '较不流行', '不流行']

def get_item_popularity(data_path):
    """Calculate item popularity"""
    train_file = os.path.join(data_path, 'train.txt')
    item_pop = {}
    
    print(f"加载训练数据: {train_file}")
    try:
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                user = parts[0]
                items = parts[1:]
                
                for item in items:
                    if item not in item_pop:
                        item_pop[item] = 0
                    item_pop[item] += 1
        
        print(f"共有不同物品: {len(item_pop)}")
        return item_pop
    except FileNotFoundError:
        print(f"错误: 未找到训练文件 {train_file}")
        return {}

def split_items_by_popularity(item_pop, n_groups=4):
    """Split items into groups by popularity"""
    if not item_pop:
        print("警告: 物品流行度字典为空")
        return [[] for _ in range(n_groups)]
        
    items_sorted = sorted(item_pop.items(), key=lambda x: x[1], reverse=True)
    items_per_group = len(items_sorted) // n_groups
    
    groups = []
    for i in range(n_groups):
        start_idx = i * items_per_group
        end_idx = (i + 1) * items_per_group if i < n_groups - 1 else len(items_sorted)
        group_items = [item for item, _ in items_sorted[start_idx:end_idx]]
        groups.append(group_items)
    
    return groups

def load_recommendations(results_path, model_name, topk=20):
    """Load recommendation results"""
    file_path = os.path.join(results_path, f"{model_name}_recommendations.json")
    print(f"加载推荐结果: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            recommendations = json.load(f)
        
        # 调试信息
        if recommendations:
            sample_user = next(iter(recommendations))
            print(f"示例用户 {sample_user}, 推荐数量: {len(recommendations[sample_user])}")
            print(f"推荐物品ID类型: {type(recommendations[sample_user][0])}")
            print(f"推荐物品样例: {recommendations[sample_user][:5]}")
        
        # 确保推荐结果的格式正确 - 将所有键和物品ID转为字符串
        processed_recommendations = {}
        for user, items in recommendations.items():
            user_key = str(user)  # 确保用户ID是字符串
            processed_recommendations[user_key] = [str(item) for item in items[:topk]]
            
        return processed_recommendations
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"JSON解析错误: {file_path}")
        return None

def compute_recall_by_popularity_group(recommendations, test_data, item_groups):
    """Compute recall for each popularity group"""
    n_groups = len(item_groups)
    recalls = [0] * n_groups
    
    # 创建item到组的映射
    item_to_group = {}
    for group_id, items in enumerate(item_groups):
        for item in items:
            item_to_group[item] = group_id
    
    # 打印调试信息
    print(f"物品分组情况:")
    for i, group in enumerate(item_groups):
        print(f"  组 {i+1}: {len(group)} 个物品")
    
    # 计算每个组的总点击次数
    group_clicks = [0] * n_groups
    
    # 计算每个组的命中次数
    group_hits = [0] * n_groups
    
    # 调试信息
    reco_items = set()
    for user, items in recommendations.items():
        reco_items.update(items)
    
    print(f"推荐结果中共有 {len(reco_items)} 个不同物品")
    
    # 检查推荐物品在各分组中的分布
    group_coverage = [0] * n_groups
    for item in reco_items:
        if item in item_to_group:
            group_id = item_to_group[item]
            group_coverage[group_id] += 1
        
    print(f"推荐结果覆盖各组的物品数量: {group_coverage}")
    
    # 检查测试集和推荐集的用户交集
    test_users = set(test_data.keys())
    reco_users = set(recommendations.keys())
    common_users = test_users.intersection(reco_users)
    
    print(f"测试集用户数: {len(test_users)}")
    print(f"推荐集用户数: {len(reco_users)}")
    print(f"共同用户数: {len(common_users)}")
    
    # 记录匹配失败的次数
    user_not_found = 0
    item_not_in_group = 0
    
    for user, test_items in test_data.items():
        if user not in recommendations:
            user_not_found += 1
            continue
            
        recs = recommendations[user]
        for test_item in test_items:
            if test_item in item_to_group:
                group_id = item_to_group[test_item]
                group_clicks[group_id] += 1
                
                if test_item in recs:
                    group_hits[group_id] += 1
            else:
                item_not_in_group += 1
    
    print(f"用户不在推荐集中的次数: {user_not_found}")
    print(f"测试物品不在任何流行度组中的次数: {item_not_in_group}")
    
    # 计算每个组的召回率
    for i in range(n_groups):
        if group_clicks[i] > 0:
            recalls[i] = group_hits[i] / group_clicks[i]
        else:
            recalls[i] = 0
    
    return recalls, group_clicks

def compute_psp(recalls):
    """Compute Popularity SP score (higher means more popularity bias)"""
    n_groups = len(recalls)
    if n_groups < 2:
        return 0
    
    # Calculate PSP using the formula from the paper
    p_max = recalls[0]
    p_min = recalls[-1]
    
    # 检查除数是否为0
    denominator = p_max + p_min
    if denominator == 0:
        return 0
        
    psp = (p_max - p_min) / denominator
    # 添加注释说明负值的含义
    if psp < 0:
        print(f"注意: PSP为负值 ({psp:.4f})，表明模型偏好推荐不太流行的物品，有效减少了流行度偏见")
    return psp

def load_test_data(data_path):
    """Load test data"""
    test_file = os.path.join(data_path, 'test.txt')
    test_data = {}
    
    print(f"加载测试数据: {test_file}")
    try:
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                user = parts[0]
                items = parts[1:]
                test_data[user] = items
        
        return test_data
    except FileNotFoundError:
        print(f"错误: 未找到测试文件 {test_file}")
        return {}

def plot_recalls_by_groups(all_recalls, model_names, n_groups, save_path):
    """Plot recalls by popularity groups"""
    plt.figure(figsize=(12, 7))
    
    bar_width = 0.8 / len(model_names)
    index = np.arange(n_groups)
    
    # 使用更好的颜色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for i, (model, recalls) in enumerate(zip(model_names, all_recalls)):
        plt.bar(index + i * bar_width, recalls, bar_width,
               color=colors[i], label=model)
    
    plt.xlabel('Item Popularity Groups', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.title('Recall by Item Popularity Groups', fontsize=16)
    plt.xticks(index + bar_width * (len(model_names) - 1) / 2, GROUP_NAMES_EN, fontsize=12)
    
    # 计算y轴最大值，确保所有柱状图都可见
    max_recall = 0
    for recalls in all_recalls:
        if recalls and max(recalls) > max_recall:
            max_recall = max(recalls)
    plt.ylim(0, max_recall * 1.2)
    
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_psp_scores(psp_scores, model_names, save_path):
    """Plot PSP scores"""
    plt.figure(figsize=(10, 6))
    
    # 使用更好的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    bars = plt.bar(model_names, psp_scores, width=0.6, color=colors)
    
    # 在柱状图上添加数值
    for bar, score in zip(bars, psp_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', fontsize=12)
    
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('PSP Score', fontsize=14)
    plt.title('Popularity SP Score by Model (Higher = More Popularity Bias)', fontsize=16)
    plt.ylim(0, max(psp_scores) * 1.2 if psp_scores and max(psp_scores) > 0 else 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot recall by popularity groups.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset.')
    parser.add_argument('--results_path', type=str, required=True, help='Path to recommendation results.')
    parser.add_argument('--models', type=str, required=True, help='Comma-separated list of model file prefixes.')
    parser.add_argument('--model_names', type=str, required=True, help='Comma-separated list of model names for the plot.')
    parser.add_argument('--topk', type=int, default=20, help='Top-k recommendations.')
    parser.add_argument('--n_groups', type=int, default=4, help='Number of popularity groups.')
    
    args = parser.parse_args()
    
    # 解析模型名称
    model_file_prefixes = args.models.split(',')
    model_names = args.model_names.split(',')
    
    if len(model_file_prefixes) != len(model_names):
        raise ValueError("Number of model file prefixes must match number of model names.")
    
    # 检查路径是否存在
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据路径不存在: {args.data_path}")
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        print(f"创建结果目录: {args.results_path}")
        
    print(f"处理数据集: {args.data_path}")
    print(f"模型: {', '.join(model_names)}")
    
    # 获取物品流行度
    item_pop = get_item_popularity(args.data_path)
    
    # 按流行度将物品分组
    item_groups = split_items_by_popularity(item_pop, args.n_groups)
    
    # 输出每个组的物品数量
    for i, group in enumerate(item_groups):
        print(f"组 {i+1} ({GROUP_NAMES_EN[i]}): {len(group)} 个物品")
    
    # 加载测试数据
    test_data = load_test_data(args.data_path)
    if not test_data:
        raise ValueError("测试数据为空，无法计算召回率")
    
    # 加载各模型的推荐结果并计算召回率
    all_recalls = []
    psp_scores = []
    group_clicks_all = None
    
    for model_prefix in model_file_prefixes:
        recommendations = load_recommendations(args.results_path, model_prefix, args.topk)
        if recommendations is None:
            print(f"警告: 无法加载模型 {model_prefix} 的推荐结果，使用零值替代")
            recalls = [0] * args.n_groups
            all_recalls.append(recalls)
            psp_scores.append(0)
            continue
            
        recalls, group_clicks = compute_recall_by_popularity_group(recommendations, test_data, item_groups)
        all_recalls.append(recalls)
        
        # 只需记录一次group_clicks，因为对所有模型都是一样的
        if group_clicks_all is None:
            group_clicks_all = group_clicks
        
        psp = compute_psp(recalls)
        psp_scores.append(psp)
    
    # 输出结果
    print("\n流行度组的召回率:")
    for i, model_name in enumerate(model_names):
        recalls_str = ", ".join([f"{r:.4f}" for r in all_recalls[i]])
        print(f"  {model_name}: {recalls_str}")
    
    if group_clicks_all:
        print("\n各组物品点击数量:")
        for i, clicks in enumerate(group_clicks_all):
            print(f"  {GROUP_NAMES_EN[i]}: {clicks}个点击")
    
    print("\n流行度偏差指标 (PSP):")
    for i, model_name in enumerate(model_names):
        print(f"  {model_name}: {psp_scores[i]:.4f}")
    
    # 绘制图表
    fig8_path = os.path.join(args.results_path, "figure8.png")
    fig9_path = os.path.join(args.results_path, "figure9.png")
    
    plot_recalls_by_groups(all_recalls, model_names, args.n_groups, fig8_path)
    plot_psp_scores(psp_scores, model_names, fig9_path)
    
    print(f"图8已保存至: {fig8_path}")
    print(f"图9已保存至: {fig9_path}")

if __name__ == "__main__":
    main() 