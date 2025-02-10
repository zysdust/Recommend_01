import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import scipy.sparse as sp
from LightGCN import LightGCN
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# 全局配置参数
N_EPOCHS = 60  # 训练轮数
DATASET_CHOICE = 'rtr'  # 可选: 'modcloth' 或 'rtr'
DATA_PATH = {
    'modcloth': 'Data_full/modcloth_final_data_processed.json',
    'rtr': 'Data_full/renttherunway_final_data_processed.json'
}
RATING_FIELD = {
    'modcloth': 'quality',
    'rtr': 'rating'
}
MAX_DATA_SAMPLES = 10000  # 使用的最大数据条数，如果大于数据集大小则使用全部数据

def load_data(file_path, rating_field, max_samples=None):
    """加载数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 提取用户ID、物品ID和评分，并进行归一化处理
    interactions = [(d['user_id'], d['item_id'], float(d[rating_field])/10.0) for d in data]
    df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])
    
    # 如果指定了最大样本数，则只使用前max_samples条数据
    if max_samples and max_samples < len(df):
        df = df.head(max_samples)
    
    # 重新映射用户ID和物品ID
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(item_ids)}
    
    df['user_id'] = df['user_id'].map(user_id_map)
    df['item_id'] = df['item_id'].map(item_id_map)
    
    return df, len(user_ids), len(item_ids)

def create_sparse_matrix(df, n_users, n_items):
    """创建稀疏交互矩阵"""
    # 创建COO格式的稀疏矩阵
    rows = df['user_id'].values
    cols = df['item_id'].values
    data = np.ones_like(rows)
    
    return sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

def convert_sp_mat_to_sp_tensor(X):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    coo = X.tocoo()
    indices = torch.LongTensor([coo.row, coo.col])
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sample_negative_items(df, n_items, n_neg=1):
    """采样负样本"""
    user_pos_items = defaultdict(set)
    for user, item in zip(df['user_id'], df['item_id']):
        user_pos_items[user].add(item)
    
    neg_items = []
    for user in df['user_id']:
        user_neg_items = []
        while len(user_neg_items) < n_neg:
            neg_item = random.randint(0, n_items-1)
            if neg_item not in user_pos_items[user]:
                user_neg_items.append(neg_item)
        neg_items.extend(user_neg_items)
    
    return neg_items

def calculate_metrics(model, data, user_embeddings, item_embeddings, device):
    """计算MAE、RMSE、MSE指标"""
    model.eval()
    with torch.no_grad():
        users = torch.LongTensor(data['user_id'].values).to(device)
        items = torch.LongTensor(data['item_id'].values).to(device)
        true_ratings = torch.FloatTensor(data['rating'].values).to(device)
        
        # 获取预测评分
        user_emb = user_embeddings[users]
        item_emb = item_embeddings[items]
        pred_ratings = torch.sum(user_emb * item_emb, dim=1)
        
        # 转换为numpy计算指标
        pred_ratings = pred_ratings.cpu().numpy()
        true_ratings = true_ratings.cpu().numpy()
        
        mae = mean_absolute_error(true_ratings, pred_ratings)
        mse = mean_squared_error(true_ratings, pred_ratings)
        rmse = np.sqrt(mse)
        
        return mae, rmse, mse

def plot_metrics(train_metrics, test_metrics, dataset_name, save_dir='results'):
    """绘制评价指标折线图"""
    metrics_names = ['MAE', 'RMSE', 'MSE', 'Loss']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name} Dataset Metrics', fontsize=16)
    
    # 绘制按轮数的指标
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        # 训练集指标
        ax.plot(train_metrics['epochs'], train_metrics[metric.lower()],
                label=f'Train {metric}', marker='o')
        # 测试集指标
        ax.plot(train_metrics['epochs'], test_metrics[metric.lower()],
                label=f'Test {metric}', marker='s')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()
        ax.set_title(f'{metric} vs Epochs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name.lower()}_metrics_epochs.png'))
    plt.close()
    
    # 绘制按时间的指标
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name} Dataset Metrics (Time)', fontsize=16)
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        # 训练集指标
        ax.plot(train_metrics['time'], train_metrics[metric.lower()],
                label=f'Train {metric}', marker='o')
        # 测试集指标
        ax.plot(train_metrics['time'], test_metrics[metric.lower()],
                label=f'Test {metric}', marker='s')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()
        ax.set_title(f'{metric} vs Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name.lower()}_metrics_time.png'))
    plt.close()

def save_metrics_to_csv(train_metrics, test_metrics, dataset_name, save_dir='results'):
    """保存评价指标到CSV文件"""
    # 创建训练集指标DataFrame
    train_df = pd.DataFrame({
        'epoch': train_metrics['epochs'],
        'time': train_metrics['time'],
        'train_mae': train_metrics['mae'],
        'train_rmse': train_metrics['rmse'],
        'train_mse': train_metrics['mse'],
        'train_loss': train_metrics['loss']
    })
    
    # 创建测试集指标DataFrame
    test_df = pd.DataFrame({
        'test_mae': test_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'test_mse': test_metrics['mse'],
        'test_loss': test_metrics['loss']
    })
    
    # 合并训练集和测试集指标
    metrics_df = pd.concat([train_df, test_df], axis=1)
    
    # 保存到CSV文件
    csv_path = os.path.join(save_dir, f'{dataset_name.lower()}_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    return metrics_df

def train_model(model, train_data, test_data, n_items, device, optimizer, adj, batch_size=1024, n_epochs=50):
    """训练模型"""
    print("Training model...")
    train_metrics = defaultdict(list)
    test_metrics = defaultdict(list)
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        total_bpr_loss = 0
        total_reg_loss = 0
        
        # 打乱训练数据
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        
        # 生成训练批次
        n_batches = len(train_data) // batch_size + 1
        
        with tqdm(range(n_batches), desc=f'Epoch {epoch}/{n_epochs}') as pbar:
            for batch in pbar:
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, len(train_data))
                
                if start_idx >= end_idx:
                    break
                    
                batch_data = train_data.iloc[start_idx:end_idx]
                users = torch.LongTensor(batch_data['user_id'].values).to(device)
                pos_items = torch.LongTensor(batch_data['item_id'].values).to(device)
                
                # 采样负例
                neg_items = torch.LongTensor(sample_negative_items(batch_data, n_items)).to(device)
                
                # 前向传播
                user_embeddings, item_embeddings = model(adj)
                
                # 计算损失
                bpr_loss, reg_loss = model.calculate_loss(user_embeddings, item_embeddings, 
                                                        users, pos_items, neg_items)
                loss = bpr_loss + reg_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_bpr_loss += bpr_loss.item()
                total_reg_loss += reg_loss.item()
                
                pbar.set_postfix({'bpr_loss': bpr_loss.item(), 'reg_loss': reg_loss.item()})
        
        # 计算训练和测试指标
        model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = model(adj)
            train_mae, train_rmse, train_mse = calculate_metrics(model, train_data, user_embeddings, item_embeddings, device)
            test_mae, test_rmse, test_mse = calculate_metrics(model, test_data, user_embeddings, item_embeddings, device)
            
            # 记录指标
            train_metrics['mae'].append(train_mae)
            train_metrics['rmse'].append(train_rmse)
            train_metrics['mse'].append(train_mse)
            train_metrics['loss'].append(total_bpr_loss/n_batches)
            train_metrics['epochs'].append(epoch)
            train_metrics['time'].append(time.time() - start_time)
            
            test_metrics['mae'].append(test_mae)
            test_metrics['rmse'].append(test_rmse)
            test_metrics['mse'].append(test_mse)
            test_metrics['loss'].append(test_mae)  # 使用MAE作为测试集损失
            test_metrics['epochs'].append(epoch)
            test_metrics['time'].append(time.time() - start_time)
            
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"Train - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, "
                  f"MSE: {train_mse:.4f}, Loss: {total_bpr_loss/n_batches:.4f}")
            print(f"Test  - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, "
                  f"MSE: {test_mse:.4f}, Loss: {test_mae:.4f}")
    
    # 保存指标
    save_metrics_to_csv(train_metrics, test_metrics, DATASET_CHOICE)
    
    # 保存模型
    torch.save(model.state_dict(), f'results/lightgcn_{DATASET_CHOICE}_model.pth')
    
    return train_metrics, test_metrics

def evaluate_model(model, test_data, train_data, device, adj, k=10):
    """评估模型"""
    model.eval()
    
    # 获取训练集中每个用户交互过的物品
    user_train_items = defaultdict(set)
    for user, item in zip(train_data['user_id'], train_data['item_id']):
        user_train_items[user].add(item)
    
    # 计算评估指标
    precisions = []
    recalls = []
    ndcgs = []
    
    with torch.no_grad():
        user_embeddings, item_embeddings = model(adj)
        
        for user in tqdm(test_data['user_id'].unique(), desc='Evaluating'):
            # 获取用户的测试集物品
            test_items = set(test_data[test_data['user_id'] == user]['item_id'])
            
            if not test_items:
                continue
                
            # 预测评分
            user_tensor = torch.LongTensor([user]).to(device)
            scores = model.predict(user_embeddings, item_embeddings, user_tensor)
            scores = scores.cpu().numpy().flatten()
            
            # 将训练集中的物品分数设为负无穷
            scores[list(user_train_items[user])] = float('-inf')
            
            # 获取topk推荐
            top_k_items = np.argsort(-scores)[:k]
            
            # 计算指标
            n_rel = len(test_items)
            n_rel_and_rec_k = len(set(top_k_items) & test_items)
            
            precisions.append(n_rel_and_rec_k / k)
            recalls.append(n_rel_and_rec_k / n_rel)
            
            # 计算NDCG
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(n_rel, k))])
            dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(top_k_items) if item in test_items])
            ndcgs.append(dcg / idcg if idcg != 0 else 0)
    
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'ndcg': np.mean(ndcgs)
    }

if __name__ == '__main__':
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 模型超参数
    EMBEDDING_DIM = 64
    N_LAYERS = 3
    LEARNING_RATE = 0.001
    DECAY = 1e-4
    BATCH_SIZE = 1024
    
    # 加载选定的数据集
    print(f"Loading {DATASET_CHOICE} data...")
    df, n_users, n_items = load_data(
        DATA_PATH[DATASET_CHOICE], 
        RATING_FIELD[DATASET_CHOICE],
        MAX_DATA_SAMPLES
    )
    print(f"Loaded {len(df)} interactions")
    print(f"Users: {n_users}, Items: {n_items}")
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # 创建交互矩阵
    train_mat = create_sparse_matrix(train_data, n_users, n_items)
    
    # 创建模型
    model = LightGCN(n_users=n_users,
                    n_items=n_items,
                    embedding_dim=EMBEDDING_DIM,
                    n_layers=N_LAYERS,
                    device=device,
                    decay=DECAY).to(device)
    
    # 创建邻接矩阵
    adj_mat = model.create_adj_mat(train_mat)
    adj = convert_sp_mat_to_sp_tensor(adj_mat).to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    
    # 训练模型并记录指标
    train_metrics, test_metrics = train_model(
        model, train_data, test_data, n_items, device, optimizer, adj,
        batch_size=BATCH_SIZE, n_epochs=N_EPOCHS)
    
    # 绘制评价指标折线图
    plot_metrics(train_metrics, test_metrics, DATASET_CHOICE)
    
    # 保存指标到CSV文件
    metrics_df = save_metrics_to_csv(train_metrics, test_metrics, DATASET_CHOICE)
    
    # 保存模型
    save_path = f'results/lightgcn_{DATASET_CHOICE.lower()}_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }, save_path)
    print(f"Model saved to {save_path}")
