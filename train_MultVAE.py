import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from MultVAE import MultVAE
from torch.utils.data import DataLoader, TensorDataset
import os

# 全局配置参数
CONFIG = {
    'NUM_EPOCHS': 60,  # 训练轮数
    'DATASET_NAME': 'modcloth',  # 可选: 'modcloth' 或 'renttherunway'
    'DATA_PATH': {
        'modcloth': 'Data_full/modcloth_final_data_processed.json',
        'renttherunway': 'Data_full/renttherunway_final_data_processed.json'
    },
    'RATING_COL': {
        'modcloth': 'quality',
        'renttherunway': 'rating'
    },
    'MAX_SAMPLES': 10000,  # 使用的最大数据条数，None表示使用全部数据
}

def load_data(file_path, rating_col, max_samples=None):
    # 读取JSON数据
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {file_path}")
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 编码user_id和item_id
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_idx'] = user_encoder.fit_transform(df['user_id'])
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])
    
    # 构建评分矩阵
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    # 将评分转换为float类型
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    
    # 构建稀疏评分矩阵
    rating_matrix = np.zeros((n_users, n_items))
    for _, row in df.iterrows():
        rating_matrix[row['user_idx'], row['item_idx']] = row[rating_col]
    
    # 对每个用户的评分进行归一化
    for i in range(n_users):
        if np.sum(rating_matrix[i]) > 0:  # 只对有评分的用户进行归一化
            rating_matrix[i] = rating_matrix[i] / np.max(rating_matrix[i])
    
    return rating_matrix, n_users, n_items

def calculate_metrics(pred, true):
    # 确保输入是numpy数组
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    
    # 只计算非零元素的指标
    mask = true != 0
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    
    pred_masked = pred[mask]
    true_masked = true[mask]
    
    mae = mean_absolute_error(true_masked, pred_masked)
    mse = mean_squared_error(true_masked, pred_masked)
    rmse = np.sqrt(mse)
    
    return mae, mse, rmse

def train_model(model, train_loader, optimizer, epoch, device, total_anneal_steps=200000):
    model.train()
    total_loss = 0
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    n_valid_batches = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        
        # 计算退火参数
        anneal = min(1.0, (batch_idx + epoch * len(train_loader)) / total_anneal_steps)
        
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar, anneal)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 计算评价指标
        mae, mse, rmse = calculate_metrics(torch.sigmoid(recon_batch), data)
        
        if mae != 0:  # 只统计有效的batch
            total_loss += loss.item()
            total_mae += mae
            total_mse += mse
            total_rmse += rmse
            n_valid_batches += 1
        
        optimizer.step()
    
    # 计算平均值
    if n_valid_batches == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    return (total_loss / n_valid_batches, 
            total_mae / n_valid_batches,
            total_mse / n_valid_batches,
            total_rmse / n_valid_batches)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    n_valid_batches = 0
    
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(device)
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar, anneal=1.0)
            
            # 计算评价指标
            mae, mse, rmse = calculate_metrics(torch.sigmoid(recon_batch), data)
            
            if mae != 0:  # 只统计有效的batch
                total_loss += loss.item()
                total_mae += mae
                total_mse += mse
                total_rmse += rmse
                n_valid_batches += 1
    
    # 计算平均值
    if n_valid_batches == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    return (total_loss / n_valid_batches,
            total_mae / n_valid_batches,
            total_mse / n_valid_batches,
            total_rmse / n_valid_batches)

def save_metrics_to_csv(train_metrics, test_metrics, times, dataset_name):
    # 创建results文件夹（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 准备数据
    metrics_names = ['Loss', 'MAE', 'MSE', 'RMSE']
    epochs = range(1, len(train_metrics[0]) + 1)
    
    # 创建DataFrame
    results_dict = {
        'Epoch': epochs,
        'Time': times
    }
    
    # 添加训练集指标
    for i, metric in enumerate(metrics_names):
        results_dict[f'Train_{metric}'] = train_metrics[i]
        results_dict[f'Test_{metric}'] = test_metrics[i]
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(results_dict)
    csv_path = f'results/{dataset_name.lower()}_metrics.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

def plot_metrics(train_metrics, test_metrics, times, dataset_name):
    # 创建results文件夹（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')
    
    metrics_names = ['Loss', 'MAE', 'MSE', 'RMSE']
    epochs = range(1, len(train_metrics[0]) + 1)
    
    # 创建2x4的子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Training Metrics for {dataset_name} Dataset', fontsize=16)
    
    # 绘制基于epoch的指标
    for i, metric_name in enumerate(metrics_names):
        ax = axes[0, i]
        ax.plot(epochs, train_metrics[i], label='Train')
        ax.plot(epochs, test_metrics[i], label='Test')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Epochs')
        ax.legend()
        ax.grid(True)
    
    # 绘制基于时间的指标
    for i, metric_name in enumerate(metrics_names):
        ax = axes[1, i]
        ax.plot(times, train_metrics[i], label='Train')
        ax.plot(times, test_metrics[i], label='Test')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Time')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name.lower()}_metrics.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 获取配置参数
    dataset_name = CONFIG['DATASET_NAME']
    file_path = CONFIG['DATA_PATH'][dataset_name]
    rating_col = CONFIG['RATING_COL'][dataset_name]
    max_samples = CONFIG['MAX_SAMPLES']
    epochs = CONFIG['NUM_EPOCHS']
    
    print(f"\nTraining on {dataset_name} dataset")
    print(f"Using up to {max_samples if max_samples else 'all'} samples")
    
    # 加载数据
    rating_matrix, n_users, n_items = load_data(file_path, rating_col, max_samples)
    print(f"Users: {n_users}, Items: {n_items}")
    
    # 训练参数
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # 数据分割
    train_matrix, test_matrix = train_test_split(rating_matrix, test_size=0.2, random_state=42)
    
    # 转换为PyTorch数据集
    train_data = TensorDataset(torch.FloatTensor(train_matrix))
    test_data = TensorDataset(torch.FloatTensor(test_matrix))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = MultVAE(input_dim=n_items).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 初始化指标记录列表
    train_metrics = [[] for _ in range(4)]  # [losses, maes, mses, rmses]
    test_metrics = [[] for _ in range(4)]
    training_times = []
    start_time = time.time()
    
    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        # 训练和评估
        train_results = train_model(model, train_loader, optimizer, epoch, device)
        test_results = evaluate(model, test_loader, device)
        
        # 记录时间
        current_time = time.time() - start_time
        training_times.append(current_time)
        
        # 记录指标
        for i in range(4):
            train_metrics[i].append(train_results[i])
            test_metrics[i].append(test_results[i])
        
        # 学习率调整
        scheduler.step(test_results[0])
        
        if test_results[0] < best_test_loss:
            best_test_loss = test_results[0]
            torch.save(model.state_dict(), f'results/multvae_{dataset_name.lower()}_best.pt')
        
        print(f'Epoch: {epoch:03d} Train Loss: {train_results[0]:.4f} Test Loss: {test_results[0]:.4f}')
        print(f'Train MAE: {train_results[1]:.4f} Test MAE: {test_results[1]:.4f}')
        print(f'Train RMSE: {train_results[3]:.4f} Test RMSE: {test_results[3]:.4f}')
    
    # 保存实验结果
    save_metrics_to_csv(train_metrics, test_metrics, training_times, dataset_name)
    plot_metrics(train_metrics, test_metrics, training_times, dataset_name)

if __name__ == '__main__':
    main()
