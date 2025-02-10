import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from WDL import WideAndDeep
import json
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# 全局配置参数
NUM_EPOCHS = 60  # 训练轮数
DATASET_CHOICE = 'modcloth'  # 'modcloth', 'rtr', 'both'
MAX_DATA_SAMPLES = 10000  # 使用的最大数据条数，无限表示使用全部数据

# 数据集配置
DATASET_CONFIG = {
    'modcloth': {
        'path': 'Data_full/modcloth_final_data_processed.json',
        'rating_col': 'quality'
    },
    'rtr': {
        'path': 'Data_full/renttherunway_final_data_processed.json',
        'rating_col': 'rating'
    }
}

class RecommendDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids.values)
        self.item_ids = torch.LongTensor(item_ids.values)
        self.ratings = torch.FloatTensor(ratings.values)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

def load_data(file_path, rating_col, max_samples=float('inf')):
    # 读取JSON数据
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    
    # 提取所需列并进行ID映射
    user_ids = pd.Categorical(df['user_id']).codes
    item_ids = pd.Categorical(df['item_id']).codes
    
    # 标准化评分到[0,1]区间
    if rating_col == 'quality':
        ratings = df[rating_col].astype(float) / 5.0
    else:  # rating
        ratings = df[rating_col].astype(float) / 10.0
    
    # 转换为pandas Series    
    user_ids = pd.Series(user_ids)
    item_ids = pd.Series(item_ids)
    ratings = pd.Series(ratings)
        
    return user_ids, item_ids, ratings

def calculate_metrics(y_true, y_pred):
    """计算各种评价指标"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return mae, rmse, mse

def save_metrics_to_csv(metrics_dict, dataset_name, save_dir='results'):
    """将评价指标保存为CSV文件"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 创建DataFrame
    data = {
        'Epoch': metrics_dict['epochs'],
        'Time': metrics_dict['times'],
        'Train_MAE': metrics_dict['train_mae'],
        'Train_RMSE': metrics_dict['train_rmse'],
        'Train_MSE': metrics_dict['train_mse'],
        'Train_Loss': metrics_dict['train_loss'],
        'Test_MAE': metrics_dict['test_mae'],
        'Test_RMSE': metrics_dict['test_rmse'],
        'Test_MSE': metrics_dict['test_mse'],
        'Test_Loss': metrics_dict['test_loss']
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, f'{dataset_name}_metrics.csv'), index=False)

def plot_metrics(metrics_dict, dataset_name, save_dir='results'):
    """绘制评价指标折线图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'{dataset_name}数据集评价指标变化图', fontsize=16)
    
    # 定义指标和位置的映射
    metrics_pos = {
        'MAE': (0, 0),
        'RMSE': (0, 1),
        'MSE': (1, 0),
        'Loss': (1, 1)
    }
    
    # 绘制每个指标的两条线（训练集和测试集）
    for metric, (i, j) in metrics_pos.items():
        ax = axes[i, j]
        metric_lower = metric.lower()
        
        # 基于轮数的线
        train_line = ax.plot(metrics_dict['epochs'], 
                           metrics_dict[f'train_{metric_lower}'],
                           label=f'训练集 vs 轮数', 
                           color='blue')
        test_line = ax.plot(metrics_dict['epochs'],
                          metrics_dict[f'test_{metric_lower}'],
                          label=f'测试集 vs 轮数',
                          color='green')
        
        # 基于时间的线
        ax2 = ax.twinx()
        train_time_line = ax2.plot(metrics_dict['epochs'],
                                 metrics_dict[f'train_{metric_lower}'],
                                 label=f'训练集 vs 时间',
                                 color='red',
                                 linestyle='--')
        test_time_line = ax2.plot(metrics_dict['epochs'],
                                metrics_dict[f'test_{metric_lower}'],
                                label=f'测试集 vs 时间',
                                color='orange',
                                linestyle='--')
        
        # 设置标题和标签
        ax.set_title(f'{metric}指标变化')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel(f'{metric}值')
        ax2.set_ylabel('时间(秒)')
        
        # 合并两个y轴的图例
        lines = train_line + test_line + train_time_line + test_time_line
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_metrics_combined.png'))
    plt.close()

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    metrics_dict = {
        'epochs': list(range(1, num_epochs + 1)),
        'times': [],
        'train_mae': [], 'train_rmse': [], 'train_mse': [], 'train_loss': [],
        'test_mae': [], 'test_rmse': [], 'test_mse': [], 'test_loss': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_preds = []
        train_trues = []
        
        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs.squeeze(), ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_preds.extend(outputs.squeeze().detach().cpu().numpy())
            train_trues.extend(ratings.cpu().numpy())
        
        # 计算训练集指标
        train_mae, train_rmse, train_mse = calculate_metrics(train_trues, train_preds)
        avg_train_loss = total_loss / len(train_loader)
        
        # 测试阶段
        model.eval()
        total_test_loss = 0
        test_preds = []
        test_trues = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                outputs = model(user_ids, item_ids)
                loss = criterion(outputs.squeeze(), ratings)
                total_test_loss += loss.item()
                
                test_preds.extend(outputs.squeeze().cpu().numpy())
                test_trues.extend(ratings.cpu().numpy())
        
        # 计算测试集指标
        test_mae, test_rmse, test_mse = calculate_metrics(test_trues, test_preds)
        avg_test_loss = total_test_loss / len(test_loader)
        
        # 记录时间和指标
        current_time = time.time() - start_time
        metrics_dict['times'].append(current_time)
        metrics_dict['train_mae'].append(train_mae)
        metrics_dict['train_rmse'].append(train_rmse)
        metrics_dict['train_mse'].append(train_mse)
        metrics_dict['train_loss'].append(avg_train_loss)
        metrics_dict['test_mae'].append(test_mae)
        metrics_dict['test_rmse'].append(test_rmse)
        metrics_dict['test_mse'].append(test_mse)
        metrics_dict['test_loss'].append(avg_test_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Time: {current_time:.2f}s')
        print(f'Train - Loss: {avg_train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}')
        print(f'Test  - Loss: {avg_test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}')
    
    return metrics_dict

def train_dataset(dataset_name, config, device):
    print(f"\n开始训练 {dataset_name} 数据集...")
    
    # 加载数据
    users, items, ratings = load_data(
        config['path'],
        config['rating_col'],
        MAX_DATA_SAMPLES
    )
    
    print(f"数据集大小: {len(ratings)} 条记录")
    print(f"用户数量: {len(np.unique(users))}") 
    print(f"物品数量: {len(np.unique(items))}")
    
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(range(len(ratings)), test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_dataset = RecommendDataset(
        users[train_indices],
        items[train_indices],
        ratings[train_indices]
    )
    test_dataset = RecommendDataset(
        users[test_indices],
        items[test_indices],
        ratings[test_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 创建模型
    num_users = len(np.unique(users))
    num_items = len(np.unique(items))
    model = WideAndDeep(num_users, num_items)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型并获取指标
    metrics_dict = train_model(model, train_loader, test_loader, criterion, optimizer, device, NUM_EPOCHS)
    
    # 保存指标到CSV
    save_metrics_to_csv(metrics_dict, dataset_name)
    
    # 绘制并保存评价指标图
    plot_metrics(metrics_dict, dataset_name)
    
    # 保存模型
    torch.save(model.state_dict(), f'wdl_model_{dataset_name.lower()}.pth')
    print(f"{dataset_name} 模型已保存")

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"最大数据条数: {MAX_DATA_SAMPLES if MAX_DATA_SAMPLES != float('inf') else '全部'}")
    
    if DATASET_CHOICE.lower() == 'modcloth':
        train_dataset('ModCloth', DATASET_CONFIG['modcloth'], device)
    elif DATASET_CHOICE.lower() == 'rtr':
        train_dataset('RTR', DATASET_CONFIG['rtr'], device)
    else:  # both
        train_dataset('ModCloth', DATASET_CONFIG['modcloth'], device)
        train_dataset('RTR', DATASET_CONFIG['rtr'], device)

if __name__ == '__main__':
    main()
