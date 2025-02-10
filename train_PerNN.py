import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PerNN import PerNN

# 选择要使用的数据集: 'modcloth' 或 'renttherunway'
SELECTED_DATASET = 'modcloth'

# 指定使用的最大数据条数，如果为None则使用全部数据
MAX_SAMPLES = 10000

# 全局配置参数
EPOCHS = 60  # 训练轮数
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EMBEDDING_DIM = 16
HIDDEN_LAYERS = [64, 32]
DROPOUT = 0.1

# 数据集配置
DATASET_CONFIG = {
    'modcloth': {
        'path': 'Data_full/modcloth_final_data_processed.json',
        'rating_col': 'quality'
    },
    'renttherunway': {
        'path': 'Data_full/renttherunway_final_data_processed.json',
        'rating_col': 'rating'
    }
}



# 确保结果保存目录存在
if not os.path.exists('results'):
    os.makedirs('results')

def load_data(file_path, rating_col='rating', max_samples=None):
    """加载数据并进行预处理"""
    # 读取数据
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # 如果指定了最大样本数，则截取数据
    if max_samples is not None and max_samples < len(data):
        data = data[:max_samples]
    
    df = pd.DataFrame(data)
    print(f"加载数据条数: {len(df)}")
    
    # 对用户和商品ID进行编码
    le_dict = {}
    categorical_cols = ['user_id', 'item_id']
    feature_dims = 0
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        feature_dims += len(le.classes_)
    
    # 获取用户数量和商品数量
    num_users = len(le_dict['user_id'].classes_)
    num_items = len(le_dict['item_id'].classes_)
    
    # 准备特征和标签
    features = df[categorical_cols].values
    user_ids = df['user_id'].values
    item_ids = df['item_id'].values
    
    # 构建用户历史交互记录
    user_history = {}
    for user_id, item_id in zip(df['user_id'], df['item_id']):
        if user_id not in user_history:
            user_history[user_id] = []
        user_history[user_id].append(item_id)
    
    # 将历史记录转换为固定长度的序列（取最近的N个交互）
    max_history_len = 10
    history_matrix = np.zeros((len(df), max_history_len), dtype=np.int64)
    for i, user_id in enumerate(df['user_id']):
        history = user_history[user_id][:-1]  # 排除当前交互
        if len(history) > max_history_len:
            history = history[-max_history_len:]
        history_matrix[i, :len(history)] = history
    
    # 根据不同数据集处理评分
    if rating_col == 'rating':
        labels = df[rating_col].astype(float).values / 10.0
    else:
        labels = df[rating_col].astype(float).values / 5.0
    
    return features, user_ids, item_ids, history_matrix, labels, le_dict, feature_dims, num_users, num_items

def train_model(model, train_data, test_data, epochs=50, batch_size=256, learning_rate=0.001):
    """训练PerNN模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 解包训练数据和测试数据
    (train_features, train_user_ids, train_item_ids, 
     train_history, train_labels) = train_data
    (test_features, test_user_ids, test_item_ids, 
     test_history, test_labels) = test_data
    
    # 转换为PyTorch张量
    train_features = torch.LongTensor(train_features)
    train_user_ids = torch.LongTensor(train_user_ids)
    train_item_ids = torch.LongTensor(train_item_ids)
    train_history = torch.LongTensor(train_history)
    train_labels = torch.FloatTensor(train_labels)
    
    test_features = torch.LongTensor(test_features)
    test_user_ids = torch.LongTensor(test_user_ids)
    test_item_ids = torch.LongTensor(test_item_ids)
    test_history = torch.LongTensor(test_history)
    test_labels = torch.FloatTensor(test_labels)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        train_features, train_user_ids, train_item_ids, train_history, train_labels
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # 初始化指标记录
    metrics = {
        'train_mae': [], 'train_rmse': [], 'train_mse': [], 'train_loss': [],
        'test_mae': [], 'test_rmse': [], 'test_mse': [], 'test_loss': [],
        'time': []
    }
    start_time = time.time()
    
    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_preds = []
        train_true = []
        
        for batch_features, batch_user_ids, batch_item_ids, batch_history, batch_labels in train_loader:
            # 前向传播
            outputs = model(batch_features, batch_user_ids, batch_item_ids, batch_history)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())
        
        # 计算训练集指标
        train_mae = mean_absolute_error(train_true, train_preds)
        train_mse = mean_squared_error(train_true, train_preds)
        train_rmse = np.sqrt(train_mse)
        train_loss = total_loss / len(train_loader)
        
        # 计算测试集指标
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features, test_user_ids, test_item_ids, test_history)
            test_loss = criterion(test_outputs, test_labels).item()
            test_mae = mean_absolute_error(test_labels.cpu().numpy(), test_outputs.cpu().numpy())
            test_mse = mean_squared_error(test_labels.cpu().numpy(), test_outputs.cpu().numpy())
            test_rmse = np.sqrt(test_mse)
        
        # 记录指标
        metrics['train_mae'].append(train_mae)
        metrics['train_rmse'].append(train_rmse)
        metrics['train_mse'].append(train_mse)
        metrics['train_loss'].append(train_loss)
        metrics['test_mae'].append(test_mae)
        metrics['test_rmse'].append(test_rmse)
        metrics['test_mse'].append(test_mse)
        metrics['test_loss'].append(test_loss)
        metrics['time'].append(time.time() - start_time)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train RMSE: {train_rmse:.4f}, '
                  f'Test RMSE: {test_rmse:.4f}')
    
    return metrics

def plot_metrics(metrics, dataset_name, x_label='Epochs'):
    """绘制训练过程中的评价指标变化"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建以轮数为横坐标的图
    plt.figure(figsize=(20, 15))
    
    # 绘制RMSE
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_rmse'], label='训练集RMSE', marker='o', markersize=2)
    plt.plot(metrics['test_rmse'], label='测试集RMSE', marker='s', markersize=2)
    plt.xlabel('训练轮数')
    plt.ylabel('RMSE值')
    plt.title('RMSE随训练轮数的变化')
    plt.grid(True)
    plt.legend()
    
    # 绘制MAE
    plt.subplot(2, 2, 2)
    plt.plot(metrics['train_mae'], label='训练集MAE', marker='o', markersize=2)
    plt.plot(metrics['test_mae'], label='测试集MAE', marker='s', markersize=2)
    plt.xlabel('训练轮数')
    plt.ylabel('MAE值')
    plt.title('MAE随训练轮数的变化')
    plt.grid(True)
    plt.legend()
    
    # 绘制MSE
    plt.subplot(2, 2, 3)
    plt.plot(metrics['train_mse'], label='训练集MSE', marker='o', markersize=2)
    plt.plot(metrics['test_mse'], label='测试集MSE', marker='s', markersize=2)
    plt.xlabel('训练轮数')
    plt.ylabel('MSE值')
    plt.title('MSE随训练轮数的变化')
    plt.grid(True)
    plt.legend()
    
    # 绘制Loss
    plt.subplot(2, 2, 4)
    plt.plot(metrics['train_loss'], label='训练集Loss', marker='o', markersize=2)
    plt.plot(metrics['test_loss'], label='测试集Loss', marker='s', markersize=2)
    plt.xlabel('训练轮数')
    plt.ylabel('Loss值')
    plt.title('Loss随训练轮数的变化')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}_metrics_by_epochs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建以时间为横坐标的图
    plt.figure(figsize=(20, 15))
    
    # 绘制RMSE
    plt.subplot(2, 2, 1)
    plt.plot(metrics['time'], metrics['train_rmse'], label='训练集RMSE', marker='o', markersize=2)
    plt.plot(metrics['time'], metrics['test_rmse'], label='测试集RMSE', marker='s', markersize=2)
    plt.xlabel('训练时间(秒)')
    plt.ylabel('RMSE值')
    plt.title('RMSE随训练时间的变化')
    plt.grid(True)
    plt.legend()
    
    # 绘制MAE
    plt.subplot(2, 2, 2)
    plt.plot(metrics['time'], metrics['train_mae'], label='训练集MAE', marker='o', markersize=2)
    plt.plot(metrics['time'], metrics['test_mae'], label='测试集MAE', marker='s', markersize=2)
    plt.xlabel('训练时间(秒)')
    plt.ylabel('MAE值')
    plt.title('MAE随训练时间的变化')
    plt.grid(True)
    plt.legend()
    
    # 绘制MSE
    plt.subplot(2, 2, 3)
    plt.plot(metrics['time'], metrics['train_mse'], label='训练集MSE', marker='o', markersize=2)
    plt.plot(metrics['time'], metrics['test_mse'], label='测试集MSE', marker='s', markersize=2)
    plt.xlabel('训练时间(秒)')
    plt.ylabel('MSE值')
    plt.title('MSE随训练时间的变化')
    plt.grid(True)
    plt.legend()
    
    # 绘制Loss
    plt.subplot(2, 2, 4)
    plt.plot(metrics['time'], metrics['train_loss'], label='训练集Loss', marker='o', markersize=2)
    plt.plot(metrics['time'], metrics['test_loss'], label='测试集Loss', marker='s', markersize=2)
    plt.xlabel('训练时间(秒)')
    plt.ylabel('Loss值')
    plt.title('Loss随训练时间的变化')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}_metrics_by_time.png', dpi=300, bbox_inches='tight')

def save_metrics_to_csv(metrics, dataset_name):
    """保存训练指标到CSV文件"""
    results = pd.DataFrame({
        'epoch': range(1, len(metrics['train_rmse']) + 1),
        'train_rmse': metrics['train_rmse'],
        'test_rmse': metrics['test_rmse'],
        'train_mae': metrics['train_mae'],
        'test_mae': metrics['test_mae'],
        'train_mse': metrics['train_mse'],
        'test_mse': metrics['test_mse'],
        'train_loss': metrics['train_loss'],
        'test_loss': metrics['test_loss'],
        'time': metrics['time']
    })
    
    # 保存到results目录
    output_path = f'results/{dataset_name}_metrics.csv'
    results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n指标已保存到: {output_path}")
    
    # 打印列名，方便查看保存了哪些指标
    print(f"保存的指标列：{', '.join(results.columns)}")

def main():
    # 获取选中数据集的配置
    if SELECTED_DATASET not in DATASET_CONFIG:
        raise ValueError(f"无效的数据集名称: {SELECTED_DATASET}")
    
    dataset_config = DATASET_CONFIG[SELECTED_DATASET]
    print(f"\n使用数据集: {SELECTED_DATASET}")
    print(f"数据集路径: {dataset_config['path']}")
    print(f"评分列名: {dataset_config['rating_col']}")
    print(f"最大数据条数: {MAX_SAMPLES if MAX_SAMPLES is not None else '全部'}")
    
    # 加载数据
    features, user_ids, item_ids, history_matrix, labels, le_dict, feature_dims, num_users, num_items = load_data(
        dataset_config['path'],
        dataset_config['rating_col'],
        MAX_SAMPLES
    )
    
    # 划分训练集和测试集
    indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_data = (
        features[train_indices],
        user_ids[train_indices],
        item_ids[train_indices],
        history_matrix[train_indices],
        labels[train_indices]
    )
    
    test_data = (
        features[test_indices],
        user_ids[test_indices],
        item_ids[test_indices],
        history_matrix[test_indices],
        labels[test_indices]
    )
    
    print(f"\n特征维度: {feature_dims}")
    print(f"用户数量: {num_users}")
    print(f"商品数量: {num_items}")
    print(f"训练集大小: {len(train_indices)}")
    print(f"测试集大小: {len(test_indices)}")
    
    # 创建并训练模型
    model = PerNN(
        num_features=feature_dims,
        num_users=num_users,
        num_items=num_items,
        num_factors=EMBEDDING_DIM,
        hidden_layers=HIDDEN_LAYERS,
        dropout=DROPOUT
    )
    
    metrics = train_model(
        model,
        train_data,
        test_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # 保存结果
    plot_metrics(metrics, SELECTED_DATASET)
    save_metrics_to_csv(metrics, SELECTED_DATASET)
    
    # 打印最终评价指标
    final_epoch = len(metrics['train_rmse']) - 1
    print("\n最终评价指标:")
    print(f"训练集 - RMSE: {metrics['train_rmse'][final_epoch]:.4f}, "
          f"MAE: {metrics['train_mae'][final_epoch]:.4f}, "
          f"MSE: {metrics['train_mse'][final_epoch]:.4f}")
    print(f"测试集 - RMSE: {metrics['test_rmse'][final_epoch]:.4f}, "
          f"MAE: {metrics['test_mae'][final_epoch]:.4f}, "
          f"MSE: {metrics['test_mse'][final_epoch]:.4f}")
    print(f"总训练时间: {metrics['time'][final_epoch]:.2f}秒")

if __name__ == '__main__':
    main() 