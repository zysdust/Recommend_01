import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
from xDeepFM import xDeepFM
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import time

# 全局配置
NUM_EPOCHS = 50  # 增加训练轮数
DATASET_NAME = 'renttherunway'  # 可选: 'modcloth' 或 'renttherunway'
DATA_ROOT = 'Data_full'  # 数据集根目录
DATASET_FILES = {
    'modcloth': 'modcloth_final_data_processed.json',
    'renttherunway': 'renttherunway_final_data_processed.json'
}
MAX_SAMPLES = 1000  # 使用的最大样本数，设为inf则使用全部数据
BATCH_SIZE = 64  # 减小batch size
LEARNING_RATE = 0.0001  # 降低学习率
WEIGHT_DECAY = 1e-5  # 增加正则化
EMBEDDING_DIM = 32  # 增加embedding维度
MLP_DIMS = [256, 128, 64]  # 加深网络
CIN_LAYER_SIZES = [128, 128, 128]  # 增加CIN层数

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
os.makedirs('results', exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'results/{DATASET_NAME}_xDeepFM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class RecommendDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def normalize_ratings(df, source):
    if source == 'modcloth':
        # Z-Score标准化
        quality = df['quality'].values
        df['rating'] = (quality - quality.mean()) / quality.std()
    else:
        # Z-Score标准化
        rating = df['rating'].astype(float).values
        df['rating'] = (rating - rating.mean()) / rating.std()
    return df

def filter_rare_categories(df, categorical_cols, min_freq=2):
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        valid_values = value_counts[value_counts >= min_freq].index
        df = df[df[col].isin(valid_values)]
    return df

def save_metrics_to_csv(train_metrics, test_metrics, train_times, save_dir, dataset_name):
    # 创建训练集和测试集的指标数据框
    train_df = pd.DataFrame(train_metrics)
    test_df = pd.DataFrame(test_metrics)
    
    # 添加epoch和时间信息
    epochs = range(1, len(train_df) + 1)
    train_df['epoch'] = epochs
    train_df['time'] = train_times
    test_df['epoch'] = epochs
    test_df['time'] = train_times
    
    # 重命名列以区分训练集和测试集
    train_df.columns = [f'train_{col}' if col not in ['epoch', 'time'] else col 
                       for col in train_df.columns]
    test_df.columns = [f'test_{col}' if col not in ['epoch', 'time'] else col 
                      for col in test_df.columns]
    
    # 合并训练集和测试集的指标
    metrics_df = pd.merge(train_df, test_df, on=['epoch', 'time'])
    
    # 调整列的顺序
    cols = ['epoch', 'time']
    metrics = ['Loss', 'RMSE', 'MAE', 'MSE']
    for metric in metrics:
        cols.extend([f'train_{metric}', f'test_{metric}'])
    metrics_df = metrics_df[cols]
    
    # 保存到CSV文件
    output_file = os.path.join(save_dir, f'{dataset_name}_all_metrics.csv')
    metrics_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f'所有指标已保存到: {output_file}')

def load_single_dataset(file_path, max_samples=float('inf')):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

def load_data(dataset_name, max_samples=float('inf')):
    file_path = os.path.join(DATA_ROOT, DATASET_FILES[dataset_name])
    df = load_single_dataset(file_path, max_samples)
    logging.info(f'{dataset_name} samples loaded: {len(df)}')
    
    # 选择特征
    categorical_cols = ['user_id', 'item_id', 'size', 'fit', 'category', 'height']
    rating_col = 'quality' if dataset_name == 'modcloth' else 'rating'
    
    # 提取特征和标签
    df_selected = df[categorical_cols + [rating_col]]
    
    # 处理缺失值
    df_selected = df_selected.dropna()
    logging.info(f'After dropna: {len(df_selected)}')
    
    # 标准化评分
    df_selected = normalize_ratings(df_selected, dataset_name)
    
    # 数据分布统计
    logging.info('Feature value counts:')
    for col in categorical_cols:
        value_counts = df_selected[col].value_counts()
        logging.info(f'{col} value counts:\n{value_counts.head()}')
    
    # 标签编码
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df_selected[col] = encoders[col].fit_transform(df_selected[col].astype(str))
    
    # 添加数据统计信息日志
    logging.info(f'Data statistics after preprocessing:')
    for col in categorical_cols:
        logging.info(f'{col} unique values: {df_selected[col].nunique()}')
    logging.info(f'Rating mean: {df_selected["rating"].mean():.2f}, std: {df_selected["rating"].std():.2f}')
    
    return df_selected, encoders, categorical_cols

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions.extend(y_pred.detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
    
    # 计算评估指标
    predictions = np.array(predictions)
    labels = np.array(labels)
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - labels))
    
    return total_loss / len(train_loader), rmse, mae, mse

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            predictions.extend(y_pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
    
    # 计算评估指标
    predictions = np.array(predictions)
    labels = np.array(labels)
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - labels))
    
    return total_loss / len(data_loader), rmse, mae, mse

def plot_metrics(train_metrics, test_metrics, train_times, save_dir, dataset_name):
    metrics = ['Loss', 'RMSE', 'MAE', 'MSE']
    epochs = range(1, len(train_metrics['Loss']) + 1)
    
    # 创建2x4的子图布局
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{dataset_name} xDeepFM模型训练评估指标', fontsize=16)
    
    # 按训练轮数绘制
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        ax.plot(epochs, train_metrics[metric], 'b-', label=f'训练集{metric}')
        ax.plot(epochs, test_metrics[metric], 'r-', label=f'测试集{metric}')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}随训练轮数的变化')
        ax.legend()
        ax.grid(True)
    
    # 按训练时间绘制
    for i, metric in enumerate(metrics):
        ax = axes[1, i]
        ax.plot(train_times, train_metrics[metric], 'b-', label=f'训练集{metric}')
        ax.plot(train_times, test_metrics[metric], 'r-', label=f'测试集{metric}')
        ax.set_xlabel('训练时间(秒)')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}随训练时间的变化')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.close()

def save_experiment_results(dataset_name, config, metrics, save_dir):
    """保存实验配置和结果到CSV文件"""
    import pandas as pd
    from datetime import datetime
    
    # 准备实验数据
    experiment_data = {
        '实验时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        '数据集': dataset_name,
        'Batch Size': config['batch_size'],
        '学习率': config['learning_rate'],
        '权重衰减': config['weight_decay'],
        'Embedding维度': config['embed_dim'],
        'MLP层维度': str(config['mlp_dims']),
        'CIN层维度': str(config['cin_layer_sizes']),
        '最佳RMSE': metrics['best_rmse'],
        '最佳MSE': metrics['best_mse'],
        '最佳MAE': metrics['best_mae'],
        '训练轮数': metrics['epochs'],
        '训练时间(秒)': metrics['train_time']
    }
    
    # 创建或追加到CSV文件
    results_file = os.path.join(save_dir, 'experiment_results.csv')
    df = pd.DataFrame([experiment_data])
    
    if os.path.exists(results_file):
        df.to_csv(results_file, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(results_file, index=False, encoding='utf-8-sig')
    
    logging.info(f'实验结果已保存到: {results_file}')

def main():
    # 设置随机种子
    torch.manual_seed(2024)
    np.random.seed(2024)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 加载数据
    df, encoders, categorical_cols = load_data(DATASET_NAME, MAX_SAMPLES)
    logging.info(f'Total samples: {len(df)}')
    
    # 准备特征和标签
    X = df[categorical_cols].values
    y = df['rating'].values
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2024
    )
    
    # 创建数据加载器
    train_dataset = RecommendDataset(
        torch.LongTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = RecommendDataset(
        torch.LongTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 模型参数
    field_dims = [len(encoders[col].classes_) for col in categorical_cols]
    embed_dim = EMBEDDING_DIM
    mlp_dims = MLP_DIMS
    dropout = 0.3  # 增加dropout
    cin_layer_sizes = CIN_LAYER_SIZES
    
    # 创建模型
    model = xDeepFM(
        field_dims=field_dims,
        embed_dim=embed_dim,
        mlp_dims=mlp_dims,
        dropout=dropout,
        cin_layer_sizes=cin_layer_sizes,
        device=device
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # 训练
    best_rmse = float('inf')
    best_epoch = 0
    
    # 记录指标
    train_metrics = {'Loss': [], 'RMSE': [], 'MAE': [], 'MSE': []}
    test_metrics = {'Loss': [], 'RMSE': [], 'MAE': [], 'MSE': []}
    train_times = []
    start_time = time.time()
    
    logging.info('Start training...')
    for epoch in range(NUM_EPOCHS):
        train_loss, train_rmse, train_mae, train_mse = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_rmse, test_mae, test_mse = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 记录训练时间
        train_times.append(time.time() - start_time)
        
        # 记录指标
        train_metrics['Loss'].append(train_loss)
        train_metrics['RMSE'].append(train_rmse)
        train_metrics['MAE'].append(train_mae)
        train_metrics['MSE'].append(train_mse)
        
        test_metrics['Loss'].append(test_loss)
        test_metrics['RMSE'].append(test_rmse)
        test_metrics['MAE'].append(test_mae)
        test_metrics['MSE'].append(test_mse)
        
        logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        logging.info(f'Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}')
        logging.info(f'Test - Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}')
        logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 记录最佳模型
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_epoch = epoch
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_rmse': best_rmse,
            }, f'results/{DATASET_NAME}_xdeepfm_best.pth')
    
    logging.info('Training finished!')
    logging.info(f'Best RMSE: {best_rmse:.4f} at epoch {best_epoch+1}')
    
    # 训练结束后，保存实验结果
    experiment_config = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'embed_dim': embed_dim,
        'mlp_dims': mlp_dims,
        'cin_layer_sizes': cin_layer_sizes
    }
    
    experiment_metrics = {
        'best_rmse': best_rmse,
        'best_mse': test_metrics['MSE'][np.argmin(test_metrics['RMSE'])],
        'best_mae': test_metrics['MAE'][np.argmin(test_metrics['RMSE'])],
        'epochs': best_epoch + 1,
        'train_time': time.time() - start_time
    }
    
    save_experiment_results(DATASET_NAME, experiment_config, experiment_metrics, 'results')
    
    # 保存指标到CSV
    save_metrics_to_csv(train_metrics, test_metrics, train_times, 'results', DATASET_NAME)
    
    # 绘制并保存指标图
    plot_metrics(train_metrics, test_metrics, train_times, 'results', DATASET_NAME)

if __name__ == '__main__':
    main()
