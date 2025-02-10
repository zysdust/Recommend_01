import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from NFM import NFM

# 全局配置参数
EPOCHS = 60  # 训练轮数
DATASET_NAME = "ModCloth"  # 可选: "ModCloth" 或 "RenttheRunway"
DATASET_PATHS = {
    "ModCloth": {
        "path": "Data_full/modcloth_final_data_processed.json",
        "rating_col": "quality"
    },
    "RenttheRunway": {
        "path": "Data_full/renttherunway_final_data_processed.json",
        "rating_col": "rating"
    }
}
MAX_SAMPLES = 10000  # 使用的最大数据条数，如果大于数据集大小则使用全部数据

def save_metrics_to_csv(metrics_dict, dataset_name):
    """将评价指标保存为CSV文件"""
    # 创建数据框
    df = pd.DataFrame({
        'epoch': range(1, len(metrics_dict['train_mae']) + 1),
        'time': metrics_dict['time'],
        'train_mae': metrics_dict['train_mae'],
        'train_rmse': metrics_dict['train_rmse'],
        'train_mse': metrics_dict['train_mse'],
        'train_loss': metrics_dict['train_loss'],
        'test_mae': metrics_dict['test_mae'],
        'test_rmse': metrics_dict['test_rmse'],
        'test_mse': metrics_dict['test_mse'],
        'test_loss': metrics_dict['test_loss']
    })
    
    # 保存到CSV文件
    save_path = f'results/{dataset_name}_metrics.csv'
    df.to_csv(save_path, index=False)
    print(f"指标数据已保存到: {save_path}")

def calculate_metrics(y_true, y_pred):
    """计算各种评价指标"""
    # 确保输入是一维数组
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def plot_metrics(metrics_dict, dataset_name, x_label='Epochs'):
    """绘制评价指标折线图"""
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    metrics = ['MAE', 'RMSE', 'MSE', 'Loss']
    colors = ['b', 'g', 'r', 'c']
    
    for idx, metric in enumerate(metrics):
        plt.subplot(2, 2, idx + 1)
        train_values = metrics_dict[f'train_{metric.lower()}']
        test_values = metrics_dict[f'test_{metric.lower()}']
        x = range(1, len(train_values) + 1) if x_label == 'Epochs' else metrics_dict['time']
        
        plt.plot(x, train_values, f'{colors[idx]}-', label=f'训练集{metric}')
        plt.plot(x, test_values, f'{colors[idx]}--', label=f'测试集{metric}')
        
        plt.xlabel(x_label)
        plt.ylabel(metric)
        plt.title(f'{dataset_name} - {metric}指标变化曲线')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    save_path = f'results/{dataset_name}_{x_label.lower()}.png'
    plt.savefig(save_path)
    plt.close()

def load_data(file_path, rating_col, max_samples=None):
    """加载数据并进行预处理"""
    # 读取数据
    df = pd.read_json(file_path, lines=True)
    
    # 如果指定了最大样本数，则截取数据
    if max_samples is not None and max_samples < len(df):
        df = df.head(max_samples)
    
    # 对分类特征进行编码
    le_dict = {}
    categorical_cols = ['user_id', 'item_id']
    feature_dims = 0
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        feature_dims += len(le.classes_)
    
    # 准备特征和标签
    features = df[categorical_cols].values
    labels = df[rating_col].values
    if rating_col == 'rating':
        labels = labels.astype(float) / 10.0
    else:
        labels = labels.astype(float) / 5.0
    
    return features, labels, le_dict, feature_dims

def train_model(model, train_features, train_labels, test_features, test_labels, 
                epochs=50, batch_size=256, learning_rate=0.001):
    """训练NFM模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 转换为PyTorch张量
    train_features = torch.LongTensor(train_features)
    train_labels = torch.FloatTensor(train_labels)
    test_features = torch.LongTensor(test_features)
    test_labels = torch.FloatTensor(test_labels)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化指标记录
    metrics_dict = {
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
        
        for batch_features, batch_labels in train_loader:
            # 前向传播
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())
        
        # 计算训练集指标
        train_mae, train_mse, train_rmse = calculate_metrics(train_true, train_preds)
        train_loss = total_loss / len(train_loader)
        
        # 计算测试集指标
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features)
            test_loss = criterion(test_outputs, test_labels).item()
            test_mae, test_mse, test_rmse = calculate_metrics(
                test_labels.cpu().numpy(), test_outputs.cpu().numpy())
        
        # 记录指标
        metrics_dict['train_mae'].append(train_mae)
        metrics_dict['train_rmse'].append(train_rmse)
        metrics_dict['train_mse'].append(train_mse)
        metrics_dict['train_loss'].append(train_loss)
        metrics_dict['test_mae'].append(test_mae)
        metrics_dict['test_rmse'].append(test_rmse)
        metrics_dict['test_mse'].append(test_mse)
        metrics_dict['test_loss'].append(test_loss)
        metrics_dict['time'].append(time.time() - start_time)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, '
                  f'RMSE: {train_rmse:.4f}, MSE: {train_mse:.4f}')
            print(f'Test  - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, '
                  f'RMSE: {test_rmse:.4f}, MSE: {test_mse:.4f}')
    
    return metrics_dict

def main():
    # 确保数据集名称有效
    if DATASET_NAME not in DATASET_PATHS:
        raise ValueError(f"无效的数据集名称: {DATASET_NAME}")
    
    # 获取数据集配置
    dataset_config = DATASET_PATHS[DATASET_NAME]
    
    print(f"加载 {DATASET_NAME} 数据集...")
    features, labels, le_dict, feature_dims = load_data(
        dataset_config['path'], 
        dataset_config['rating_col'],
        MAX_SAMPLES
    )
    
    print(f"数据集大小: {len(features)}")
    print(f"特征维度: {feature_dims}")
    
    # 划分训练集和测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    
    # 创建模型
    model = NFM(num_features=feature_dims, num_factors=10)
    
    # 训练模型并获取指标
    metrics_dict = train_model(
        model, train_features, train_labels, test_features, test_labels,
        epochs=EPOCHS
    )
    
    # 保存指标到CSV文件
    save_metrics_to_csv(metrics_dict, DATASET_NAME)
    
    # 绘制评价指标随训练轮数的变化
    plot_metrics(metrics_dict, DATASET_NAME, 'Epochs')
    
    # 绘制评价指标随训练时间的变化
    plot_metrics(metrics_dict, DATASET_NAME, 'Time (s)')

if __name__ == "__main__":
    # 创建results文件夹（如果不存在）
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    main()
