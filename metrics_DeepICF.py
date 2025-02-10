import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

class MetricsManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_metrics = {
            'loss': [], 'mae': [], 'rmse': [], 'mse': [],
            'time': []
        }
        self.test_metrics = {
            'loss': [], 'mae': [], 'rmse': [], 'mse': [],
            'time': []
        }
        self.start_time = time.time()
        
    def update_metrics(self, y_true, y_pred, loss, is_train=True):
        metrics = self.train_metrics if is_train else self.test_metrics
        
        # 计算各项指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # 更新指标
        metrics['loss'].append(loss)
        metrics['mae'].append(mae)
        metrics['rmse'].append(rmse)
        metrics['mse'].append(mse)
        metrics['time'].append(time.time() - self.start_time)
    
    def plot_metrics(self):
        # 创建results文件夹（如果不存在）
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'{self.model_name} Training Metrics', fontsize=16)
        
        metrics_names = ['loss', 'mae', 'rmse', 'mse']
        
        # 绘制训练轮数-指标图
        for idx, metric in enumerate(metrics_names):
            ax = axes[0, idx]
            epochs = range(1, len(self.train_metrics[metric]) + 1)
            
            # 训练集
            ax.plot(epochs, self.train_metrics[metric], 
                   label='Train', marker='o', markersize=2)
            # 测试集
            if len(self.test_metrics[metric]) > 0:
                test_epochs = np.linspace(1, len(epochs), len(self.test_metrics[metric]))
                ax.plot(test_epochs, self.test_metrics[metric], 
                       label='Test', marker='s', markersize=2)
            
            ax.set_title(f'{metric.upper()} vs Epochs')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True)
        
        # 绘制时间-指标图
        for idx, metric in enumerate(metrics_names):
            ax = axes[1, idx]
            
            # 训练集
            ax.plot(self.train_metrics['time'], self.train_metrics[metric], 
                   label='Train', marker='o', markersize=2)
            # 测试集
            if len(self.test_metrics[metric]) > 0:
                ax.plot(self.test_metrics['time'], self.test_metrics[metric], 
                       label='Test', marker='s', markersize=2)
            
            ax.set_title(f'{metric.upper()} vs Time')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/{self.model_name}_metrics.png', dpi=300, bbox_inches='tight')
        plt.close() 