import os
import tensorflow as tf
import numpy as np
from data_processor_DeepICF import DataProcessor
from xDeepFM_tf import xDeepFM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import time
import math
import pandas as pd

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# 全局配置
EPOCHS = 60  # 训练轮数
DATASET_CHOICE = 'modcloth'  # 可选: 'modcloth', 'rtr', 'both'
DATA_PATH = {
    'modcloth': 'Data_full/modcloth_final_data_processed.json',
    'rtr': 'Data_full/renttherunway_final_data_processed.json'
}
MAX_SAMPLES = 10000  # 使用的最大数据条数，设为inf则使用全部数据

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置参数
class Config:
    def __init__(self):
        self.embedding_dim = 16
        self.dnn_hidden_units = (256, 128, 64)
        self.cin_layer_size = (128, 128)
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 256

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, test_data, dataset_name):
        super(CustomCallback, self).__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.dataset_name = dataset_name
        self.best_mae = float('inf')
        self.history = {
            'epoch': [], 'time': [],
            'train_mae': [], 'train_mse': [], 'train_rmse': [], 'train_loss': [],
            'val_mae': [], 'val_mse': [], 'val_rmse': [], 'val_loss': []
        }
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        # 记录轮数和训练时间
        current_time = time.time() - self.start_time
        self.history['epoch'].append(epoch + 1)
        self.history['time'].append(current_time)
        
        # 获取训练集指标
        train_pred = self.model.predict(self.train_data[0], verbose=0)
        train_mse = mean_squared_error(self.train_data[1], train_pred)
        train_rmse = math.sqrt(train_mse)
        train_mae = mean_absolute_error(self.train_data[1], train_pred)
        
        # 获取测试集指标
        test_pred = self.model.predict(self.test_data[0], verbose=0)
        test_mse = mean_squared_error(self.test_data[1], test_pred)
        test_rmse = math.sqrt(test_mse)
        test_mae = mean_absolute_error(self.test_data[1], test_pred)
        
        # 记录训练集指标
        self.history['train_mae'].append(train_mae)
        self.history['train_mse'].append(train_mse)
        self.history['train_rmse'].append(train_rmse)
        self.history['train_loss'].append(logs['loss'])
        
        # 记录测试集指标
        self.history['val_mae'].append(test_mae)
        self.history['val_mse'].append(test_mse)
        self.history['val_rmse'].append(test_rmse)
        self.history['val_loss'].append(logs['val_loss'])
        
        if test_mae < self.best_mae:
            self.best_mae = test_mae
            logging.info(f'New best MAE for {self.dataset_name}: {test_mae:.4f}')

def save_metrics_to_csv(history, dataset_name):
    # 将历史记录转换为DataFrame
    df = pd.DataFrame(history)
    
    # 保存到CSV文件
    output_path = f'results/{dataset_name.lower().replace(" ", "_")}_metrics.csv'
    df.to_csv(output_path, index=False)
    logging.info(f"Metrics saved to {output_path}")

def plot_metrics(modcloth_history, rtr_history):
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle('xDeepFM Training Metrics', fontsize=16)
    
    metrics = ['mae', 'mse', 'rmse', 'loss']
    datasets = {}
    
    if modcloth_history is not None:
        datasets['ModCloth'] = modcloth_history
    if rtr_history is not None:
        datasets['Rent The Runway'] = rtr_history
    
    # 绘制按轮数的指标
    for i, metric in enumerate(metrics):
        ax = axes[i, 0]
        for dataset_name, history in datasets.items():
            epochs = range(1, len(history['train_' + metric]) + 1)
            ax.plot(epochs, history['train_' + metric], 'o-', label=f'{dataset_name} Train')
            ax.plot(epochs, history['val_' + metric], 's--', label=f'{dataset_name} Test')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Epochs')
        ax.legend()
        ax.grid(True)
    
    # 绘制按时间的指标
    for i, metric in enumerate(metrics):
        ax = axes[i, 1]
        for dataset_name, history in datasets.items():
            ax.plot(history['time'], history['train_' + metric], 'o-', label=f'{dataset_name} Train')
            ax.plot(history['time'], history['val_' + metric], 's--', label=f'{dataset_name} Test')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} vs Time')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_modcloth():
    logging.info("Training on ModCloth dataset...")
    
    # 加载数据
    data_processor = DataProcessor(
        modcloth_path=DATA_PATH['modcloth'],
        renttherunway_path=DATA_PATH['rtr']
    )
    modcloth_df, _ = data_processor.load_data()
    
    # 限制数据条数
    if len(modcloth_df) > MAX_SAMPLES:
        modcloth_df = modcloth_df.head(MAX_SAMPLES)
    
    # 预处理数据
    modcloth_df, num_users, num_items = data_processor.preprocess_data(modcloth_df, 'quality')
    X_train, X_test, y_train, y_test = data_processor.split_data(modcloth_df, 'quality')
    
    logging.info(f"Dataset info - Users: {num_users}, Items: {num_items}")
    logging.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # 配置参数
    config = Config()
    
    # 创建模型
    model = xDeepFM(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config.embedding_dim,
        dnn_hidden_units=config.dnn_hidden_units,
        cin_layer_size=config.cin_layer_size,
        dropout_rate=config.dropout_rate
    )
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # 创建回调
    custom_callback = CustomCallback((X_train, y_train), (X_test, y_test), "ModCloth")
    
    # 训练模型
    train_dataset = train_dataset.shuffle(10000).batch(config.batch_size)
    test_dataset = test_dataset.batch(config.batch_size)
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[custom_callback],
        verbose=1
    )
    
    # 评估模型
    y_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    logging.info("\nModCloth Final Results:")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    
    # 保存指标到CSV
    save_metrics_to_csv(custom_callback.history, "ModCloth")
    
    return model, custom_callback.history

def train_renttherunway():
    logging.info("\nTraining on Rent The Runway dataset...")
    
    # 加载数据
    data_processor = DataProcessor(
        modcloth_path=DATA_PATH['modcloth'],
        renttherunway_path=DATA_PATH['rtr']
    )
    _, rtr_df = data_processor.load_data()
    
    # 限制数据条数
    if len(rtr_df) > MAX_SAMPLES:
        rtr_df = rtr_df.head(MAX_SAMPLES)
    
    # 预处理数据
    rtr_df, num_users, num_items = data_processor.preprocess_data(rtr_df, 'rating')
    X_train, X_test, y_train, y_test = data_processor.split_data(rtr_df, 'rating')
    
    logging.info(f"Dataset info - Users: {num_users}, Items: {num_items}")
    logging.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # 配置参数
    config = Config()
    
    # 创建模型
    model = xDeepFM(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config.embedding_dim,
        dnn_hidden_units=config.dnn_hidden_units,
        cin_layer_size=config.cin_layer_size,
        dropout_rate=config.dropout_rate
    )
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # 创建回调
    custom_callback = CustomCallback((X_train, y_train), (X_test, y_test), "Rent The Runway")
    
    # 训练模型
    train_dataset = train_dataset.shuffle(10000).batch(config.batch_size)
    test_dataset = test_dataset.batch(config.batch_size)
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[custom_callback],
        verbose=1
    )
    
    # 评估模型
    y_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    logging.info("\nRent The Runway Final Results:")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    
    # 保存指标到CSV
    save_metrics_to_csv(custom_callback.history, "Rent_The_Runway")
    
    return model, custom_callback.history

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("results", exist_ok=True)
    
    try:
        modcloth_model = modcloth_history = None
        rtr_model = rtr_history = None
        
        # 根据选择训练相应的数据集
        if DATASET_CHOICE in ['modcloth', 'both']:
            modcloth_model, modcloth_history = train_modcloth()
            modcloth_model.save_weights("results/modcloth_xdeepfm_tf.weights.h5")
            
        if DATASET_CHOICE in ['rtr', 'both']:
            rtr_model, rtr_history = train_renttherunway()
            rtr_model.save_weights("results/rtr_xdeepfm_tf.weights.h5")
        
        # 绘制并保存训练指标图
        plot_metrics(modcloth_history, rtr_history)
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
