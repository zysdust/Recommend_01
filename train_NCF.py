import tensorflow as tf
import numpy as np
import pandas as pd
import json
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from NCF import NCF

# 全局配置参数
NUM_EPOCHS = 60  # 训练轮数
DATASET_NAME = 'modcloth'  # 可选: 'modcloth' 或 'renttherunway'
MAX_SAMPLES = 10000  # 使用的最大数据条数，如果为None则使用全部数据
RESULTS_DIR = 'results'  # 结果保存目录

# 数据集路径配置
DATASET_CONFIG = {
    'modcloth': {
        'data_dir': 'Data_full',
        'data_file': 'modcloth_final_data_processed.json',
        'mini_file': 'modcloth_final_data_mini.json'  # 小规模测试用数据集
    },
    'renttherunway': {
        'data_dir': 'Data_full',
        'data_file': 'renttherunway_final_data_processed.json',
        'mini_file': 'renttherunway_final_data_mini.json'  # 小规模测试用数据集
    }
}

# 是否使用小规模数据集进行测试
USE_MINI_DATASET = False

# 创建结果保存目录
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def get_dataset_path(dataset_name):
    """获取数据集文件的完整路径"""
    config = DATASET_CONFIG[dataset_name]
    filename = config['mini_file'] if USE_MINI_DATASET else config['data_file']
    return os.path.join(config['data_dir'], filename)

def load_data(file_path, max_samples=None):
    """加载数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            data.append(json.loads(line))
    return pd.DataFrame(data)

def preprocess_data(df, dataset_name):
    """数据预处理"""
    # 编码用户和物品ID
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_id'] = user_encoder.fit_transform(df['user_id'])
    df['item_id'] = item_encoder.fit_transform(df['item_id'])
    
    # 根据数据集选择不同的评分标签
    if dataset_name == 'modcloth':
        # modcloth数据集使用quality (1-5)作为标签
        df['score'] = df['quality'].astype(float)
    else:
        # renttherunway数据集使用rating (1-10)作为标签
        df['score'] = df['rating'].astype(float)
        # 将10分制转换为5分制，保持与modcloth一致
        df['score'] = df['score'] * 0.5
    
    # 将评分标准化到[0,1]区间
    df['score'] = (df['score'] - 1) / 4  # (rating - min) / (max - min)
    
    return df, user_encoder, item_encoder

def create_ncf_model(num_users, num_items, embedding_size=16):
    """创建并编译NCF模型"""
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_size=embedding_size,
        mlp_layers=[64, 32, 16, 8],
        alpha=0.5
    )
    
    # 使用Adam优化器,添加学习率衰减
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # 使用MSE损失函数，适用于回归任务
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']  # 使用MAE和MSE作为评估指标
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict([X_test[:, 0], X_test[:, 1]])
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return metrics

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, test_data):
        super(MetricsCallback, self).__init__()
        # 确保训练集和测试集数据的完全隔离
        self.train_inputs = [train_data[0][0], train_data[0][1]]
        self.train_labels = train_data[1]
        self.test_inputs = [test_data[0][0], test_data[0][1]]
        self.test_labels = test_data[1]
        
        # 记录指标
        self.train_loss = []
        self.test_loss = []
        self.train_mae = []
        self.test_mae = []
        self.train_mse = []
        self.test_mse = []
        self.train_rmse = []
        self.test_rmse = []
        self.times = []
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 记录时间
        self.times.append(time.time() - self.start_time)
        
        # 计算训练集损失和指标
        train_metrics = self.model.evaluate(
            self.train_inputs,
            self.train_labels,
            verbose=0,
            batch_size=256
        )
        train_loss = train_metrics[0]
        self.train_loss.append(train_loss)
        
        # 计算测试集损失和指标
        test_metrics = self.model.evaluate(
            self.test_inputs,
            self.test_labels,
            verbose=0,
            batch_size=256
        )
        test_loss = test_metrics[0]
        self.test_loss.append(test_loss)
        
        # 计算训练集详细指标
        train_pred = self.model.predict(
            self.train_inputs,
            verbose=0,
            batch_size=256
        )
        train_mae = mean_absolute_error(self.train_labels, train_pred)
        train_mse = mean_squared_error(self.train_labels, train_pred)
        self.train_mae.append(train_mae)
        self.train_mse.append(train_mse)
        self.train_rmse.append(np.sqrt(train_mse))
        
        # 计算测试集详细指标
        test_pred = self.model.predict(
            self.test_inputs,
            verbose=0,
            batch_size=256
        )
        test_mae = mean_absolute_error(self.test_labels, test_pred)
        test_mse = mean_squared_error(self.test_labels, test_pred)
        self.test_mae.append(test_mae)
        self.test_mse.append(test_mse)
        self.test_rmse.append(np.sqrt(test_mse))
        
        # 打印当前epoch的指标
        print(f'\nEpoch {epoch + 1} - 评估指标:')
        print(f'训练集 - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {np.sqrt(train_mse):.4f}')
        print(f'测试集 - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {np.sqrt(test_mse):.4f}')

def save_metrics_to_csv(metrics_callback, dataset_name):
    """保存训练指标到CSV文件"""
    metrics_df = pd.DataFrame({
        'epoch': range(1, len(metrics_callback.train_loss) + 1),
        'time': metrics_callback.times,
        'train_loss': metrics_callback.train_loss,
        'test_loss': metrics_callback.test_loss,
        'train_mae': metrics_callback.train_mae,
        'test_mae': metrics_callback.test_mae,
        'train_mse': metrics_callback.train_mse,
        'test_mse': metrics_callback.test_mse,
        'train_rmse': metrics_callback.train_rmse,
        'test_rmse': metrics_callback.test_rmse
    })
    
    csv_path = os.path.join(RESULTS_DIR, f'{dataset_name}_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"指标已保存到 {csv_path}")

def plot_metrics(metrics_callback, dataset_name):
    epochs = range(1, len(metrics_callback.train_loss) + 1)
    
    # 创建4x2的子图
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle(f'训练指标 - {dataset_name}数据集', fontsize=16)
    
    # 损失 vs Epochs
    axes[0,0].plot(epochs, metrics_callback.train_loss, 'b-', label='训练')
    axes[0,0].plot(epochs, metrics_callback.test_loss, 'r-', label='测试')
    axes[0,0].set_title('损失 vs Epochs')
    axes[0,0].set_xlabel('Epochs')
    axes[0,0].set_ylabel('损失')
    axes[0,0].legend()
    
    # MAE vs Epochs
    axes[0,1].plot(epochs, metrics_callback.train_mae, 'b-', label='训练')
    axes[0,1].plot(epochs, metrics_callback.test_mae, 'r-', label='测试')
    axes[0,1].set_title('MAE vs Epochs')
    axes[0,1].set_xlabel('Epochs')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].legend()
    
    # MSE vs Epochs
    axes[1,0].plot(epochs, metrics_callback.train_mse, 'b-', label='训练')
    axes[1,0].plot(epochs, metrics_callback.test_mse, 'r-', label='测试')
    axes[1,0].set_title('MSE vs Epochs')
    axes[1,0].set_xlabel('Epochs')
    axes[1,0].set_ylabel('MSE')
    axes[1,0].legend()
    
    # RMSE vs Epochs
    axes[1,1].plot(epochs, metrics_callback.train_rmse, 'b-', label='训练')
    axes[1,1].plot(epochs, metrics_callback.test_rmse, 'r-', label='测试')
    axes[1,1].set_title('RMSE vs Epochs')
    axes[1,1].set_xlabel('Epochs')
    axes[1,1].set_ylabel('RMSE')
    axes[1,1].legend()
    
    # 损失 vs 时间
    axes[2,0].plot(metrics_callback.times, metrics_callback.train_loss, 'b-', label='训练')
    axes[2,0].plot(metrics_callback.times, metrics_callback.test_loss, 'r-', label='测试')
    axes[2,0].set_title('损失 vs 时间')
    axes[2,0].set_xlabel('时间 (秒)')
    axes[2,0].set_ylabel('损失')
    axes[2,0].legend()
    
    # MAE vs 时间
    axes[2,1].plot(metrics_callback.times, metrics_callback.train_mae, 'b-', label='训练')
    axes[2,1].plot(metrics_callback.times, metrics_callback.test_mae, 'r-', label='测试')
    axes[2,1].set_title('MAE vs 时间')
    axes[2,1].set_xlabel('时间 (秒)')
    axes[2,1].set_ylabel('MAE')
    axes[2,1].legend()
    
    # MSE vs 时间
    axes[3,0].plot(metrics_callback.times, metrics_callback.train_mse, 'b-', label='训练')
    axes[3,0].plot(metrics_callback.times, metrics_callback.test_mse, 'r-', label='测试')
    axes[3,0].set_title('MSE vs 时间')
    axes[3,0].set_xlabel('时间 (秒)')
    axes[3,0].set_ylabel('MSE')
    axes[3,0].legend()
    
    # RMSE vs 时间
    axes[3,1].plot(metrics_callback.times, metrics_callback.train_rmse, 'b-', label='训练')
    axes[3,1].plot(metrics_callback.times, metrics_callback.test_rmse, 'r-', label='测试')
    axes[3,1].set_title('RMSE vs 时间')
    axes[3,1].set_xlabel('时间 (秒)')
    axes[3,1].set_ylabel('RMSE')
    axes[3,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{dataset_name}_metrics.png'))
    plt.close()

def main():
    # 加载数据
    print(f"加载{DATASET_NAME}数据集...")
    dataset_path = get_dataset_path(DATASET_NAME)
    print(f"使用数据集文件: {dataset_path}")
    df = load_data(dataset_path, MAX_SAMPLES)
    
    print(f"使用数据条数: {len(df)}")
    
    # 预处理数据
    print("预处理数据...")
    df, user_encoder, item_encoder = preprocess_data(df, DATASET_NAME)
    
    # 准备训练数据
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    X = df[['user_id', 'item_id']].values
    y = df['score'].values  # 使用标准化后的评分作为标签
    
    # 划分训练集和测试集
    print("划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 创建模型
    print("创建模型...")
    model = create_ncf_model(num_users, num_items)
    
    # 设置回调
    checkpoint_path = os.path.join(RESULTS_DIR, f'{DATASET_NAME}_best_model.weights.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        ),
        MetricsCallback(
            train_data=((X_train[:,0], X_train[:,1]), y_train),
            test_data=((X_test[:,0], X_test[:,1]), y_test)
        )
    ]
    
    # 训练模型
    print("开始训练...")
    history = model.fit(
        [X_train[:,0], X_train[:,1]],
        y_train,
        batch_size=256,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存和绘制指标
    metrics_callback = callbacks[1]
    save_metrics_to_csv(metrics_callback, DATASET_NAME)
    plot_metrics(metrics_callback, DATASET_NAME)
    
    # 评估模型
    print("\n评估模型性能...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n测试集结果:")
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")
    
    # 在测试集上计算其他指标
    test_pred = model.predict([X_test[:,0], X_test[:,1]])
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")

if __name__ == '__main__':
    main() 