import tensorflow as tf
import numpy as np
import pandas as pd
from data_processor_DeepICF import DataProcessor
from deep_icf import DeepICF
from metrics_DeepICF import MetricsManager

# 全局配置
CONFIG = {
    'num_epochs': 60,  # 训练轮数
    'dataset': 'modcloth',  # 'modcloth' 或 'renttherunway'
    'data_paths': {
        'modcloth': 'Data_full/modcloth_final_data_processed.json',
        'renttherunway': 'Data_full/renttherunway_final_data_processed.json'
    },
    'max_samples': 10000,  # 使用数据集的最大样本数，float('inf')表示使用全部数据
    'batch_size': 64,
    'learning_rate': 0.001
}

def prepare_batch_data(X, y, user_history, batch_size, max_history=50):
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        
        # 准备历史交互数据
        history_items = []
        for user_idx in batch_X[:, 0]:
            history = user_history.get(user_idx, [])
            # 填充或截断历史记录
            if len(history) > max_history:
                history = history[:max_history]
            else:
                history = history + [0] * (max_history - len(history))
            history_items.append(history)
        
        history_items = np.array(history_items)
        yield [batch_X[:, 0], batch_X[:, 1], history_items], batch_y

def save_metrics_to_csv(metrics_manager, save_path):
    # 准备数据
    data = {
        'epoch': list(range(1, len(metrics_manager.train_metrics['loss']) + 1)),
        'train_loss': metrics_manager.train_metrics['loss'],
        'train_mae': metrics_manager.train_metrics['mae'],
        'train_rmse': metrics_manager.train_metrics['rmse'],
        'train_mse': metrics_manager.train_metrics['mse'],
        'train_time': metrics_manager.train_metrics['time']
    }
    
    # 如果有测试集数据，添加到数据字典中
    if len(metrics_manager.test_metrics['loss']) > 0:
        test_epochs = np.linspace(1, len(metrics_manager.train_metrics['loss']), 
                                len(metrics_manager.test_metrics['loss']))
        data.update({
            'test_loss': metrics_manager.test_metrics['loss'],
            'test_mae': metrics_manager.test_metrics['mae'],
            'test_rmse': metrics_manager.test_metrics['rmse'],
            'test_mse': metrics_manager.test_metrics['mse'],
            'test_time': metrics_manager.test_metrics['time']
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

def train_model():
    # 数据处理
    data_processor = DataProcessor(
        CONFIG['data_paths']['modcloth'],
        CONFIG['data_paths']['renttherunway']
    )
    
    # 根据配置选择数据集
    if CONFIG['dataset'] == 'modcloth':
        df, _ = data_processor.load_data()
        rating_col = 'quality'
    else:
        _, df = data_processor.load_data()
        rating_col = 'rating'
    
    # 限制数据样本数
    if CONFIG['max_samples'] < len(df):
        df = df.head(int(CONFIG['max_samples']))
    
    # 预处理数据
    df, num_users, num_items = data_processor.preprocess_data(df, rating_col)
    X_train, X_test, y_train, y_test = data_processor.split_data(df, rating_col)
    user_history = data_processor.get_item_history(df)
    
    # 创建模型和评价指标管理器
    model = DeepICF(num_users, num_items)
    metrics_manager = MetricsManager(CONFIG['dataset'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # 训练循环
    for epoch in range(CONFIG['num_epochs']):
        epoch_predictions = []
        epoch_labels = []
        total_loss = 0
        num_batches = 0
        
        # 训练
        for batch_inputs, batch_labels in prepare_batch_data(X_train, y_train, user_history, CONFIG['batch_size']):
            with tf.GradientTape() as tape:
                predictions = model(batch_inputs)
                batch_labels = tf.cast(batch_labels, tf.float32)
                loss = loss_fn(batch_labels, tf.squeeze(predictions))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # 收集预测结果
            epoch_predictions.extend(tf.squeeze(predictions).numpy())
            epoch_labels.extend(batch_labels.numpy())
            total_loss += loss.numpy()
            num_batches += 1
        
        # 计算训练集指标
        avg_loss = total_loss / num_batches
        metrics_manager.update_metrics(
            np.array(epoch_labels),
            np.array(epoch_predictions),
            avg_loss,
            is_train=True
        )
        
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')
        
        # 评估测试集
        test_predictions = []
        test_labels = []
        test_loss = 0
        test_batches = 0
        
        for batch_inputs, batch_labels in prepare_batch_data(X_test, y_test, user_history, CONFIG['batch_size']):
            predictions = model(batch_inputs)
            batch_labels = tf.cast(batch_labels, tf.float32)
            test_loss += loss_fn(batch_labels, tf.squeeze(predictions)).numpy()
            
            # 收集预测结果
            test_predictions.extend(tf.squeeze(predictions).numpy())
            test_labels.extend(batch_labels.numpy())
            test_batches += 1
        
        # 计算测试集指标
        avg_test_loss = test_loss / test_batches
        metrics_manager.update_metrics(
            np.array(test_labels),
            np.array(test_predictions),
            avg_test_loss,
            is_train=False
        )
        
        print(f'Test Loss: {avg_test_loss:.4f}')
    
    # 保存评价指标到CSV文件
    save_metrics_to_csv(metrics_manager, f'results/{CONFIG["dataset"]}_metrics.csv')
    
    # 绘制评价指标图
    metrics_manager.plot_metrics()
    return model

if __name__ == '__main__':
    print(f"Training on {CONFIG['dataset']} dataset...")
    print(f"Using up to {CONFIG['max_samples']} samples")
    print(f"Training for {CONFIG['num_epochs']} epochs")
    model = train_model() 