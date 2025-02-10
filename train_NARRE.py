import tensorflow as tf
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from NARRE import NARRE

# 全局参数设置
DATASET_NAME = 'modcloth'  # 可选: 'modcloth' 或 'renttherunway'

# 数据集路径配置
DATASET_CONFIG = {
    'modcloth': {
        'path': 'Data_full',
        'filename': 'modcloth_final_data_processed.json'
    },
    'renttherunway': {
        'path': 'Data_full',
        'filename': 'renttherunway_final_data_processed.json'
    }
}

# 其他参数设置
MAX_SAMPLES = 10000  # 使用的最大数据量，如果大于数据集大小则使用全部数据
EPOCHS = 60

# 模型参数设置
MAX_NUM_WORDS = 50000  # 词汇表大小
MAX_SEQUENCE_LENGTH = 100  # 评论最大长度
EMBEDDING_DIM = 100  # 词嵌入维度
NUM_FACTORS = 50  # 隐因子维度
BATCH_SIZE = 64

def get_dataset_path():
    """获取当前选择的数据集完整路径"""
    config = DATASET_CONFIG[DATASET_NAME]
    return os.path.join(config['path'], config['filename'])

def save_metrics_to_csv(train_metrics, test_metrics, train_loss_history, test_loss_history, times, save_dir):
    """将训练指标保存为CSV文件"""
    metrics_names = ['MSE', 'RMSE', 'MAE']
    epochs = range(1, len(train_metrics[0]) + 1)
    
    # 创建结果字典
    results = {
        'Epoch': epochs,
        'Time': times,
        'Train_Loss': train_loss_history,
        'Test_Loss': test_loss_history
    }
    
    # 添加训练集和测试集的指标
    for i, metric in enumerate(metrics_names):
        results[f'Train_{metric}'] = train_metrics[i]
        results[f'Test_{metric}'] = test_metrics[i]
    
    # 创建DataFrame并保存
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, f'{DATASET_NAME}_metrics.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

def compute_metrics(y_true, y_pred):
    """计算MSE、RMSE和MAE评估指标"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred), tf.float32)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mse, rmse, mae

def plot_metrics(train_metrics, test_metrics, times, save_dir):
    """绘制评估指标随训练轮数和时间的变化图"""
    metrics_names = ['MSE', 'RMSE', 'MAE']
    epochs = range(1, len(train_metrics[0]) + 1)
    
    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'NARRE模型训练结果 ({DATASET_NAME}数据集)', fontsize=16)
    
    # 绘制随训练轮数的变化
    for i, metric_name in enumerate(metrics_names):
        ax = axes[0, i]
        ax.plot(epochs, train_metrics[i], 'b-', label=f'训练集{metric_name}')
        ax.plot(epochs, test_metrics[i], 'r-', label=f'测试集{metric_name}')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}随训练轮数的变化')
        ax.legend()
        ax.grid(True)
    
    # 绘制随训练时间的变化
    for i, metric_name in enumerate(metrics_names):
        ax = axes[1, i]
        ax.plot(times, train_metrics[i], 'b-', label=f'训练集{metric_name}')
        ax.plot(times, test_metrics[i], 'r-', label=f'测试集{metric_name}')
        ax.set_xlabel('训练时间(秒)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}随训练时间的变化')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{DATASET_NAME}_training_metrics.png'))
    plt.close()

def load_data(data_path):
    """加载指定的数据集"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 如果指定了最大数据量，则限制数据量
    if MAX_SAMPLES > 0:
        data = data[:min(MAX_SAMPLES, len(data))]
    
    # 提取用户ID、商品ID和评论
    user_ids = []
    item_ids = []
    reviews = []
    ratings = []
    
    # 数据清洗
    for d in data:
        # 获取评论文本
        review_text = d.get('review_text')
        if review_text is None:  # 如果review_text是None
            review_text = ''
        review_text = str(review_text).strip()  # 转换为字符串并去除空白
        if not review_text:  # 如果评论为空字符串
            review_text = "no review"
        
        # 根据数据集选择评分字段
        if DATASET_NAME == 'modcloth':
            rating = d.get('quality')  # modcloth数据集使用quality作为评分
        else:  # renttherunway
            rating = d.get('rating')   # renttherunway数据集使用rating作为评分
            
        try:
            rating = float(rating)
            if rating <= 0:  # 确保评分为正数
                continue
        except (ValueError, TypeError):
            continue  # 跳过无效评分的数据
            
        # 确保user_id和item_id存在
        if 'user_id' not in d or 'item_id' not in d:
            continue
            
        user_ids.append(d['user_id'])
        item_ids.append(d['item_id'])
        reviews.append(review_text)
        ratings.append(rating)
    
    print(f"原始数据量: {len(data)}")
    print(f"清洗后数据量: {len(ratings)}")
    print(f"使用评分字段: {'quality' if DATASET_NAME == 'modcloth' else 'rating'}")
    
    if len(ratings) == 0:
        raise ValueError(f"数据集 {data_path} 清洗后没有有效数据!")
    
    # 编码用户ID和商品ID
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    user_ids_encoded = user_encoder.fit_transform(user_ids)
    item_ids_encoded = item_encoder.fit_transform(item_ids)
    
    # 处理文本评论
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(reviews)
    
    reviews_seq = tokenizer.texts_to_sequences(reviews)
    reviews_padded = pad_sequences(reviews_seq, maxlen=MAX_SEQUENCE_LENGTH)
    
    return (user_ids_encoded, item_ids_encoded, reviews_padded, np.array(ratings, dtype=np.float32),
            len(user_encoder.classes_), len(item_encoder.classes_), len(tokenizer.word_index) + 1)

def main():
    # 获取数据集路径
    data_path = get_dataset_path()
    print(f"\n使用数据集: {data_path}")
    
    print(f"正在加载{DATASET_NAME}数据集...")
    (user_ids, item_ids, reviews, ratings,
     num_users, num_items, vocab_size) = load_data(data_path)
    
    print(f"数据集统计:")
    print(f"用户数: {num_users}")
    print(f"商品数: {num_items}")
    print(f"词汇表大小: {vocab_size}")
    print(f"使用数据量: {len(ratings)}")
    
    # 创建保存结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 初始化指标记录列表
    train_mse_history = []
    train_rmse_history = []
    train_mae_history = []
    test_mse_history = []
    test_rmse_history = []
    test_mae_history = []
    train_loss_history = []  # 添加训练Loss记录
    test_loss_history = []   # 添加测试Loss记录
    training_times = []
    start_time = time.time()
    
    # 划分训练集和测试集
    indices = np.arange(len(ratings))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # 创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (user_ids[train_indices], 
         item_ids[train_indices],
         reviews[train_indices],
         reviews[train_indices]),
        ratings[train_indices]
    )).shuffle(10000).batch(BATCH_SIZE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        (user_ids[test_indices],
         item_ids[test_indices],
         reviews[test_indices],
         reviews[test_indices]),
        ratings[test_indices]
    )).batch(BATCH_SIZE)
    
    # 创建模型
    print("\n构建NARRE模型...")
    model = NARRE(
        num_users=num_users,
        num_items=num_items,
        num_factors=NUM_FACTORS,
        vocab_size=vocab_size,
        embedding_size=EMBEDDING_DIM,
        max_review_length=MAX_SEQUENCE_LENGTH
    )
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        # 训练
        train_loss = 0
        train_mse = 0
        train_rmse = 0
        train_mae = 0
        train_steps = 0
        
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                predictions, _, _ = model(x)
                loss = loss_fn(y, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # 计算评估指标
            mse, rmse, mae = compute_metrics(y, predictions)
            train_loss += loss.numpy()
            train_mse += mse.numpy()
            train_rmse += rmse.numpy()
            train_mae += mae.numpy()
            train_steps += 1
        
        train_loss /= train_steps
        train_mse /= train_steps
        train_rmse /= train_steps
        train_mae /= train_steps
        
        # 评估
        test_loss = 0
        test_mse = 0
        test_rmse = 0
        test_mae = 0
        test_steps = 0
        
        for x, y in test_dataset:
            predictions, _, _ = model(x)
            loss = loss_fn(y, predictions)
            mse, rmse, mae = compute_metrics(y, predictions)
            
            test_loss += loss.numpy()
            test_mse += mse.numpy()
            test_rmse += rmse.numpy()
            test_mae += mae.numpy()
            test_steps += 1
        
        test_loss /= test_steps
        test_mse /= test_steps
        test_rmse /= test_steps
        test_mae /= test_steps
        
        # 记录指标
        train_mse_history.append(train_mse)
        train_rmse_history.append(train_rmse)
        train_mae_history.append(train_mae)
        test_mse_history.append(test_mse)
        test_rmse_history.append(test_rmse)
        test_mae_history.append(test_mae)
        train_loss_history.append(train_loss)  # 记录训练Loss
        test_loss_history.append(test_loss)    # 记录测试Loss
        training_times.append(time.time() - start_time)
        
        print(f'训练集 - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}')
        print(f'测试集 - Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}')
    
    # 保存模型
    print("\n保存模型...")
    model.save_weights(f'results/{DATASET_NAME}_model_weights.weights.h5')
    print(f"模型已保存到 results/{DATASET_NAME}_model_weights.weights.h5")
    
    # 保存评估指标到CSV
    print("\n保存评估指标到CSV...")
    train_metrics = [train_mse_history, train_rmse_history, train_mae_history]
    test_metrics = [test_mse_history, test_rmse_history, test_mae_history]
    csv_path = save_metrics_to_csv(train_metrics, test_metrics, train_loss_history, test_loss_history, training_times, 'results')
    print(f"评估指标已保存到 {csv_path}")
    
    # 绘制并保存评估指标图
    print("\n绘制评估指标图...")
    plot_metrics(train_metrics, test_metrics, training_times, 'results')
    print(f"评估指标图已保存到 results/{DATASET_NAME}_training_metrics.png")

if __name__ == '__main__':
    main() 