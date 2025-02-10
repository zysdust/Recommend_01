import json
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from DeepCoNN import DeepCoNN

# 全局配置
NUM_EPOCHS = 60  # 训练轮数
DATASET_CHOICE = 'modcloth'  # 可选: 'modcloth', 'renttherunway', 'both'
BATCH_SIZE = 32
MAX_SAMPLES = 10000  # 指定使用的最大数据条数，如果为None则使用全部数据
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 数据集路径配置
DATASET_PATHS = {
    'mini': {
        'modcloth': 'Data_mini/modcloth_final_data_mini.json',
        'renttherunway': 'Data_mini/renttherunway_final_data_mini.json'
    },
    'full': {
        'modcloth': 'Data_full/modcloth_final_data_processed.json',
        'renttherunway': 'Data_full/renttherunway_final_data_processed.json'
    }
}
DATASET_VERSION = 'full'  # 可选: 'mini', 'full'

def safe_text(text):
    """安全地处理文本，确保返回有效的字符串"""
    if text is None or not isinstance(text, str):
        return ''
    return text.strip()

# 加载数据
def load_data(modcloth_path, renttherunway_path):
    modcloth_reviews = []
    renttherunway_reviews = []
    
    try:
        # 加载ModCloth数据
        if DATASET_CHOICE in ['modcloth', 'both']:
            print(f"尝试加载ModCloth数据: {modcloth_path}")
            with open(modcloth_path, 'r', encoding='utf-8') as f:
                for line in f:
                    review = json.loads(line)
                    # ModCloth数据集中使用quality作为评分
                    if 'quality' in review:
                        review_text = safe_text(review.get('review_text', ''))
                        if not review_text:  # 如果文本为空，跳过该条记录
                            continue
                        modcloth_reviews.append({
                            'user_id': str(review['user_id']),
                            'item_id': str(review['item_id']),
                            'rating': float(review['quality']) / 5.0,
                            'review_text': review_text,
                            'dataset': 'modcloth'
                        })
        
        # 加载Rent the Runway数据
        if DATASET_CHOICE in ['renttherunway', 'both']:
            print(f"尝试加载Rent the Runway数据: {renttherunway_path}")
            with open(renttherunway_path, 'r', encoding='utf-8') as f:
                for line in f:
                    review = json.loads(line)
                    if 'review_text' in review and 'rating' in review:
                        review_text = safe_text(review['review_text'])
                        if not review_text:  # 如果文本为空，跳过该条记录
                            continue
                        renttherunway_reviews.append({
                            'user_id': str(review['user_id']),
                            'item_id': str(review['item_id']),
                            'rating': float(review['rating']) / 10.0,
                            'review_text': review_text,
                            'dataset': 'renttherunway'
                        })
    except FileNotFoundError as e:
        print(f"错误：找不到数据文件 - {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"错误：JSON解析失败 - {e}")
        raise
    except Exception as e:
        print(f"错误：加载数据时发生未知错误 - {e}")
        raise
    
    # 根据选择的数据集返回
    if DATASET_CHOICE == 'modcloth':
        reviews = modcloth_reviews
    elif DATASET_CHOICE == 'renttherunway':
        reviews = renttherunway_reviews
    else:
        reviews = modcloth_reviews + renttherunway_reviews
    
    print(f"ModCloth数据条数: {len(modcloth_reviews)}")
    print(f"Rent the Runway数据条数: {len(renttherunway_reviews)}")
    
    # 过滤掉没有评分的数据
    valid_reviews = [review for review in reviews if review['rating'] is not None]
    print(f"有效数据条数: {len(valid_reviews)}")
    
    if len(valid_reviews) == 0:
        raise ValueError("没有找到有效的评论数据！请检查数据文件路径和数据格式。")
    
    # 限制数据条数
    if MAX_SAMPLES is not None and MAX_SAMPLES > 0:
        valid_reviews = valid_reviews[:min(MAX_SAMPLES, len(valid_reviews))]
        print(f"使用数据条数: {len(valid_reviews)}")
    
    return valid_reviews

# 预处理文本数据
def preprocess_data(reviews, max_text_length=200):
    try:
        # 提取所有评论文本
        texts = [review['review_text'] for review in reviews]
        
        # 创建tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        
        # 转换文本为序列
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_text_length, padding='post')
        
        # 获取评分
        ratings = np.array([review['rating'] for review in reviews])
        
        # 创建用户和商品的评论映射
        user_reviews = {}
        item_reviews = {}
        
        for i, review in enumerate(reviews):
            user_id = review['user_id']
            item_id = review['item_id']
            
            if user_id not in user_reviews:
                user_reviews[user_id] = []
            user_reviews[user_id].append(padded_sequences[i])
            
            if item_id not in item_reviews:
                item_reviews[item_id] = []
            item_reviews[item_id].append(padded_sequences[i])
        
        # 对每个用户和商品的评论取平均
        user_sequences = []
        item_sequences = []
        final_ratings = []
        
        for i, review in enumerate(reviews):
            user_id = review['user_id']
            item_id = review['item_id']
            
            user_seq = np.mean(user_reviews[user_id], axis=0)
            item_seq = np.mean(item_reviews[item_id], axis=0)
            
            user_sequences.append(user_seq)
            item_sequences.append(item_seq)
            final_ratings.append(ratings[i])
        
        return np.array(user_sequences), np.array(item_sequences), np.array(final_ratings), len(tokenizer.word_index) + 1
    except Exception as e:
        print(f"预处理数据时发生错误: {e}")
        raise

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_times = []
        self.train_losses = []  # 添加训练Loss记录
        self.train_maes = []
        self.train_mses = []
        self.train_rmses = []
        self.val_losses = []    # 添加验证Loss记录
        self.val_maes = []
        self.val_mses = []
        self.val_rmses = []
        self.start_time = None
        self.epochs = []
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time() - self.start_time
        self.train_times.append(current_time)
        self.epochs.append(epoch + 1)
        
        # 训练集指标
        self.train_losses.append(logs['loss'])  # 记录训练Loss
        self.train_maes.append(logs['mae'])
        self.train_mses.append(logs['loss'])
        self.train_rmses.append(np.sqrt(logs['loss']))
        
        # 验证集指标
        self.val_losses.append(logs['val_loss'])  # 记录验证Loss
        self.val_maes.append(logs['val_mae'])
        self.val_mses.append(logs['val_loss'])
        self.val_rmses.append(np.sqrt(logs['val_loss']))
        
    def save_to_csv(self, save_dir='results'):
        # 创建DataFrame保存训练过程中的指标
        df = pd.DataFrame({
            'Epoch': self.epochs,
            'Time': self.train_times,
            'Train_Loss': self.train_losses,  # 添加训练Loss
            'Train_MAE': self.train_maes,
            'Train_MSE': self.train_mses,
            'Train_RMSE': self.train_rmses,
            'Val_Loss': self.val_losses,      # 添加验证Loss
            'Val_MAE': self.val_maes,
            'Val_MSE': self.val_mses,
            'Val_RMSE': self.val_rmses
        })
        # 添加实验配置信息
        df['Dataset'] = DATASET_CHOICE
        df['Dataset_Version'] = DATASET_VERSION
        df['Max_Samples'] = MAX_SAMPLES
        
        df.to_csv(f'{save_dir}/training_metrics_{DATASET_CHOICE}_{DATASET_VERSION}.csv', index=False)

def plot_metrics(metrics_callback, save_dir='results'):
    # 创建图表
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'模型训练过程中的评价指标变化 (Dataset: {DATASET_CHOICE})', fontsize=16)
    
    # 绘制随训练轮数的变化
    epochs = range(1, len(metrics_callback.train_maes) + 1)
    
    # MAE
    axes[0,0].plot(epochs, metrics_callback.train_maes, 'b-', label='训练集')
    axes[0,0].plot(epochs, metrics_callback.val_maes, 'r-', label='验证集')
    axes[0,0].set_title('MAE vs Epochs')
    axes[0,0].set_xlabel('Epochs')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].legend()
    
    # MSE
    axes[0,1].plot(epochs, metrics_callback.train_mses, 'b-', label='训练集')
    axes[0,1].plot(epochs, metrics_callback.val_mses, 'r-', label='验证集')
    axes[0,1].set_title('MSE vs Epochs')
    axes[0,1].set_xlabel('Epochs')
    axes[0,1].set_ylabel('MSE')
    axes[0,1].legend()
    
    # RMSE
    axes[0,2].plot(epochs, metrics_callback.train_rmses, 'b-', label='训练集')
    axes[0,2].plot(epochs, metrics_callback.val_rmses, 'r-', label='验证集')
    axes[0,2].set_title('RMSE vs Epochs')
    axes[0,2].set_xlabel('Epochs')
    axes[0,2].set_ylabel('RMSE')
    axes[0,2].legend()
    
    # 绘制随时间的变化
    # MAE
    axes[1,0].plot(metrics_callback.train_times, metrics_callback.train_maes, 'b-', label='训练集')
    axes[1,0].plot(metrics_callback.train_times, metrics_callback.val_maes, 'r-', label='验证集')
    axes[1,0].set_title('MAE vs Time')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('MAE')
    axes[1,0].legend()
    
    # MSE
    axes[1,1].plot(metrics_callback.train_times, metrics_callback.train_mses, 'b-', label='训练集')
    axes[1,1].plot(metrics_callback.train_times, metrics_callback.val_mses, 'r-', label='验证集')
    axes[1,1].set_title('MSE vs Time')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('MSE')
    axes[1,1].legend()
    
    # RMSE
    axes[1,2].plot(metrics_callback.train_times, metrics_callback.train_rmses, 'b-', label='训练集')
    axes[1,2].plot(metrics_callback.train_times, metrics_callback.val_rmses, 'r-', label='验证集')
    axes[1,2].set_title('RMSE vs Time')
    axes[1,2].set_xlabel('Time (s)')
    axes[1,2].set_ylabel('RMSE')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_plot_{DATASET_CHOICE}.png')
    plt.close()

def main():
    print(f"使用数据集: {DATASET_CHOICE}")
    print(f"数据集版本: {DATASET_VERSION}")
    print(f"训练轮数: {NUM_EPOCHS}")
    
    print("开始加载数据...")
    # 根据配置加载相应的数据集
    modcloth_path = DATASET_PATHS[DATASET_VERSION]['modcloth']
    renttherunway_path = DATASET_PATHS[DATASET_VERSION]['renttherunway']
    reviews = load_data(modcloth_path, renttherunway_path)
    
    if len(reviews) == 0:
        print("错误：没有加载到任何数据！")
        return
    
    print(f"总共加载了 {len(reviews)} 条评论")
    
    print("开始预处理数据...")
    # 预处理数据
    user_sequences, item_sequences, ratings, vocab_size = preprocess_data(reviews)
    
    if len(ratings) == 0:
        print("错误：预处理后没有有效数据！")
        return
    
    print(f"词汇表大小: {vocab_size}")
    print(f"用户序列形状: {user_sequences.shape}")
    print(f"商品序列形状: {item_sequences.shape}")
    print(f"评分形状: {ratings.shape}")
    
    # 划分训练集和测试集
    user_train, user_test, item_train, item_test, ratings_train, ratings_test = train_test_split(
        user_sequences, item_sequences, ratings, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    print("\n开始训练模型...")
    # 创建并训练模型
    model = DeepCoNN(vocab_size=vocab_size)
    
    # 创建回调函数
    metrics_callback = MetricsCallback()
    
    history = model.fit(
        x={'user_input': user_train, 'item_input': item_train},
        y=ratings_train,
        validation_data=({'user_input': user_test, 'item_input': item_test}, ratings_test),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[metrics_callback]
    )
    
    # 评估模型
    test_loss = model.model.evaluate(
        {'user_input': user_test, 'item_input': item_test},
        ratings_test,
        verbose=1
    )
    print(f'\nTest Loss: {test_loss[0]:.4f}')
    print(f'Test MAE: {test_loss[1]:.4f}')
    
    # 计算并打印最终的RMSE
    y_pred = model.predict({'user_input': user_test, 'item_input': item_test})
    final_rmse = np.sqrt(mean_squared_error(ratings_test, y_pred))
    print(f'Test RMSE: {final_rmse:.4f}')
    
    # 绘制评价指标图
    plot_metrics(metrics_callback)
    
    # 保存训练过程中的指标到CSV
    metrics_callback.save_to_csv()
    
    # 保存最终的评价指标
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'RMSE'],
        'Value': [test_loss[0], test_loss[1], final_rmse],
        'Dataset': DATASET_CHOICE
    })
    results_df.to_csv(f'results/final_metrics_{DATASET_CHOICE}.csv', index=False)

if __name__ == '__main__':
    # 创建结果目录
    import os
    os.makedirs('results', exist_ok=True)
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {e}") 