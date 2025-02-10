import tensorflow as tf
import numpy as np

class NARRE(tf.keras.Model):
    def __init__(self, num_users, num_items, num_factors, vocab_size, embedding_size, 
                 max_review_length, num_filters=100, kernel_size=3, dropout_rate=0.5):
        super(NARRE, self).__init__()
        
        # 基础嵌入层
        self.user_embedding = tf.keras.layers.Embedding(num_users, num_factors)
        self.item_embedding = tf.keras.layers.Embedding(num_items, num_factors)
        self.word_embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        
        # 评论处理层
        self.review_cnn = tf.keras.Sequential([
            tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        
        # 用户评论注意力网络
        self.user_attention = tf.keras.Sequential([
            tf.keras.layers.Dense(num_filters, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # 商品评论注意力网络
        self.item_attention = tf.keras.Sequential([
            tf.keras.layers.Dense(num_filters, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # 预测层
        self.prediction = tf.keras.Sequential([
            tf.keras.layers.Dense(num_factors, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1)
        ])
        
        self.num_factors = num_factors
        
    def call(self, inputs):
        user_ids, item_ids, user_reviews, item_reviews = inputs
        
        # 获取用户和商品的基础嵌入
        user_embed = self.user_embedding(user_ids)  # [batch_size, num_factors]
        item_embed = self.item_embedding(item_ids)  # [batch_size, num_factors]
        
        # 处理用户评论
        user_reviews_embed = self.word_embedding(user_reviews)  # [batch_size, max_len, embedding_size]
        user_reviews_features = self.review_cnn(user_reviews_embed)  # [batch_size, num_filters]
        
        # 处理商品评论
        item_reviews_embed = self.word_embedding(item_reviews)  # [batch_size, max_len, embedding_size]
        item_reviews_features = self.review_cnn(item_reviews_embed)  # [batch_size, num_filters]
        
        # 计算用户评论注意力权重
        user_attention_weights = self.user_attention(user_reviews_features)  # [batch_size, 1]
        user_reviews_repr = user_attention_weights * user_reviews_features  # [batch_size, num_filters]
        
        # 计算商品评论注意力权重
        item_attention_weights = self.item_attention(item_reviews_features)  # [batch_size, 1]
        item_reviews_repr = item_attention_weights * item_reviews_features  # [batch_size, num_filters]
        
        # 合并所有特征
        concat_features = tf.concat([user_embed, item_embed, user_reviews_repr, item_reviews_repr], axis=1)
        
        # 预测评分
        prediction = self.prediction(concat_features)
        
        return prediction, user_attention_weights, item_attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_factors': self.num_factors
        })
        return config
