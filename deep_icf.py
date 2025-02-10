import tensorflow as tf
import numpy as np

class DeepICF(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_layers=[64, 32, 16]):
        super(DeepICF, self).__init__()
        
        # 嵌入层
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_dim, name='user_embedding'
        )
        self.item_embedding = tf.keras.layers.Embedding(
            num_items, embedding_dim, name='item_embedding'
        )
        
        # 多层感知机层
        self.mlp_layers = []
        for units in mlp_layers:
            self.mlp_layers.append(tf.keras.layers.Dense(
                units, activation='relu'
            ))
        
        # 输出层
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        user_input, item_input, history_items = inputs
        
        # 转换输入为张量
        user_input = tf.cast(user_input, tf.int32)
        item_input = tf.cast(item_input, tf.int32)
        history_items = tf.cast(history_items, tf.int32)
        
        # 获取嵌入
        user_emb = self.user_embedding(user_input)
        item_emb = self.item_embedding(item_input)
        history_emb = self.item_embedding(history_items)
        
        # 计算目标物品与历史物品的相似度
        item_emb_expanded = tf.expand_dims(item_emb, axis=1)  # [batch_size, 1, embedding_dim]
        similarity = tf.reduce_sum(
            tf.multiply(item_emb_expanded, history_emb), axis=2
        )  # [batch_size, max_history]
        
        # 注意力权重
        attention_weights = tf.nn.softmax(similarity, axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=2)
        
        # 加权历史物品表示
        weighted_history = tf.reduce_sum(
            tf.multiply(history_emb, attention_weights), axis=1
        )
        
        # 连接特征
        concat_features = tf.concat([user_emb, item_emb, weighted_history], axis=1)
        
        # 多层感知机
        x = concat_features
        for layer in self.mlp_layers:
            x = layer(x)
        
        # 输出预测
        output = self.output_layer(x)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_users': self.user_embedding.input_dim,
            'num_items': self.item_embedding.input_dim,
            'embedding_dim': self.user_embedding.output_dim,
            'mlp_layers': [layer.units for layer in self.mlp_layers]
        })
        return config 