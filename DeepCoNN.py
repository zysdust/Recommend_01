import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model

class DeepCoNN:
    def __init__(self, 
                 vocab_size,
                 embedding_dim=300,
                 max_text_length=200,
                 num_filters=100,
                 filter_sizes=[3,4,5],
                 dropout_rate=0.5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_text_length = max_text_length
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout_rate
        
        self.model = self._build_model()
        
    def _build_model(self):
        # 用户评论输入
        user_input = Input(shape=(self.max_text_length,), name='user_input')
        # 商品评论输入
        item_input = Input(shape=(self.max_text_length,), name='item_input')
        
        # 共享的词嵌入层
        embedding_layer = Embedding(self.vocab_size,
                                  self.embedding_dim,
                                  input_length=self.max_text_length)
        
        # 用户评论的词嵌入
        user_embedding = embedding_layer(user_input)
        # 商品评论的词嵌入
        item_embedding = embedding_layer(item_input)
        
        # 用户评论的CNN层
        user_conv_layers = []
        for filter_size in self.filter_sizes:
            conv = Conv1D(filters=self.num_filters,
                         kernel_size=filter_size,
                         activation='relu')(user_embedding)
            pool = MaxPooling1D(pool_size=self.max_text_length - filter_size + 1)(conv)
            user_conv_layers.append(pool)
        
        # 商品评论的CNN层
        item_conv_layers = []
        for filter_size in self.filter_sizes:
            conv = Conv1D(filters=self.num_filters,
                         kernel_size=filter_size,
                         activation='relu')(item_embedding)
            pool = MaxPooling1D(pool_size=self.max_text_length - filter_size + 1)(conv)
            item_conv_layers.append(pool)
        
        # 合并用户的CNN特征
        user_concat = Concatenate(axis=1)(user_conv_layers)
        user_flat = Flatten()(user_concat)
        user_vector = Dense(100, activation='relu')(user_flat)
        user_vector = Dropout(self.dropout_rate)(user_vector)
        
        # 合并商品的CNN特征
        item_concat = Concatenate(axis=1)(item_conv_layers)
        item_flat = Flatten()(item_concat)
        item_vector = Dense(100, activation='relu')(item_flat)
        item_vector = Dropout(self.dropout_rate)(item_vector)
        
        # 特征交互层
        concat_vector = Concatenate()([user_vector, item_vector])
        
        # 全连接层
        fc1 = Dense(64, activation='relu')(concat_vector)
        fc1 = Dropout(self.dropout_rate)(fc1)
        fc2 = Dense(32, activation='relu')(fc1)
        
        # 输出层
        output = Dense(1, activation='sigmoid')(fc2)
        
        model = Model(inputs={'user_input': user_input, 'item_input': item_input},
                     outputs=output)
        
        model.compile(optimizer='adam',
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def fit(self, x, y, **kwargs):
        return self.model.fit(x=x, y=y, **kwargs)
    
    def predict(self, x):
        return self.model.predict(x)
