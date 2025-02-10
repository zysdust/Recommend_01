import tensorflow as tf
import numpy as np

class NCF(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=8, mlp_layers=[64,32,16,8], alpha=0.5):
        super(NCF, self).__init__()
        
        # GMF部分的嵌入层
        self.gmf_user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size, name='gmf_user_embedding'
        )
        self.gmf_item_embedding = tf.keras.layers.Embedding(
            num_items, embedding_size, name='gmf_item_embedding'
        )
        
        # MLP部分的嵌入层
        self.mlp_user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size, name='mlp_user_embedding'
        )
        self.mlp_item_embedding = tf.keras.layers.Embedding(
            num_items, embedding_size, name='mlp_item_embedding'
        )
        
        # MLP层
        self.mlp_layers = []
        input_size = embedding_size * 2  # 连接后的用户和物品嵌入维度
        for i, units in enumerate(mlp_layers):
            self.mlp_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    name=f'mlp_layer_{i}'
                )
            )
            
        # NeuMF层
        self.alpha = alpha  # GMF和MLP的融合权重
        self.predict_layer = tf.keras.layers.Dense(
            1, 
            activation='linear',  # 改为线性激活函数用于回归
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name='prediction'
        )
        
        # Dropout层
        self.dropout = tf.keras.layers.Dropout(0.2)
        
    def call(self, inputs, training=False):
        user_input, item_input = inputs
        
        # GMF部分
        gmf_user_embed = self.gmf_user_embedding(user_input)
        gmf_item_embed = self.gmf_item_embedding(item_input)
        gmf_output = tf.multiply(gmf_user_embed, gmf_item_embed)  # 元素级乘法
        
        # MLP部分
        mlp_user_embed = self.mlp_user_embedding(user_input)
        mlp_item_embed = self.mlp_item_embedding(item_input)
        mlp_concat = tf.concat([mlp_user_embed, mlp_item_embed], axis=-1)
        
        # MLP前向传播
        mlp_output = mlp_concat
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
            if training:
                mlp_output = self.dropout(mlp_output)
                
        # NeuMF融合
        concat_output = tf.concat([
            self.alpha * gmf_output,
            (1-self.alpha) * mlp_output
        ], axis=-1)
        
        # 最终预测
        prediction = self.predict_layer(concat_output)
        return prediction
        
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            
            # 计算损失
            loss = self.compiled_loss(y, y_pred)
            
            # 添加L2正则化损失
            reg_losses = []
            for layer in self.mlp_layers:
                if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer:
                    reg_losses.append(layer.kernel_regularizer(layer.kernel))
            if reg_losses:
                loss += tf.add_n(reg_losses)
            
        # 计算梯度
        grads = tape.gradient(loss, self.trainable_variables)
        
        # 梯度裁剪
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        
        # 更新权重
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # 更新评估指标
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.gmf_user_embedding.input_dim,
            "num_items": self.gmf_item_embedding.input_dim,
            "embedding_size": self.gmf_user_embedding.output_dim,
            "mlp_layers": [layer.units for layer in self.mlp_layers],
            "alpha": self.alpha
        })
        return config
