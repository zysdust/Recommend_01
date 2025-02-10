import tensorflow as tf

class CIN(tf.keras.layers.Layer):
    def __init__(self, layer_size, activation='relu'):
        super(CIN, self).__init__()
        self.layer_size = list(layer_size)  # 将tuple转换为list
        self.activation = activation
        
    def build(self, input_shape):
        # input_shape = (batch_size, field_nums, embedding_size)
        self.field_nums = [input_shape[1]] + self.layer_size
        self.filters = []
        
        for i in range(len(self.layer_size)):
            filter_shape = [1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]]
            curr_filter = self.add_weight(
                name=f'filter_{i}',
                shape=filter_shape,
                dtype=tf.float32,
                trainable=True,
                initializer='glorot_uniform'
            )
            self.filters.append(curr_filter)
    
    def call(self, inputs):
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []
        
        for i, layer_size in enumerate(self.layer_size):
            z_i = tf.einsum('bhd,bmd->bhmd', hidden_nn_layers[0], hidden_nn_layers[-1])
            z_i = tf.reshape(z_i, shape=[tf.shape(inputs)[0], self.field_nums[0] * self.field_nums[i], dim])
            
            z_i = tf.transpose(z_i, [0, 2, 1])  # (batch_size, dim, field_nums[0] * field_nums[i])
            z_i = tf.expand_dims(z_i, 2)        # (batch_size, dim, 1, field_nums[0] * field_nums[i])
            
            filter_v = tf.expand_dims(self.filters[i], 0)
            curr_out = tf.nn.conv2d(z_i, filter_v, strides=[1, 1, 1, 1], padding='VALID')
            curr_out = tf.squeeze(curr_out, axis=2) # (batch_size, dim, layer_size[i])
            curr_out = tf.transpose(curr_out, [0, 2, 1]) # (batch_size, layer_size[i], dim)
            
            if self.activation:
                curr_out = tf.nn.relu(curr_out)
                
            hidden_nn_layers.append(curr_out)
            final_result.append(curr_out)
            
        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, axis=-1)  # (batch_size, sum(layer_size))
        
        return result

class xDeepFM(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=16, dnn_hidden_units=(256, 128, 64), 
                 cin_layer_size=(128, 128), dropout_rate=0.2):
        super(xDeepFM, self).__init__()
        
        # Embedding layers
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        
        # Linear part
        self.linear = tf.keras.layers.Dense(1)
        
        # Deep part
        self.dnn_network = []
        for units in dnn_hidden_units:
            self.dnn_network.extend([
                tf.keras.layers.Dense(units, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(dropout_rate)
            ])
        self.dnn_network.append(tf.keras.layers.Dense(1))
        
        # CIN part
        self.cin = CIN(cin_layer_size)
        self.cin_linear = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        user_ids, item_ids = inputs[:, 0], inputs[:, 1]
        
        # Embedding
        user_emb = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item_ids)  # (batch_size, embedding_dim)
        
        # Concatenate embeddings
        concat_emb = tf.concat([
            tf.expand_dims(user_emb, axis=1),
            tf.expand_dims(item_emb, axis=1)
        ], axis=1)  # (batch_size, 2, embedding_dim)
        
        # Linear part
        linear_input = tf.reshape(concat_emb, [-1, 2 * self.user_embedding.output_dim])
        linear_out = self.linear(linear_input)
        
        # Deep part
        deep_input = tf.reshape(concat_emb, [-1, 2 * self.user_embedding.output_dim])
        deep_out = deep_input
        for layer in self.dnn_network:
            deep_out = layer(deep_out, training=training)
            
        # CIN part
        cin_out = self.cin(concat_emb)  # concat_emb已经是(batch_size, field_nums, embedding_dim)格式
        cin_out = self.cin_linear(cin_out)
        
        # Combine outputs
        output = tf.nn.sigmoid(linear_out + deep_out + cin_out)
        return output
