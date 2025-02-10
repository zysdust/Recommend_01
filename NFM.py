import torch
import torch.nn as nn
import torch.nn.functional as F

class NFM(nn.Module):
    def __init__(self, num_features, num_factors, hidden_layers=[64, 32], dropout=0.1):
        super(NFM, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(num_features, num_factors)
        
        # 一阶特征线性层
        self.linear = nn.Embedding(num_features, 1)
        
        # 深度神经网络层
        layers = []
        input_dim = num_factors
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.deep_layers = nn.Sequential(*layers)
        
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 初始化嵌入层
        nn.init.normal_(self.embedding.weight, std=0.01)
        
        # 初始化线性层
        nn.init.normal_(self.linear.weight, std=0.01)
        
        # 初始化深度网络层
        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def bi_interaction(self, embedding_vectors):
        # 计算二阶特征交互
        sum_square = torch.sum(embedding_vectors, dim=1).pow(2)
        square_sum = torch.sum(embedding_vectors.pow(2), dim=1)
        bi_pooling = (sum_square - square_sum) * 0.5
        return bi_pooling
    
    def forward(self, features):
        # 一阶特征
        linear_out = torch.sum(self.linear(features), dim=1).squeeze(-1)
        
        # 获取嵌入向量
        embedding_vectors = self.embedding(features)
        
        # 二阶特征交互
        bi_pooling = self.bi_interaction(embedding_vectors)
        
        # 深度网络
        deep_out = self.deep_layers(bi_pooling)
        
        # 组合输出
        output = linear_out + deep_out.squeeze(-1)
        
        return output
