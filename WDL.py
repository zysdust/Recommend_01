import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32]):
        super(WideAndDeep, self).__init__()
        
        # Wide部分 - 线性模型
        self.wide = nn.Linear(num_users + num_items, 1)
        
        # Deep部分 - 深度神经网络
        # Embedding层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 构建深度网络层
        layers = []
        input_dim = embedding_dim * 2  # user和item embedding拼接
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)
        
    def forward(self, user_indices, item_indices):
        # Wide部分
        wide_input = torch.zeros(user_indices.size(0), self.wide.in_features, device=user_indices.device)
        wide_input.scatter_(1, user_indices.unsqueeze(1), 1)
        wide_input.scatter_(1, item_indices.unsqueeze(1) + self.user_embedding.num_embeddings, 1)
        wide_out = self.wide(wide_input)
        
        # Deep部分
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        deep_input = torch.cat([user_emb, item_emb], dim=1)
        deep_out = self.deep(deep_input)
        
        # 组合Wide和Deep的输出
        return torch.sigmoid(wide_out + deep_out)
