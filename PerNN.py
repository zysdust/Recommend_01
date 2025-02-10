import torch
import torch.nn as nn
import torch.nn.functional as F

class PersonalizedFeaturesLayer(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(PersonalizedFeaturesLayer, self).__init__()
        # 用户和物品的嵌入矩阵 Wi
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 特征对之间的交互矩阵 Mij
        self.feature_interaction = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim)
        )
        
        # 注意力网络，用于计算历史交互的权重
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_ids, item_ids, user_history=None):
        # 计算用户和物品的嵌入向量: vi = Wi · xi
        user_emb = self.user_embedding(user_ids)  # [batch_size, embed_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, embed_dim]
        
        # 计算特征向量间的非线性交互: vi^T Mij vj
        interaction = torch.matmul(user_emb, self.feature_interaction)  # [batch_size, embed_dim]
        interaction = torch.sum(interaction * item_emb, dim=1, keepdim=True)  # [batch_size, 1]
        
        if user_history is not None:
            # 计算历史交互的表示
            history_emb = self.item_embedding(user_history)  # [batch_size, history_len, embed_dim]
            
            # 计算注意力权重
            attention_input = torch.cat([
                user_emb.unsqueeze(1).expand(-1, history_emb.size(1), -1), 
                history_emb
            ], dim=-1)
            attention_weights = self.attention(attention_input)  # [batch_size, history_len, 1]
            
            # 加权求和得到历史交互的表示
            history_repr = torch.sum(attention_weights * history_emb, dim=1)  # [batch_size, embed_dim]
            
            # 将历史信息融入用户表示
            user_emb = user_emb + history_repr
        
        return user_emb, item_emb, interaction

class PerNN(nn.Module):
    def __init__(self, num_features, num_users, num_items, num_factors, hidden_layers=[64, 32], dropout=0.1):
        super(PerNN, self).__init__()
        
        self.num_factors = num_factors  # 保存embedding维度以便后续使用
        
        # 个性化特征层
        self.pf_layer = PersonalizedFeaturesLayer(num_users, num_items, num_factors)
        
        # FM部分的嵌入层: vi = Wi · xi
        self.embedding = nn.Embedding(num_features, num_factors)
        
        # FM部分的一阶特征
        self.linear = nn.Embedding(num_features, 1)
        
        # 高阶特征交互层的参数
        # 计算输入维度：2(特征数) * num_factors + 2 * num_factors(用户物品)
        high_order_input_dim = num_factors * 4
        self.high_order_layer = nn.Sequential(
            nn.Linear(high_order_input_dim, num_factors),
            nn.ReLU()
        )
        
        # Deep部分的层
        layers = []
        # 输入维度：原始特征 + 用户特征 + 物品特征 + 高阶特征
        input_dim = num_factors * 5
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.deep_layers = nn.Sequential(*layers)
        
        # L2正则化参数
        self.lambda_l2 = 0.01
        
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 初始化嵌入层
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.pf_layer.user_embedding.weight, std=0.01)
        nn.init.normal_(self.pf_layer.item_embedding.weight, std=0.01)
        
        # 初始化线性层
        nn.init.normal_(self.linear.weight, std=0.01)
        
        # 初始化深度网络层
        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def fm_layer(self, embedding_vectors):
        # 计算FM部分的二阶交互: PF_interactions = Σ Σ <vi, vj>
        square_of_sum = torch.sum(embedding_vectors, dim=1).pow(2)
        sum_of_square = torch.sum(embedding_vectors.pow(2), dim=1)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * cross_term
        return cross_term
    
    def forward(self, features, user_ids=None, item_ids=None, user_history=None):
        # 一阶特征
        linear_out = torch.sum(self.linear(features), dim=1).squeeze(-1)
        
        # 获取基础特征嵌入向量: vi = Wi · xi
        embedding_vectors = self.embedding(features)  # [batch_size, num_fields, embed_dim]
        
        if user_ids is not None and item_ids is not None:
            # 获取个性化特征和特征交互
            user_emb, item_emb, pf_interaction = self.pf_layer(user_ids, item_ids, user_history)
            
            # 计算FM部分的特征交互
            fm_out = self.fm_layer(embedding_vectors)
            
            # 准备高阶特征输入
            base_features = embedding_vectors.view(embedding_vectors.size(0), -1)  # [batch_size, num_fields*embed_dim]
            concat_features = torch.cat([
                base_features,
                user_emb,
                item_emb
            ], dim=1)
            
            # 计算高阶特征
            high_order = self.high_order_layer(concat_features)
            
            # Deep部分：将所有特征展平并拼接
            deep_in = torch.cat([
                base_features,
                user_emb,
                item_emb,
                high_order
            ], dim=1)
            
            # 深度网络前向传播: h^(l+1) = σ(W^(l) h^(l) + b^(l))
            deep_out = self.deep_layers(deep_in)
            
            # 组合所有输出并应用sigmoid激活函数
            output = linear_out + torch.sum(fm_out, dim=1) + pf_interaction.squeeze(-1) + deep_out.squeeze(-1)
        else:
            fm_out = self.fm_layer(embedding_vectors)
            deep_in = embedding_vectors.view(embedding_vectors.size(0), -1)
            deep_out = self.deep_layers(deep_in)
            output = linear_out + torch.sum(fm_out, dim=1) + deep_out.squeeze(-1)
        
        # 最终输出: y = sigmoid(output)
        return torch.sigmoid(output)
    
    def get_l2_loss(self):
        """计算L2正则化损失: L_re = L + λ||W^(l)||_F^2"""
        l2_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:  # 只对权重参数进行正则化
                l2_loss += torch.norm(param, p=2)
        return self.lambda_l2 * l2_loss
