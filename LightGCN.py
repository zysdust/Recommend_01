import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_layers, device, decay=1e-4):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        self.decay = decay

        # 用户和物品的嵌入矩阵
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 初始化嵌入 - 使用标准的Xavier初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # 层组合的权重系数 (α)
        self.alpha = 1. / (n_layers + 1)

    def create_adj_mat(self, user_item_mat):
        """创建归一化的邻接矩阵 (使用对称归一化)"""
        n_nodes = self.n_users + self.n_items
        adj_mat = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
        
        # 构建用户-物品交互矩阵
        adj_mat[:self.n_users, self.n_users:] = user_item_mat
        adj_mat[self.n_users:, :self.n_users] = user_item_mat.T
        
        # 对称归一化
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # D^(-1/2) * A * D^(-1/2)
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        return norm_adj.tocsr()

    def sparse_dropout(self, x, rate, noise_shape):
        """对稀疏矩阵进行dropout"""
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - rate))

    def forward(self, adj):
        """前向传播 - 实现标准的LightGCN传播规则"""
        # 获取初始嵌入
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        # 多层图卷积
        for k in range(self.n_layers):
            # 邻居聚合: E^(k+1) = (D^(-1/2) * A * D^(-1/2)) * E^k
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        # 层间聚合 - 使用加权平均
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = self.alpha * torch.sum(all_embeddings, dim=1)
        
        # 分离用户和物品嵌入
        user_all_embeddings = all_embeddings[:self.n_users, :]
        item_all_embeddings = all_embeddings[self.n_users:, :]
        
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, user_embeddings, item_embeddings, users, pos_items, neg_items):
        """计算BPR损失和正则化损失"""
        user_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]
        
        # 计算正样本和负样本的得分
        pos_scores = torch.sum(user_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_embeddings, dim=1)
        
        # BPR损失
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # L2正则化 - 只对初始嵌入进行正则化
        l2_loss = self.decay * (
            torch.norm(self.user_embedding.weight[users]) ** 2 +
            torch.norm(self.item_embedding.weight[pos_items]) ** 2 +
            torch.norm(self.item_embedding.weight[neg_items]) ** 2
        ) / len(users)
        
        return bpr_loss, l2_loss

    def predict(self, user_embeddings, item_embeddings, users):
        """预测用户对所有物品的评分"""
        user_embeddings = user_embeddings[users]
        scores = torch.matmul(user_embeddings, item_embeddings.t())
        return scores

    def get_ego_embeddings(self):
        """获取初始嵌入"""
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
