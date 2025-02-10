import torch
import torch.nn as nn
import torch.nn.functional as F

class CIN(nn.Module):
    """Compressed Interaction Network"""
    def __init__(self, field_dims, layer_sizes, device):
        super().__init__()
        self.field_dims = field_dims
        self.layer_sizes = layer_sizes
        self.device = device
        
        # 参数初始化
        self.cin_layer = nn.ParameterList()
        field_nums = [len(field_dims)]  # H_0
        for i, size in enumerate(layer_sizes):
            field_num = field_nums[-1]
            self.cin_layer.append(
                nn.Parameter(torch.Tensor(1, field_num * field_nums[0], size)).to(device)
            )
            field_nums.append(size)
            
        # 初始化参数
        for param in self.cin_layer:
            nn.init.xavier_uniform_(param)
            
    def forward(self, x):
        """
        x: B x F x D
        """
        batch_size = x.shape[0]
        embedding_dim = x.shape[2]
        hidden_layers = [x]
        final_result = []
        
        for i, layer_size in enumerate(self.layer_sizes):
            # 计算特征交互
            x_hi = hidden_layers[-1]  # B x Hi x D
            x_h0 = hidden_layers[0]   # B x H0 x D
            
            # 特征图交互
            x_h = torch.einsum('bhd,bmd->bhmd', x_h0, x_hi)  # B x H0 x Hi x D
            x_h = x_h.reshape(batch_size, -1, embedding_dim)  # B x (H0*Hi) x D
            
            # 应用卷积核
            z = torch.einsum('bhd,whk->bkd', x_h, self.cin_layer[i])  # B x K x D
            
            hidden_layers.append(z)
            final_result.append(torch.sum(z, dim=2))  # B x K
            
        result = torch.cat(final_result, dim=1)  # B x K'
        return result

class xDeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cin_layer_sizes, device):
        super().__init__()
        self.device = device
        
        # Embedding层
        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        for embedding in self.embedding:
            nn.init.xavier_uniform_(embedding.weight.data)
            
        # Linear部分
        self.linear = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])
        for embedding in self.linear:
            nn.init.xavier_uniform_(embedding.weight.data)
            
        # Deep部分
        self.mlp = nn.ModuleList()
        input_dim = len(field_dims) * embed_dim
        for dim in mlp_dims:
            self.mlp.append(nn.Linear(input_dim, dim))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout))
            input_dim = dim
            
        # CIN部分
        self.cin = CIN(field_dims, cin_layer_sizes, device)
        
        # 输出层
        self.output_layer = nn.Linear(
            mlp_dims[-1] + sum(cin_layer_sizes) + len(field_dims), 1
        )
        
    def forward(self, x):
        """
        x: (batch_size, num_fields)
        """
        # Linear部分
        linear_out = torch.cat([
            emb(x[:, i]).reshape(-1, 1) for i, emb in enumerate(self.linear)
        ], dim=0)
        linear_out = linear_out.reshape(x.size(0), -1)
        
        # Embedding
        embeddings = [emb(x[:, i]) for i, emb in enumerate(self.embedding)]
        embeddings = torch.stack(embeddings, dim=1)  # B x F x D
        
        # CIN部分
        cin_out = self.cin(embeddings)  # B x K'
        
        # Deep部分
        mlp_in = embeddings.reshape(x.size(0), -1)  # B x (F*D)
        for layer in self.mlp:
            mlp_in = layer(mlp_in)
            
        # 组合所有输出
        out = torch.cat([linear_out, cin_out, mlp_in], dim=1)
        out = self.output_layer(out)
        out = torch.sigmoid(out.squeeze(1))
        
        return out
