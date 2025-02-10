import torch
import torch.nn as nn
import torch.nn.functional as F

class MultVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[600, 200], latent_dim=100, dropout_rate=0.5):
        super(MultVAE, self).__init__()
        
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        # 编码器层
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
            
        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        # 均值和方差层
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器层
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder_layers(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def decode(self, z):
        return self.decoder_layers(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, anneal=1.0):
        # 重构损失 (负对数似然)
        recon_loss = -torch.sum(F.log_softmax(recon_x, dim=-1) * x, dim=-1)
        recon_loss = torch.mean(recon_loss)
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # 总损失
        total_loss = recon_loss + anneal * kl_loss
        
        return total_loss
