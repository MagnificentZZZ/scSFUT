import torch.nn as nn
import torch
import math
import numpy as np

from rtdl_num_embeddings import (
    LinearReLUEmbeddings,

)
# from mamba_ssm import Mamba
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoolingLayer(nn.Module):
    def forward(self, x):
        
        return torch.max(x, dim=2)[0]  # Simple max pooling


class TransformerEncoder(nn.Module): 
    def __init__(self, seq_length, token_dim, conv_emb_dim, num_layers_1 = 1, num_layers_2 = 4, num_heads = 8, mask_percentage = 0.4):
        super(TransformerEncoder, self).__init__()

        self.token_dim = token_dim
        self.seq_length = seq_length
        self.position_encoding = self.our_position_encoding(max_len=2000, d_model=token_dim)
        self.conv_layer = nn.Conv1d(in_channels=token_dim, out_channels=conv_emb_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2 * token_dim + conv_emb_dim, nhead=num_heads, batch_first=True),
        

            num_layers_1
        )

        self.pooling_layer = PoolingLayer()
    
        self.backembed = LinearReLUEmbeddings(n_features= seq_length, d_embedding= 2 * token_dim + conv_emb_dim)
        self.transformer_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2 * token_dim + conv_emb_dim, nhead=num_heads, batch_first=True),
        
            num_layers_2
        )

        self.Rec_Loss = nn.MSELoss()
        self.active = nn.ReLU()

    def our_position_encoding(self, max_len, d_model):

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        position_encoding = torch.zeros(max_len, d_model)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding



    def forward(self, x):
        batch_size, seq_len, token_dim = x.size()

        x = x.view(batch_size * seq_len, token_dim, 1)
        conv_embedding = self.conv_layer(x)
        conv_embedding = conv_embedding.view(batch_size, seq_len, -1)
        x = x.view(batch_size, seq_len, -1)
        x = torch.cat([x, conv_embedding], dim=-1)   
        position_embedding = self.position_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, seq_len, self.token_dim).to(device)
        x = torch.cat([x, position_embedding], dim=-1)  
        rec_x = self.transformer_encoder(x)
        x = self.pooling_layer(rec_x)

        return x
        
