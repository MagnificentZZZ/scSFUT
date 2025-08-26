import torch.nn as nn
import torch
import math
import numpy as np
from performer_pytorch import Performer

from rtdl_num_embeddings import (
    LinearReLUEmbeddings,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PreprocessLayer(nn.Module):
    def __init__(self, mask_percentage):
        super(PreprocessLayer, self).__init__()
        self.mask_percentage = mask_percentage

    def forward(self, data):
        # 判断每个向量是否为空（所有值都为零）
        is_empty_vector = torch.all(data == 0, dim=-1)
        # 统计每个 batch 中非空向量的数量
        non_empty_counts = torch.sum(~is_empty_vector, dim=1)
        # 统计空向量的数量
        empty_counts = data.size(1) - non_empty_counts

        # 计算每个样本中需要 mask 的向量数（非空向量数 * percentage）
        num_vectors_to_mask = (non_empty_counts * self.mask_percentage).long()
        # 计算每个样本中需要 mask 的空向量数（空向量数 * (percentage /5)）
        num_empty_vectors_to_mask = (empty_counts * 0.10).long()
        # num_empty_vectors_to_mask = 0

        # 创建 mask
        mask = torch.zeros_like(data)

        # 对每个样本进行处理
        for i in range(data.size(0)):  # 遍历 batch_size
            # 找到非零的索引
            non_empty_indices = torch.nonzero(~is_empty_vector[i, :], as_tuple=False)

            # 随机选择需要 mask 的非空向量索引
            indices_to_mask = non_empty_indices[torch.randperm(non_empty_indices.size(0))[:num_vectors_to_mask[i]]]
            # 对 mask 中对应位置赋值为 1
            mask[i, indices_to_mask, :] = 1

            # 随机选择需要 mask 的空向量索引
            empty_indices_to_mask = torch.randperm(data.size(1))[:num_empty_vectors_to_mask[i]]
            # 对 mask 中对应位置赋值为 1
            mask[i, empty_indices_to_mask, :] = 1

        # 返回遮盖后的数据和遮盖掩码

        return (1 - mask) * data, mask

class PoolingLayer(nn.Module):
    def forward(self, x):
        
        return torch.max(x, dim=2)[0]  # Simple max pooling
        # return torch.mean(x, dim=2)

class TransformerEncoder(nn.Module): 
    def __init__(self, seq_length, token_dim, conv_emb_dim, num_layers_1, num_layers_2, num_heads, mask_percentage):
        super(TransformerEncoder, self).__init__()

        self.token_dim = token_dim
        self.seq_length = seq_length
        self.preprocess_layer = PreprocessLayer(mask_percentage)
        self.position_encoding = self.our_position_encoding(max_len=2000, d_model=token_dim)
        self.conv_layer = nn.Conv1d(in_channels=token_dim, out_channels=conv_emb_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2 * token_dim + conv_emb_dim, nhead=num_heads, batch_first=True),
            # nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True),

            num_layers_1
        )

        self.pooling_layer = PoolingLayer()
        # self.backembed = LinearReLUEmbeddings(n_features= seq_length, d_embedding= 2 * token_dim + conv_emb_dim)
        self.backembed = LinearReLUEmbeddings(n_features= seq_length, d_embedding= 2 * token_dim + conv_emb_dim)
        self.transformer_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2 * token_dim + conv_emb_dim, nhead=num_heads, batch_first=True),
            # nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True),
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
        # 卷积操作
        conv_embedding = self.conv_layer(x)
        # 调整形状回 (batch_size, seq_len, token_dim)
        conv_embedding = conv_embedding.view(batch_size, seq_len, -1)
        x = x.view(batch_size, seq_len, -1)
        x = torch.cat([x, conv_embedding], dim=-1)   

        masked_x, mask = self.preprocess_layer(x)
        position_embedding = self.position_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, seq_len, self.token_dim).to(device)#[64*258*10]
        x = torch.cat([x, position_embedding], dim=-1)  # 在 token_dim 维度上拼接绝对位置编码[64*258*35]
        masked_x = torch.cat([masked_x, position_embedding], dim=-1)  # 在 token_dim 维度上拼接绝对位置编码[64*258*35]
        mask = torch.cat([mask,torch.zeros_like(position_embedding)], dim=-1)
        # 对 mask 进行拼接
        mask = torch.where(mask[:, :,:1] == 0, torch.zeros_like(mask), torch.ones_like(mask))
        mask[:,-1,:] = 0
        masked_ori_data = mask * x

        rec_masked_x = self.transformer_encoder(masked_ori_data)
        rec_masked_x = self.pooling_layer(rec_masked_x)
        rec_masked_x = self.backembed(rec_masked_x)
        rec_masked_x = self.transformer_decoder(rec_masked_x)

        #只计算mask区域的损失
        masked_fin_data = mask * rec_masked_x
        rec_loss = self.Rec_Loss(masked_ori_data, masked_fin_data)
        
        #前向输入的是完整的，未被mask的真实数据
        rec_x = self.transformer_encoder(x)

        x = self.pooling_layer(rec_x)

        return x, rec_loss
        
