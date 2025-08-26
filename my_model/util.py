import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import json
from datetime import datetime

path_all = './results/'

def setup_seed(seed):
    np.random.seed(seed) # numpy 的设置
    random.seed(seed)  # python random module
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了使得hash随机化，使得实验可以复现
    torch.manual_seed(seed) # 为cpu设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU为，所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False # 设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法。
        torch.backends.cudnn.deterministic = True # 每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，
                                                    # 应该可以保证每次运行网络的时候相同输入的输出是固定的
            
            
def plot(y1, name, X, Y, y2 = None):
    # 创建与y轴数据相同长度的x轴数据
    x = np.arange(len(y1))
    plt.scatter(x, y1, label=name)
    if y2 is not None: plt.scatter(x, y2, label='y_pred')
    plt.legend()
    # 设置图像标题和坐标轴标签
    plt.title(name)
    plt.xlabel(X)
    plt.ylabel(Y)
    # 显示图像
    plt.savefig(path + name)
    # plt.show()
    plt.close()
    
def plot_cmp(Y_TEST_TRUE, Y_TEST_HAT, fold, name): 
    plt.figure(figsize=(10, 6))
    plt.plot(Y_TEST_TRUE, label='True Values', color='blue')
    plt.plot(Y_TEST_HAT, label='Predicted Values', color='orange')
    plt.title("True vs. Predicted Values")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    
    save_filename = path_all + f'/pre_vs_true_{fold}' + name
    plt.savefig(save_filename)
    plt.close()
     
def count_labels(data_loader):
    labels = []
    for _, label in data_loader:
        labels.extend(label.tolist())
    return Counter(labels)

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 将输入值转换为概率值
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 获取每个样本的预测概率
        pt = torch.exp(-CE_loss)

        # 计算 Focal Loss
        F_loss = (1 - pt) ** self.gamma * CE_loss

        # 如果设置了 alpha 参数，则乘以 alpha 的权重
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            F_loss = F_loss * alpha_t

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def compute_mean_std(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # 使用样本标准差
    return round(mean, 4), round(std, 4)

import numpy as np
import pandas as pd
import scanpy as sc

def seq2token(adata, embedding_dim):
   
    adata = adata
    data_df = pd.DataFrame(adata.X, index=adata.obs.index)
    data_df['cell_type'] = adata.obs['cell_type']
    # print(data_df['cell_type'])
    has_label = data_df['cell_type'].notna().any()
    if ~has_label:
        data_df['cell_type'] = adata.obs['pse_cell_type']

    # 特征部分
    features = data_df.iloc[:, :-1]
    features.index = range(len(features))

    # 计算需要填充的位数
    num_samples, num_features = features.shape
    remaining_features = num_features % embedding_dim

    # 补充 features 的维度
    if remaining_features != 0:
        padding_size = embedding_dim - remaining_features
        features_padded = pd.concat([features, pd.DataFrame(np.zeros((num_samples, padding_size)))], axis = 1)

    else:
        features_padded = features

    # 计算 token 的数量
    # print(features_padded.iloc[:, -2])
    num_tokens = features_padded.shape[1] // embedding_dim
    grouped_features = np.zeros((num_samples, num_tokens, embedding_dim))
    # print(grouped_features)

    # 将特征分组到 embedding 维度
    for i in range(num_tokens):
        start_idx = i * embedding_dim
        end_idx = start_idx + embedding_dim
        grouped_features[:, i, :] = features_padded.iloc[:, start_idx:end_idx]

    #保存特征和标签
    if has_label:
        np.save('./input_data/label_data_x.npy', grouped_features)
        np.save('./input_data/label_data_y.npy', data_df['cell_type'].values)
    else:
        np.save('./input_data/unlabelled_data_x.npy', grouped_features)
        np.save('./input_data/unlabelled_data_y.npy', data_df['cell_type'].values)

    print(f"Data processed. {'Labels saved.' if has_label else 'No labels to save.'}")

def save_args_and_results(args, results, save_path):
    args_dict = vars(args)  # 将 args 转换为字典
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    new_data = {
        'args': args_dict,  # 保存参数
        'results': results,  # 保存五折结果
        'timestamp': timestamp  # 保存时间戳
    }

    # 如果文件已经存在，读取旧的内容
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            try:
                data = json.load(f)  # 读取现有的 JSON 数据
            except json.JSONDecodeError:
                data = []  # 如果文件为空或损坏，初始化为空列表
    else:
        data = []

    # 将新的数据追加到已有数据
    data.append(new_data)

    # 写入文件，覆盖写入合并后的结果
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def shuffle_inner_token_values(dataset, new_seed):
    """
    对输入数据集中的每个样本的 token 维度进行乱序操作，不改变样本顺序或标签。

    参数:
    - dataset: 数据集，包含形如 (样本数据, 标签) 的二元组，样本数据的形状为 [token_num, token_dim]。

    返回:
    - 新的数据集，样本的顺序不变，但每个样本的 token 内部的值被随机打乱。
    """
    shuffled_dataset = []
    
    for sample, label in dataset:
        token_num, token_dim = sample.shape
        torch.manual_seed(new_seed)
        # 对每个 token 内部的值进行乱序
        for j in range(token_num):
            sample[j] = sample[j][torch.randperm(token_dim)]

        shuffled_dataset.append((sample, label))
    
    return shuffled_dataset

def shuffle_inner_token_values_aligned(dataset, new_seed):
    """
    对输入数据集中的每个样本的 token 维度进行相同顺序的乱序操作，不改变样本顺序或标签。

    参数:
    - dataset: 数据集，包含形如 (样本数据, 标签) 的二元组，样本数据的形状为 [token_num, token_dim]。
    - new_seed: 随机种子，用于控制乱序过程的可重复性。

    返回:
    - 新的数据集，样本的顺序不变，但每个样本的 token 内部的值根据相同的索引顺序被随机打乱。
    """
    shuffled_dataset = []

    # 使用指定种子确保每次实验的随机顺序相同
    torch.manual_seed(new_seed)

    # 从第一个样本中获取 token_dim，用于生成相同的乱序索引
    sample_dim = dataset[0][0].shape[1]
    
    # 生成固定的随机索引，用于每个样本的所有 token 维度
    permuted_indices = torch.randperm(sample_dim)
    
    for sample, label in dataset:
        token_num, token_dim = sample.shape

        # 对每个 token 内部的值按照相同的索引顺序进行乱序
        for j in range(token_num):
            sample[j] = sample[j][permuted_indices]

        shuffled_dataset.append((sample, label))

    return shuffled_dataset

def shuffle_inter_token_values(dataset, new_seed):
    """
    对输入数据集中的每个样本的 token 序列进行乱序操作，不改变样本顺序或标签。

    参数:
    - dataset: 数据集，包含形如 (样本数据, 标签) 的二元组，样本数据的形状为 [token_num, token_dim]。
    - new_seed: 随机种子，用于控制每次实验的随机性。

    返回:
    - 新的数据集，样本的顺序不变，但每个样本的 token 序列被随机打乱。
    """
    shuffled_dataset = []
    
    for sample, label in dataset:
        token_num, token_dim = sample.shape
        
        # 设置随机种子，确保每次打乱顺序不同
        torch.manual_seed(new_seed)
        
        # 对 token 的顺序进行乱序操作
        shuffled_indices = torch.randperm(token_num)  # 生成打乱后的 token 索引
        sample = sample[shuffled_indices]  # 按打乱的索引重排 token 顺序

        shuffled_dataset.append((sample, label))
    
    return shuffled_dataset

def shuffle_inter_token_values_aligned(dataset, new_seed):
    """
    对输入数据集中的每个样本的 token 序列进行相同的乱序操作，不改变样本顺序或标签。

    参数:
    - dataset: 数据集，包含形如 (样本数据, 标签) 的二元组，样本数据的形状为 [token_num, token_dim]。
    - new_seed: 随机种子，用于控制每次实验的随机性。

    返回:
    - 新的数据集，样本的顺序不变，但每个样本的 token 序列被按照相同的方式随机打乱。
    """
    shuffled_dataset = []
    
    # 设置一次随机种子，保证所有样本的 token 序列使用相同的打乱顺序
    torch.manual_seed(new_seed)
    
    # 假设所有样本的 token_num 相同，取第一个样本的 token_num
    token_num = dataset[0][0].shape[0]
    
    # 生成打乱后的 token 索引
    shuffled_indices = torch.randperm(token_num)

    for sample, label in dataset:
        # 按相同的打乱索引顺序对 token 进行重排
        sample = sample[shuffled_indices]
        shuffled_dataset.append((sample, label))
    
    return shuffled_dataset








