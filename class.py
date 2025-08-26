import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from my_model.mlp_cls import MLP
from my_model.trans_enc_cls import PoolingLayer, TransformerEncoder
from my_model.mydata import mydataSet
from my_model.util import setup_seed, count_labels, FocalLoss, compute_mean_std, save_args_and_results
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import copy
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pickle
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split

# Ignore UndefinedMetricWarning when cal precision
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Train with', device)

parser = argparse.ArgumentParser()
#Global Params
parser.add_argument("--dataset", type=str, default='Kidney', help='Name of data for train, val and test.')
parser.add_argument("--batch_size", type=int, default=64, help='Number of batch.')
parser.add_argument("--num_epoch", type=int, default=25, help='Number of epochs.')
parser.add_argument("--num_fold", type=int, default=5, help='Number of fold.')
parser.add_argument("--seed", type=int, default=42, help='Random seed.')
parser.add_argument("--learning_rate", type=float, default=0.0005, help='Learning rate.')
parser.add_argument("--scheduler_type", type=bool, default=True, help='CosineAnnealingLR exists or not.')
parser.add_argument("--w_recloss", type=float, default=1.0, help='Weight of recloss.')
parser.add_argument("--w_clsloss", type=float, default=1.5, help='Weight of cls loss(FocalLoss).')
# Params of MLP
parser.add_argument("--hidden_size_1", type=int, default=512, help='Size of Linear 1.')
parser.add_argument("--hidden_size_2", type=int, default=64, help='Size of Linear 2.')
parser.add_argument("--dropout", type=float, default=0.2, help='Dropout rate of classfier.')
#Params of Transformer
parser.add_argument("--token_dim", type=int, default=64, help='Dimension of token.')
parser.add_argument("--conv_dim", type=int, default=128, help='Dimension of conv block.')
parser.add_argument("--layer1", type=int, default=2, help='Layer num of Transformer encoder.')
parser.add_argument("--layer2", type=int, default=3, help='Layer num of Transformer decoder.')
parser.add_argument("--mask_percentage", type=float, default=0.4, help='Mask percentage of non_zero token, and the mask percentage of zero token is 1/10 of this num.')
parser.add_argument("--num_mulhead", type=int, default=8, help='Number of head in Transformer.')
args = parser.parse_args()

DATASET_NAME = args.dataset
BATCH_SIZE = args.batch_size
EPOCH = args.num_epoch
FOLD = args.num_fold
RANDOM_SEED = args.seed
LR = args.learning_rate
IF_COS = args.scheduler_type
W_REC = args.w_recloss
W_CLS = args.w_clsloss
HIDDEN_SIZE_1 = args.hidden_size_1
HIDDEN_SIZE_2 = args.hidden_size_2
DROPOUT = args.dropout
TOKEN_DIM = args.token_dim
CONV_DIM = args.conv_dim
NUM_LAYER1 = args.layer1
NUM_LAYER2 = args.layer2
M_PERCENTAGE = args.mask_percentage
NUM_HEAD = args.num_mulhead

def data_produce():
  
    adata = anndata.read_h5ad(f"./cls_data/{DATASET_NAME}/preprocessed_{DATASET_NAME}.h5ad")   #Load
    # print(adata.var_names)
    print('Raw data shape', adata.X.shape)
    print('Raw data max:', adata.X.max())

    # sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_genes(adata, min_cells=3)
    # sc.pp(adata, flavor='seurat_v3', n_top_genes=f_dim, subset=True)

    # 3 Human datasets had been normed by scBERT(preprocess.py)
    print(DATASET_NAME)
    # sc.pp.normalize_total(adata)
    if DATASET_NAME in ['Kidney', 'Mat']:
        print('With Norm')
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    # # adata.write_h5ad("./cls_data/preprocessed_{DATASET_NAME}.h5ad")

    X = adata.X
    if DATASET_NAME == 'Zheng68K':
        y = adata.obs['celltype']
    elif DATASET_NAME in ['Baron', 'Segerstolpe']:
        y = adata.obs['cell_type']
    else:
        y = adata.obs['cell_ontology_class']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    data = pd.DataFrame(X.toarray() if hasattr(X, 'toarray') else X, columns=adata.var_names)
    data["label"] = y_encoded
    print('Preprocessed data shape:', data.shape)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    with open(f'label_mapping_{DATASET_NAME}.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
    print("Label Mapping:", label_mapping)
    label_counts = data["label"].value_counts()
    print("Label Counts:", label_counts)

    #-----------------------From seq2token------------------------
    embedding_dim = TOKEN_DIM
    features = data.iloc[:, :-1]
    num_samples, num_features = features.shape

    #Calculate and add the dim of last token if needed
    remaining_features = num_features % embedding_dim
    if remaining_features != 0:
        padding_size = embedding_dim - remaining_features
        features_padded = pd.concat([features, pd.DataFrame(np.zeros((num_samples, padding_size)))], axis=1)
    else:
        features_padded = features

    # Calculate the number of tokens
    num_tokens = features_padded.shape[1] // embedding_dim
    # Create a 3D array to store data
    grouped_features = np.zeros((num_samples, num_tokens, embedding_dim))

    # Fill the array with token vectors
    for i in range(num_tokens):  

        start_idx = i * embedding_dim
        end_idx = start_idx + embedding_dim
        grouped_features[:, i, :] = features_padded.iloc[:, start_idx:end_idx]

    np.save(f'input_data/{DATASET_NAME}/data_x.npy', grouped_features)
    np.save(f'input_data/{DATASET_NAME}/data_y.npy', data.iloc[:, -1])

def train():
    
    # Read preprocessed features and label
    data = np.load(f'input_data/{DATASET_NAME}/data_x.npy')
    _, token_num, _ = data.shape
    label = np.load(f'input_data/{DATASET_NAME}/data_y.npy')
    NUM_CLASS = int(label.max()) + 1
    print('num_class:', NUM_CLASS)
    my_dataset = mydataSet(data, label)
    print(data.shape)

    kf = StratifiedKFold(n_splits=FOLD, shuffle=True)
    kf_second = StratifiedKFold(n_splits=4, shuffle=True)

    # -----------------------cross val------------------------------
    test_ACC = []
    test_F1 = []
    test_PRE = []
    for fold, (train_indices, test_indices) in enumerate(kf.split(my_dataset, label)):

        
        best_model_wts = None
        best_f1 = 0.0

        # tmp_x = my_dataset[train_indices][0]
        # tmp_y = my_dataset[train_indices][1]

        # new_train_indices, new_val_indices = next(kf_second.split(tmp_x,tmp_y), tmp_y)
        train_indices = np.array(train_indices)  # 确保是 NumPy 数组

        # 按照 3:1 的比例随机划分
        train_split_indices, val_split_indices = train_test_split(
            train_indices, 
            test_size=0.25,  # 验证集占 25%（即 1/4）
        )
        train_indices = train_indices[train_split_indices]
        val_indices = train_indices[val_split_indices]

        # from sklearn.model_selection import train_test_split

        # # 假设 label 是一个包含所有样本标签的数组
        # # train_index 是当前保留的训练集索引列表
        # train_labels = label[train_indices]

        # # 获取新的索引列表，分层抽样，每类保留10%
        # _, new_index_in_train = train_test_split(
        #     range(len(train_labels)),  # 按当前训练集的长度生成索引
        #     test_size=0.3,  # 保留 10% 数据
        #     stratify=train_labels,  # 根据标签分层
        #     random_state=42  # 确保随机性可复现
        # )

        # # 将相对于 train_index 的索引，映射回原始数据的索引
        # new_index = np.array(train_indices)[new_index_in_train]
        # train_indices = new_index
        # print(train_indices.shape)
        # 打印前五个样本
        # print("训练样本中的前五个样本:")
        # for i in range(0, 10):
        #     print(f"标签 = {my_dataset[train_indices][1][i]}")
        
        # new_adata = anndata.read_h5ad("./cls_data/Baron/Baron_fold_1_train.h5ad")   #Load
        # print(new_adata.obs['cell_type'][:10])
        
        # return 

        # # 检查验证集和训练集是否有重叠
        # val_train_overlap = set(train_indices) & set(val_indices)
        # print("Validation set and training set overlap:", val_train_overlap)
        # # 检查测试集和训练集是否有重叠
        # test_train_overlap = set(train_indices) & set(test_indices)
        # print("Test set and training set overlap:", test_train_overlap)
        # # 检查测试集和验证集是否有重叠
        # test_val_overlap = set(val_indices) & set(test_indices)
        # print("Test set and validation set overlap:", test_val_overlap)

        # adata_fold_1 = adata[train_indices].copy()  # 选择保存测试数据，也可以选择 train_indices
        # adata_fold_2 = adata[val_indices].copy()  # 选择保存测试数据，也可以选择 train_indices
        # adata_fold_3 = adata[test_indices].copy()  # 选择保存测试数据，也可以选择 train_indices

        # fold_path_1 = os.path.join(output_dir, f'Mat_fold_{fold + 1}_train.h5ad')
        # fold_path_2 = os.path.join(output_dir, f'Mat_fold_{fold + 1}_val.h5ad')
        # fold_path_3 = os.path.join(output_dir, f'Mat_fold_{fold + 1}_test.h5ad')

        # adata_fold_1.write_h5ad(fold_path_1)
        # adata_fold_2.write_h5ad(fold_path_2)
        # adata_fold_3.write_h5ad(fold_path_3)

        # print(f'Fold {fold + 1} saved')
        # continue

        transformer_model = TransformerEncoder(seq_length=token_num, token_dim=TOKEN_DIM, conv_emb_dim=CONV_DIM, num_layers_1=NUM_LAYER1, num_layers_2=NUM_LAYER2, num_heads=NUM_HEAD, mask_percentage=M_PERCENTAGE).double()
        classification_model = MLP(input_dim=token_num, hidden_dim1 = HIDDEN_SIZE_1, hidden_dim2 = HIDDEN_SIZE_2, num_classes=NUM_CLASS, dropout=DROPOUT).double()
        transformer_model.to(device)
        classification_model.to(device)

        # def count_parameters(model):
        #     total_params = sum(p.numel() for p in model.parameters())
        #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     return total_params, trainable_params

        # transformer_total, transformer_trainable = count_parameters(transformer_model)
        # classification_total, classification_trainable = count_parameters(classification_model)

        # print(f"Transformer Model - Total Parameters: {transformer_total}, Trainable Parameters: {transformer_trainable}")
        # print(f"Classification Model - Total Parameters: {classification_total}, Trainable Parameters: {classification_trainable}")
        from fvcore.nn import FlopCountAnalysis

        # 假设模型和设备
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # # 假设 Transformer 模型和分类模型
        # transformer_model = transformer_model.to(device)  # 将 Transformer 模型迁移到设备
        # classification_model = classification_model.to(device)  # 将分类模型迁移到设备

        # # 假设输入维度
        # seq_len = 202
        # token_dim = 64
        # hidden_dim = 202  # Transformer 输出的隐藏维度（根据模型定义）

        # # Transformer 模型 FLOPs 和参数计算
        # transformer_input = torch.randn(64, seq_len, token_dim).to(device)  # Batch size = 1
        # flops_transformer = FlopCountAnalysis(transformer_model, transformer_input.double()).total()
        # params_transformer = sum(p.numel() for p in transformer_model.parameters())

        # # 分类模型 FLOPs 和参数计算
        # classification_input = torch.randn(64, hidden_dim).to(device)  # Batch size = 1
        # flops_class = FlopCountAnalysis(classification_model, classification_input.double()).total()
        # params_class = sum(p.numel() for p in classification_model.parameters())

        # # 汇总结果
        # total_flops = flops_transformer + flops_class
        # total_params = params_transformer + params_class

        # # 打印结果
        # print(f"Transformer Model - FLOPs: {flops_transformer / 1e9:.2f} GFLOPs, Params: {params_transformer / 1e6:.2f} M")
        # print(f"Classification Model - FLOPs: {flops_class / 1e9:.2f} GFLOPs, Params: {params_class / 1e6:.2f} M")
        # print(f"Total - FLOPs: {total_flops / 1e9:.2f} GFLOPs, Params: {total_params / 1e6:.2f} M")



        criterion = FocalLoss(gamma = 0)
        optimizer = torch.optim.Adam(list(transformer_model.parameters()) + list(classification_model.parameters()), lr=LR, weight_decay=1e-4)
        if IF_COS:
           scheduler = CosineAnnealingLR(optimizer, T_max=5)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, num_workers=0)
        val_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, num_workers=0)
        val_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, pin_memory=True, num_workers=0)

        # Count labels in training, validation, and test sets
        train_label_counts = count_labels(train_loader)
        val_label_counts = count_labels(val_loader)
        # test_label_counts = count_labels(test_loader)
        print(f"Training labels distribution: {train_label_counts}")
        print(f"Validation labels distribution: {val_label_counts}")
        # print(f"Test labels distribution: {test_label_counts}")

        for epoch in tqdm(range(EPOCH), desc=f'Fold {fold + 1}/{FOLD}'):
            transformer_model.train()
            classification_model.train()

            for data_batch, label_batch in train_loader:
                data_batch, label_batch = data_batch.to(device).double(), label_batch.to(device).long()
                optimizer.zero_grad()
                transformer_output, rec_loss= transformer_model(data_batch)
                # transformer_output= transformer_model(data_batch)

                predictions = classification_model(transformer_output)
                loss = W_CLS * criterion(predictions, label_batch) + W_REC * rec_loss
                loss.backward()
                optimizer.step()
            if IF_COS:
                scheduler.step()

            # Validation phase
            transformer_model.eval()
            classification_model.eval()
            with torch.no_grad():
                all_val_predictions = []
                all_val_labels = []
                for val_data_batch, val_label_batch in val_loader:
                    val_data_batch, val_label_batch = val_data_batch.to(device), val_label_batch.to(device)
                    val_transformer_output, _ = transformer_model(val_data_batch)
                    # val_transformer_output = transformer_model(val_data_batch)

                    val_predictions = classification_model(val_transformer_output)
                    all_val_predictions.append(val_predictions.cpu().numpy())
                    all_val_labels.append(val_label_batch.cpu().numpy())
                all_val_predictions = np.concatenate(all_val_predictions)
                all_val_labels = np.concatenate(all_val_labels)

                val_pred_classes = np.argmax(all_val_predictions, axis=1)
                val_accuracy = accuracy_score(all_val_labels, val_pred_classes)
                val_f1 = f1_score(all_val_labels, val_pred_classes, average='macro')
                val_precision_final = precision_score(all_val_labels, val_pred_classes, average= 'macro')

                print(f"Epoch {epoch + 1}/{EPOCH}, Fold {fold + 1}/{EPOCH} - Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}, Val Precision_final Score: {val_precision_final:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_model_wts = copy.deepcopy({'transformer': transformer_model.state_dict(), 'classification': classification_model.state_dict()})


        # Load best model weights for final testing
        if best_model_wts:
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            torch.save(best_model_wts, f'./ckpts/{DATASET_NAME}/{timestamp}_fold{fold + 1}.pt')
            transformer_model.load_state_dict(best_model_wts['transformer'])
            classification_model.load_state_dict(best_model_wts['classification'])

        # best_model_wts = torch.load(f'./ckpts/{DATASET_NAME}/fold_{fold + 1}.pt')
        transformer_model.load_state_dict(best_model_wts['transformer'])
        classification_model.load_state_dict(best_model_wts['classification'])
        transformer_model.eval()
        classification_model.eval()
        with torch.no_grad():
            all_test_predictions = []
            all_test_labels = []
            for test_data_batch, test_label_batch in val_loader:
                test_data_batch, test_label_batch = test_data_batch.to(device), test_label_batch.to(device)
                test_transformer_output, _ = transformer_model(test_data_batch)
                # test_transformer_output = transformer_model(test_data_batch)

                test_predictions = classification_model(test_transformer_output)
                all_test_predictions.append(test_predictions.cpu().numpy())
                all_test_labels.append(test_label_batch.cpu().numpy())
            all_test_predictions = np.concatenate(all_test_predictions)
            all_test_labels = np.concatenate(all_test_labels)

            test_pred_classes = np.argmax(all_test_predictions, axis=1)
            test_accuracy = accuracy_score(all_test_labels, test_pred_classes)
            test_f1 = f1_score(all_test_labels, test_pred_classes, average='macro')
            test_f1_all = f1_score(all_test_labels, test_pred_classes, average= None)
            test_precision = precision_score(all_test_labels, test_pred_classes, average= 'macro')
            test_ACC.append(test_accuracy)
            test_F1.append(test_f1)
            test_PRE.append(test_precision)

            # conf_matrix = confusion_matrix(all_test_labels, test_pred_classes)
            # print(all_test_labels[:10])

            # 假设类别标签为0到10
            # classes = [f'Class_{i}' for i in range(11)]

            # 保存预测值到文件
            # pred_df = pd.DataFrame({'Predictions': test_pred_classes})
            # pred_df.to_csv(f'pictures_paper/predictions_fold{fold+1}).csv', index=False)

            # 保存混淆矩阵到文件
            # conf_matrix_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
            # conf_matrix_df.to_csv(f'pictures_paper/confusion_matrix_fold{fold+1}.csv')

            print(f"Fold {fold + 1}/{FOLD} - Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, Test Precision Score: {test_precision:.4f}\nTest F1_All: {test_f1_all}\n\n")

    acc_mean, acc_std = compute_mean_std(test_ACC)
    f1_mean, f1_std = compute_mean_std(test_F1)
    pre_mean, pre_std = compute_mean_std(test_PRE)

    # 输出结果
    print(f"ACC: {acc_mean}±{acc_std}")
    print(f"F1: {f1_mean}±{f1_std}")
    print(f"Pre: {pre_mean}±{pre_std}")
    results = {
    "ACC": [test_ACC, acc_mean , acc_std],
    "F1": [test_F1, f1_mean , f1_std],
    "PRE": [test_PRE, pre_mean , pre_std]
}
    save_args_and_results(args, results, save_path='./results/few-train.json')

setup_seed(RANDOM_SEED)
data_produce()
train()
        



    
