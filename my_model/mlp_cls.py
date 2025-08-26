import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)  
        # self.dropout2 = nn.Dropout(p = 0.2)  

        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        # x = self.relu(x)
        x = self.fc3(x)

        # x = self.softmax(x)  # 应用softmax激活函数
        
        return x