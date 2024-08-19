from torch.utils import data
class mydataSet(data.dataset.Dataset):
    def __init__(self, data, label):
        super(mydataSet, self).__init__()
        self.data = data
        self.label = label
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]    
        return x, y
    def __len__(self):
        return len(self.data)