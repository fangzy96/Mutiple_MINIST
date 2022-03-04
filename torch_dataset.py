from torch.utils import data
from sklearn.preprocessing import MinMaxScaler
import torch
# import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# this is the dataset for torch
class dataset(data.Dataset):
    def __init__(self,data):
        self.data = data
        self.feature = self.data.iloc[:,1:].values
        self.feature = MinMaxScaler().fit_transform(self.feature)
        self.label = self.data.iloc[:, 0].values # label is the last column
        self.len = len(data)

    def __getitem__(self, item):
        feature = torch.tensor(self.feature[item],dtype=torch.float)
        label = torch.tensor(self.label[item],dtype=torch.int64)
        return feature, label

    def __len__(self):
        return self.len