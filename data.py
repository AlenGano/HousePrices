import torch 

class HousePriceDataSetTrain:
    def __init__ (self,dataset_path):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(dataset_path)

    def __len__ (self):
        return len(self.data)

    def __getitem__ (self,i):
        tensor = self.data.iloc[i].values
        return torch.tensor(tensor[:-1]),torch.tensor(tensor[-1])

class HousePriceDataSetTest:
    def __init__ (self,dataset_path):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(dataset_path)

    def __len__ (self):
        return len(self.data)

    def __getitem__ (self,i):
        tensor = self.data.iloc[i].values
        return torch.tensor(tensor)