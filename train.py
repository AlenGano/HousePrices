from data import HousePriceDataSetTrain
from data import HousePriceDataSetTest
from model import HousePriceModel
from torch.utils.data import DataLoader

Train_DataSet = HousePriceDataSetTrain('train_encoded.csv')
Test_DataSet = HousePriceDataSetTest('test_encoded.csv')

train_DataLoader = DataLoader(Train_DataSet, batch_size = 16, shuffle = True)
test_DataLoader = DataLoader(Test_DataSet, batch_size = 16)

num_epochs = 50 
lr = 0.003
model = HousePriceModel(
    Train_DataSet[0][0].shape[0],512,128,32
)
optizer = torch.optim.Adam(model.parameters(),lr = lr)
loss_func = torch.nn.MSELoss()

for i in range(num_epochs):
    for x,y in train_DataLoader:
        y_pred = model(x)
        loss = loss_func(y_pred,y)
        optimzer.zero_grad()
        loss.backward()
        optimizer.step()
        print()