import.nn as nn

class HousePriceModel:
    def __init__ (input_dim,h_1,h_2,h_3,output_dim = 1):
        self.lin1 = nn.Linear(input_dim,h1)
        self.lin2 = nn.Linear(h1,h2)
        self.lin3 = nn.Linear(h2,h3)
        self.lin4 = nn.Linear(h3,output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.relu(x)
        return x 