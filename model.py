import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = F.softmax(self.linear(x), dim=1)
        return outputs

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        nb_hidden_units = 1000
        self.fc1 = nn.Linear(input_dim,nb_hidden_units)
        self.fc2 = nn.Linear(nb_hidden_units,nb_hidden_units)
        self.fc3 = nn.Linear(nb_hidden_units,10)
        #self.dropout = nn.Dropout(0.5)


    def forward(self,x):
        x = x.view(x.shape[0], -1)    #flatten the data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #y = torch.sigmoid(self.fc3(x))
        return x