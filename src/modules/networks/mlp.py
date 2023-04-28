import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, din=None, dout=None) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(din, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,dout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NET_G(nn.Module):
    def __init__(self, din=None, dout=None) -> None:
        super(NET_G, self).__init__()
        self.fc1 = nn.Linear(din, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,dout)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NET_D(nn.Module):
    def __init__(self, din=None, dout=None) -> None:
        super(NET_D, self).__init__()
        self.fc1 = nn.Linear(din, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,dout)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class RWDpredictor(nn.Module):
    def __init__(self, input_shape, args) -> None:
        super(RWDpredictor, self).__init__()

        self.args = args
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x