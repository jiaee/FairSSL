
import torch.nn.functional as F
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=24)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=16)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=8)
        self.pool  = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.pool(x).squeeze(-1)

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, hidden=128, dim_out=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim_out, bias=False),
        )
    def forward(self, x):
        return self.net(x)

class LinearHead(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.fc = nn.Linear(dim_in, 2)
    def forward(self, x):
        return self.fc(x)