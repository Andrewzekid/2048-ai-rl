import torch.nn as nn
import torch.nn.functional as F
class RLAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=16,kernel_size=(2,2),out_channels=256)
        self.conv2 = nn.Conv2d(in_channels=256,kernel_size=(2,2),out_channels=512)
        self.dense1 = nn.Linear(512 * 2 * 2,1024)
        self.dense2 = nn.Linear(1024,256)
        self.output = nn.Linear(256,4)
    def forward(self,x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c2 = c2.view(-1,512 *2 * 2)
        d1 = F.relu(self.dense1(c2))
        d2 = F.relu(self.dense2(d1))
        output = self.output(d2)
        return output
