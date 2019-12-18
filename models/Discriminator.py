import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 12, 3, stride=1, padding=1) # [12, 32, 32]
        self.bn1_1 = nn.BatchNorm2d(12)
        self.conv1_2 = nn.Conv2d(12, 20, 3, stride=2, padding=1) # [20, 16, 16]
        self.bn1_2 = nn.BatchNorm2d(20)

        self.conv2_1 = nn.Conv2d(20, 32, 3, stride=1, padding=1) # [32, 16, 16]
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 50, 3, stride=2, padding=1) # [50, 8, 8]
        self.bn2_2 = nn.BatchNorm2d(50)

        self.conv3_1 = nn.Conv2d(50, 62, 3, stride=1, padding=1) # [62, 8, 8]
        self.bn3_1 = nn.BatchNorm2d(62)
        self.conv3_2 = nn.Conv2d(62, 80, 3, stride=2, padding=1) # [80, 4, 4]
        self.bn3_2 = nn.BatchNorm2d(80)

        self.conv4_1 = nn.Conv2d(80, 40, 3, stride=1, padding=1) # [40, 4, 4]
        self.bn4_1 = nn.BatchNorm2d(40)
        self.conv4_2 = nn.Conv2d(40, 40, 3, stride=2, padding=1) # [40, 2, 2]
        self.bn4_2 = nn.BatchNorm2d(40)

        self.conv5_1 = nn.Conv2d(40, 20, 3, stride=1, padding=1) # [20, 2, 2]
        self.bn5_1 = nn.BatchNorm2d(20)
        self.conv5_2 = nn.Conv2d(20, 2, 3, stride=2, padding=1) # [2, 1, 1]

    def forward(self, x): 
        x = F.leaky_relu(self.conv1_1(x))
        x = self.bn1_1(x)
        x = F.leaky_relu(self.conv1_2(x))
        x = self.bn1_2(x)

        x = F.leaky_relu(self.conv2_1(x))
        x = self.bn2_1(x)
        x = F.leaky_relu(self.conv2_2(x))
        x = self.bn2_2(x)

        x = F.leaky_relu(self.conv3_1(x))
        x = self.bn3_1(x)
        x = F.leaky_relu(self.conv3_2(x))
        x = self.bn3_2(x)
        
        x = F.leaky_relu(self.conv4_1(x))
        x = self.bn4_1(x)
        x = F.leaky_relu(self.conv4_2(x))
        x = self.bn4_2(x)

        x = F.leaky_relu(self.conv5_1(x))
        x = self.bn5_1(x)
        x = F.relu(self.conv5_2(x))

        return x.view(x.size(0), 2)