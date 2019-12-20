import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv_down1 = nn.Conv2d(1, 5, 3, stride=2, padding=1) # [5, 16, 16]
        self.conv_down2 = nn.Conv2d(5, 10, 3, stride=2, padding=1) # [10, 8, 8]
        self.conv_down3 = nn.Conv2d(20, 5, 3, stride=2, padding=1) # [5, 4, 4]
        self.dense_down1 = nn.Linear(20*4*4, 400)

        self.dense_up1 = nn.Linear(400, 20*4*4)
        self.conv_up1 = nn.ConvTranspose2d(20, 10, 2, stride=2) # [5, 8, 8]
        self.conv_up2 = nn.ConvTranspose2d(10, 5, 2, stride=2) # [5, 16, 16]
        self.conv_up3 = nn.ConvTranspose2d(5, 1, 2, stride=2) # [1, 32, 32]

        self.optimizer = torch.optim.Adam(self.parameters())
    
    def encode(self, x):
        x = F.relu(self.conv_down1(x))
        x = F.relu(self.conv_down2(x))
        x = F.relu(self.conv_down3(x))
        x = F.relu(self.dense_down1(x.view(x.size(0), -1)))
        
        return x.view(x.size(0), -1)
    
    def decode(self, z):
        z = self.dense_up1(z)
        z = z.view(z.size(0), int(z.size(1)/16), 4, 4)
        
        z = F.leaky_relu(self.conv_up1(z))
        z = F.leaky_relu(self.conv_up2(z))
        z = F.leaky_relu(self.conv_up3(z))

        return z
    
    def forward(self, x):
        return self.decode(self.encode(x))