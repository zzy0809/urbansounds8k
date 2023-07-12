import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class se_resnet(nn.Module):
    def __init__(self, num_classes, input_size,out_channels=256,embd_dim=256):
        super().__init__()
        self.layer1_0 = Conv1dReluBn(401, out_channels,padding=0)
        self.layer1_1 = Conv1dReluBn(401, out_channels, padding=0)

        self.layer2_0 = Conv1dReluBn(in_channels=79, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.layer2_1 = Conv1dReluBn(in_channels=200, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.layer3_0 = SE_Res2Block(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1, scale=4)
        self.layer3_1 = SE_Res2Block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, scale=4)

        self.layer4_0 = SE_Res2Block(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1, scale=4)
        self.layer4_1 = SE_Res2Block(in_channels=256,out_channels=256, kernel_size=3, stride=1, padding=1, scale=4)
        self.w1 = torch.nn.Parameter(torch.nn.init.constant_(torch.FloatTensor(768, 256), 1.))
        self.w2 = torch.nn.Parameter(torch.nn.init.constant_(torch.FloatTensor(768, 256), 1.))

        cat_channels = embd_dim * 3
        dog_channels = cat_channels * 2
        self.conv = nn.Conv1d(in_channels=768,out_channels=768,kernel_size=1,stride=1)
        self.pooling = AttentiveStatsPool(cat_channels, 128)
        self.bn1 = nn.BatchNorm1d(dog_channels,eps=1e-05)
        self.linear = nn.Linear(dog_channels, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)
        self.fc = nn.Linear(embd_dim, num_classes)
        self.fc1 = nn.Linear(num_classes, num_classes)


    def forward(self, x):
        out = torch.split(x,201, dim=1)
        out0_0 = out[1]
        out0_1 = out[0]
        # print(out0_1.size())
        out1_0 = self.layer1_0(out0_0)
        out1_1 = self.layer1_1(out0_1)
        # print(out1.size())
        out2_0 = self.layer2_0(out1_0)
        out2_1 = self.layer2_1(out1_1)

        out3_0 = self.layer3_0(out2_0) + out2_0
        out3_1 = self.layer3_1(out2_1) + out2_1

        out4_0 = self.layer4_0(out3_0) + out3_0
        out4_1 = self.layer4_1(out3_1) + out3_1

        out5_0 = torch.cat([out2_0, out3_0,out4_0], dim=1)
        out5_1 = torch.cat([out2_1, out3_1,out4_1], dim=1)
        out6_0 = F.relu(self.conv(out5_0))
        out6_1 = F.relu(self.conv(out5_1))
        out = self.w1 * out6_0 + self.w2 * out6_1
        out=nn.functional.dropout(out, p=0.6)
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))
        out = self.fc(out)
        return out
