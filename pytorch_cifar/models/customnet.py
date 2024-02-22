import torch.nn as nn
import torch.nn.functional as F

class ToyModelv1(nn.Module):
    def __init__(self):
        super(ToyModelv1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out



class ToyModelv2(nn.Module):
    def __init__(self):
        super(ToyModelv2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 1, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 2, 7),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(20)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(inplace=True),
            nn.Linear(2, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    