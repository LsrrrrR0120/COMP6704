"""CIFAR-style ResNets (He et al., 2016). The stem is `conv1` and the classifier `fc`,
which are the AdamW-handled input/output layers for Muon/Shampoo."""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    AUX_PREFIXES = ("conv1.", "fc.")

    def __init__(self, blocks=(2, 2, 2, 2), widths=(64, 128, 256, 512), num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        layers, in_ch = [], widths[0]
        for i, (nb, w) in enumerate(zip(blocks, widths)):
            for b in range(nb):
                layers.append(BasicBlock(in_ch, w, 2 if (b == 0 and i > 0) else 1))
                in_ch = w
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def build_resnet(name):
    if name == "resnet18":
        return ResNet((2, 2, 2, 2), (64, 128, 256, 512))
    if name == "resnet_small":  # ~1.2M params, for fast learning-rate sweeps
        return ResNet((1, 1, 1, 1), (32, 64, 128, 256))
    raise ValueError(name)
