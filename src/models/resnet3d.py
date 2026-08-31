"""Compact 3D ResNet for volumetric nodule classification.

Small on purpose: ResNet3D-10 has ~1.2M params and trains on a free-tier
Colab T4 (or even CPU for a smoke test). Bump `depth` to 18 once you have
a GPU and the full LIDC patch set.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv3x3x3(cin, cout, stride=1):
    return nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(cout, cout)
        self.bn2 = nn.BatchNorm3d(cout)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    def __init__(self, layers, widths, in_ch=1, num_classes=2, dropout=0.2):
        super().__init__()
        w0 = widths[0]
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, w0, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm3d(w0),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.inplanes = w0
        self.layer1 = self._make_layer(widths[0], layers[0], stride=1)
        self.layer2 = self._make_layer(widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(widths[3], layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        layers = [BasicBlock3D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


_DEPTHS = {
    10: [1, 1, 1, 1],
    18: [2, 2, 2, 2],
}


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    layers = _DEPTHS[int(m.get("depth", 10))]
    return ResNet3D(
        layers=layers,
        widths=m.get("widths", [16, 32, 64, 128]),
        in_ch=1,
        num_classes=2,
        dropout=float(m.get("dropout", 0.2)),
    )
