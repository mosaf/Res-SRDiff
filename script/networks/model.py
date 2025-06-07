# model.py

import torch
import torch.nn as nn
from unet import UNetModelSwin  # Assuming this file contains the UNet model


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class SuperResolutionModel(nn.Module):
    def __init__(self, model_name, device):
        super(SuperResolutionModel, self).__init__()
        self.model_name = model_name.lower()
        self.device = device
        self.model = self._build_model().to(device)
        self.model.apply(_weights_init)

    def _build_model(self):
        if self.model_name == 'unet':
            return UNetModelSwin()  # Replace with actual initialization of UNet
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def forward(self, x):
        return self.model(x)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
