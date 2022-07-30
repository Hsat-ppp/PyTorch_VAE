import torch
import torch.nn as nn
import torch.nn.functional as F

from PyTorch_VAE.model.settings import *


# ダウンサンプリング
class ds(nn.Module):
    # ネットワーク構造の定義
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(o_channel, o_channel, 3, padding=1) for _ in range(c_num)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(o_channel) for _ in range(c_num)])
        self.ds1 = nn.MaxPool2d(2, stride=2)  # down sampling to a half size

    # 順方向計算
    def forward(self, x):
        x_input = x.clone()
        for l, bn in zip(self.conv1, self.bn1):
            x = F.leaky_relu(bn(l(x)))
        x = x + x_input
        x = self.ds1(x)
        return x


# アップサンプリング
class us(nn.Module):
    # ネットワーク構造の定義
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(o_channel, o_channel, 3, padding=1) for _ in range(c_num)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(o_channel) for _ in range(c_num)])
        self.us1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # up sampling to a doubled size

    # 順方向計算
    def forward(self, x):
        x_input = x.clone()
        for l, bn in zip(self.conv1, self.bn1):
            x = F.leaky_relu(bn(l(x)))
        x = x + x_input
        x = self.us1(x)
        return x


class encoder(nn.Module):
    # ネットワーク構造の定義
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(i_channel, o_channel, 3, padding=1)
        self.down_sampling = nn.ModuleList([ds() for _ in range(s_num)])
        self.fc1 = nn.ModuleList([nn.Linear(o_channel*(height//(2**s_num))*(width//(2**s_num)), fc1_nodes)])
        self.output_fc = nn.Linear(fc1_nodes, dim_latent)

    # 順方向計算
    def forward(self, x):
        x = F.leaky_relu(self.input_conv(x))
        for l in self.down_sampling:
            x = l(x)
        x = x.view(-1, o_channel*(height//(2**s_num))*(width//(2**s_num)))  # flatten
        for l in self.fc1:
            x = F.leaky_relu(l(x))
        x = self.output_fc(x)
        return x


class decoder(nn.Module):
    # ネットワーク構造の定義
    def __init__(self):
        super().__init__()
        self.output_conv = nn.ConvTranspose2d(o_channel, i_channel, 3, padding=1)
        self.up_sampling = nn.ModuleList([us() for _ in range(s_num)])
        self.fc1 = nn.ModuleList([nn.Linear(fc1_nodes, o_channel*(height//(2**s_num))*(width//(2**s_num)))])
        self.input_fc = nn.Linear(dim_latent, fc1_nodes)

    # 順方向計算
    def forward(self, x):
        x = F.leaky_relu(self.input_fc(x))
        for l in self.fc1:
            x = F.leaky_relu(l(x))
        x = x.view(-1, o_channel, (height//(2**s_num)), (width//(2**s_num)))  # unflatten
        for l in self.up_sampling:
            x = l(x)
        x = torch.sigmoid(self.output_conv(x))
        return x


class encoder_vae(nn.Module):
    # ネットワーク構造の定義
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(i_channel, o_channel, 3, padding=1)
        self.down_sampling = nn.ModuleList([ds() for _ in range(s_num)])
        self.fc1 = nn.ModuleList([nn.Linear(o_channel*(height//(2**s_num))*(width//(2**s_num)), fc1_nodes)])
        self.mean_fc = nn.Linear(fc1_nodes, dim_latent)
        self.var_fc = nn.Linear(fc1_nodes, dim_latent)

    # 順方向計算
    def forward(self, x):
        x = F.leaky_relu(self.input_conv(x))
        for l in self.down_sampling:
            x = l(x)
        x = x.view(-1, o_channel*(height//(2**s_num))*(width//(2**s_num)))  # flatten
        for l in self.fc1:
            x = F.leaky_relu(l(x))
        mean = self.mean_fc(x)
        log_var = self.var_fc(x)
        return mean, log_var


class decoder_vae(nn.Module):
    # ネットワーク構造の定義
    def __init__(self):
        super().__init__()
        if loss_mode.lower() == 'cel':
            self.mean_output_conv = nn.ConvTranspose2d(o_channel, i_channel, 3, padding=1)
            self.var_output_conv = None
        elif loss_mode.lower() == 'mse':
            self.mean_output_conv = nn.ConvTranspose2d(o_channel, i_channel, 3, padding=1)
            self.var_output_conv = nn.ConvTranspose2d(o_channel, i_channel, 3, padding=1)
        self.up_sampling = nn.ModuleList([us() for _ in range(s_num)])
        self.fc1 = nn.ModuleList([nn.Linear(fc1_nodes, o_channel*(height//(2**s_num))*(width//(2**s_num)))])
        self.input_fc = nn.Linear(dim_latent, fc1_nodes)

    # 順方向計算
    def forward(self, x):
        x = F.leaky_relu(self.input_fc(x))
        for l in self.fc1:
            x = F.leaky_relu(l(x))
        x = x.view(-1, o_channel, (height//(2**s_num)), (width//(2**s_num)))  # unflatten
        for l in self.up_sampling:
            x = l(x)
        if loss_mode.lower() == 'cel':
            mean = F.sigmoid(self.mean_output_conv(x))
            log_var = None
        elif loss_mode.lower() == 'mse':
            mean = self.mean_output_conv(x)
            log_var = self.var_output_conv(x)
        return mean, log_var
