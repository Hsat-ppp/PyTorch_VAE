import torch
import torch.nn as nn

from PyTorch_VAE.model import VAE_parts
from PyTorch_VAE.model.settings import *


class AutoEncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAE_parts.encoder()
        self.decoder = VAE_parts.decoder()

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return z, x


class VariationalAutoEncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAE_parts.encoder_vae()
        self.decoder = VAE_parts.decoder_vae()

    def encode(self, x):
        mean, log_var = self.encoder(x)
        z = mean + torch.exp(log_var / 2.0)*(torch.randn(log_var.shape).to(log_var.get_device()))
        return mean, log_var, z

    def decode(self, z):
        mean_dec, log_var_dec = self.decoder(z)
        if likelihood_x.lower() == 'bernoulli':
            x = torch.sigmoid(mean_dec)
        elif likelihood_x.lower() == 'gaussian':
            x = mean_dec + torch.exp(log_var_dec / 2.0)*(torch.randn(log_var_dec.shape).to(log_var_dec.get_device()))
        return mean_dec, log_var_dec, x

    def forward(self, x):
        mean, log_var, z = self.encode(x)
        mean_dec, log_var_dec, x = self.decode(z)
        return mean, log_var, z, x, mean_dec, log_var_dec
