import numpy as np
import torch
import torch.nn as nn

from PyTorch_VAE.model.settings import *
from PyTorch_VAE.utils.utils import INF, eps


class AELossCEL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data, output_data):
        loss = -1.0 * torch.mean(torch.sum(input_data * torch.log(output_data+eps)
                                           + (1.0 - input_data) * torch.log((1.0 - output_data)+eps), dim=(1, 2, 3)))
        return loss


class AELossMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data, output_data):
        loss = 0.5 * torch.mean(torch.sum((input_data - output_data)**2, dim=(1, 2, 3)))
        return loss


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean, log_var, input_data, output_data, output_data_mean, output_data_log_var):
        if likelihood_x.lower() == 'bernoulli':
            log_likelihood = torch.mean(torch.sum(input_data * torch.log(output_data + eps)
                                                  + (1.0 - input_data) * torch.log((1.0 - output_data) + eps),
                                                  dim=(1, 2, 3)))
        elif likelihood_x.lower() == 'gaussian':
            output_data_var = torch.exp(output_data_log_var)
            log_likelihood = torch.mean(torch.sum(-0.5 * (torch.log(2.0 * torch.pi * output_data_var)
                                                          + ((input_data - output_data_mean) ** 2) / output_data_var),
                                                  dim=(1, 2, 3)))
        minus_kld = torch.mean(0.5 * torch.sum(1 + log_var - (mean * mean) - torch.exp(log_var), dim=1))
        loss = -1.0 * (lrec * log_likelihood + lkld * minus_kld)  # ELBOから損失に変換するため，符号を反転
        return loss, -1.0 * log_likelihood, -1.0 * minus_kld
