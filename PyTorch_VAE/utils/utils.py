import logging
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn
import torch.utils.data
import tqdm

logger = logging.getLogger('utils')

# constants
INF = 1e+30
eps = 1e-3


def set_seed_num(seed_num):
    """set seed number.
    set seed number to python-random, numpy, torch (torch, cuda, backends), and os environ for reproductivity
    :param: seed number to set
    """
    if seed_num is None:
        seed_num = np.random.randint(0, (2 ** 30) - 1)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    with open('seed_num.csv', 'w') as f:
        print(seed_num, sep=',', file=f)
    return


def set_GPU():
    """get GPU settings and return.
    :return: obtained device
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('using processor: {}'.format(device))
    return device


def make_enum_loader(loader, is_quiet):
    """generate loader with enumerate and also tqdm progress bar.
    :param loader:
    :param is_quiet:
    :return: loader
    """
    if is_quiet:
        enum_loader = enumerate(loader)
    else:
        enum_loader = enumerate(tqdm.tqdm(loader))
    return enum_loader


def convert_np(img):
    npimg = img.detach().to('cpu').numpy().copy()
    # npimg = npimg * std + ave
    npimg = npimg * 255
    npimg[npimg < 0] = 0
    npimg[npimg > 255] = 255
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    return npimg


def imshow(img, name):
    npimg = convert_np(img)
    try:
        pilimg = Image.fromarray(npimg)
    except TypeError:
        logger.warning('Convert error. Trying 0/1 img...')
        pilimg = Image.fromarray(npimg[:, :, 0])
    pilimg.save(name)
