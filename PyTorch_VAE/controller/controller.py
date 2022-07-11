import argparse
import json
import logging

import torch
import torch.optim as optim

import PyTorch_VAE.model.VAE_model
import PyTorch_VAE.model.trainer
import PyTorch_VAE.model.data_treatment
import PyTorch_VAE.model.loss_functions
from PyTorch_VAE.utils.utils import set_GPU, set_seed_num

logger = logging.getLogger('controller')


def set_argparser_options():
    """set argparser options
    :return: ArgumentParser object
    """
    parser = argparse.ArgumentParser(description='''
                                    VAE implementation
                                    ''')
    parser.add_argument('-e', '--num_of_epochs', default=10, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        help='batch size.')
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float,
                        help='initial learning rate.')
    parser.add_argument('-g', '--gamma_exp', default=0.95, type=float,
                        help='value for gamma of exp lr reducer.')
    parser.add_argument('-s', '--seed_num', type=int,
                        help='seed number for reproduction.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='if set, we disables progress bar.')
    return parser


def check_args(args):
    """check args. if something is wrong, assert error will arise.
    :param args: args to be checked
    :return:
    """
    assert args.num_of_epochs >= 1, 'Option "num_of_epochs" need to be positive. ' \
                                    'Got: {}'.format(args.num_of_epochs)
    assert args.batch_size >= 1, 'Option "batch_size" need to be positive. ' \
                                 'Got: {}'.format(args.batch_size)
    assert 0.0 < args.learning_rate < 1.0, 'Option "learning_rate" need to be between 0.0 to 1.0. ' \
                                           'Got: {}'.format(args.learning_rate)
    assert 0.0 < args.gamma_exp < 1.0, 'Option "gamma_exp" need to be between 0.0 to 1.0. ' \
                                           'Got: {}'.format(args.gamma_exp)


def evaluate_model():
    """evaluate a model.
    including:
        :parse args
        :obtain model from model folder
        :load data
        :define criterion and optimization strategy
        :run training
    :return:
    """
    # get args
    parser = set_argparser_options()
    args = parser.parse_args()
    check_args(args)
    # save args
    logger.info('args options')
    logger.info(args.__dict__)
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # set seed num
    set_seed_num(args.seed_num)

    # define model and trainer
    device = set_GPU()
    model = PyTorch_VAE.model.VAE_model.VariationalAutoEncoderCNN()
    trainer = PyTorch_VAE.model.trainer.ModelTrainerVAE(model, device, 'history.csv', 'best.pth')

    # load data
    train_loader = PyTorch_VAE.model.data_treatment.load_data(args.batch_size)
    trainer.set_loader(train_loader)

    # set criterion
    trainer.set_criterion(PyTorch_VAE.model.loss_functions.VAELossMSE())
    trainer.set_optimizer(optim.Adam(model.parameters(), lr=args.learning_rate))

    # run training and test
    trainer.train(args.num_of_epochs, args.gamma_exp, args.quiet)


def visualize_training_result():
    # get args
    parser = set_argparser_options()
    args = parser.parse_args()
    check_args(args)

    # set seed num
    set_seed_num(args.seed_num)

    # define model and trainer
    device = set_GPU()
    model = PyTorch_VAE.model.VAE_model.VariationalAutoEncoderCNN()
    trainer = PyTorch_VAE.model.trainer.ModelTrainerVAE(model, device, 'history.csv', 'best.pth')

    # load data
    train_loader = PyTorch_VAE.model.data_treatment.load_data(args.batch_size)
    trainer.set_loader(train_loader)

    # set criterion
    trainer.set_criterion(PyTorch_VAE.model.loss_functions.VAELossMSE())

    # visualizations
    trainer.generate_image_grid(args.quiet)
    trainer.show_latent_distribution(args.quiet)
    trainer.visualize_label_in_latent_space()
