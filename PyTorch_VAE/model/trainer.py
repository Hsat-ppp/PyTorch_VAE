import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PyTorch_VAE.utils.utils import make_enum_loader, convert_np, imshow, INF
from PyTorch_VAE.model.settings import *

logger = logging.getLogger('trainer')


class ModelTrainerVAE(object):
    """
    model training class for VAE_joint.
    """
    def __init__(self, model, device, history_file_name, output_model_file_name):
        """init function.
        :param model:
        :param history_file_name:
        :param output_model_file_name:
        """
        self.model = model
        self.device = device
        self.train_loader = None
        self.history_file_name = history_file_name
        self.output_model_file_name = output_model_file_name
        self.criterion = None
        self.optimizer = None
        self.best_loss_value = INF
        self.test_loss_value = INF

    def set_loader(self, train_loader):
        """set data loader
        :param train_loader:
        :return:
        """
        self.train_loader = train_loader

    def set_criterion(self, criterion):
        """set criterion
        :param criterion:
        :return:
        """
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        """set optimizer.
        :param optimizer:
        :return:
        """
        self.optimizer = optimizer

    def set_evaluation_mode_with_pre_trained_weight(self, pre_trained_file=None):
        if pre_trained_file is None:
            pre_trained_file = self.output_model_file_name
        # model setting
        self.model.load_state_dict(torch.load(pre_trained_file))
        self.model.eval()  # evaluation mode
        self.model.to(self.device)

    def calculate_loss(self, inputs, labels):
        """calculate loss. please change this properly if criterion is changed.
        :param inputs:
        :param labels:
        :return:
        """
        mean, log_var, z, outputs = self.model(inputs)  # forward calculation
        loss, recon_loss, kld_loss = self.criterion(mean, log_var, inputs, outputs)  # calculate loss
        return loss, recon_loss, kld_loss

    def train(self, num_of_epochs, gamma_exp, is_quiet=False):
        """run training.
        :param num_of_epochs:
        :param gamma_exp:
        :param is_quiet:
        :return:
        """
        # settings
        self.model.to(self.device)
        if parallel_gpu:
            self.model = nn.DataParallel(self.model)
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma_exp)
        with open("history.csv", 'w') as f:
            print("epoch", "loss", "recon_loss", "kld_loss", "pred_loss", sep=',', file=f)

        # training process
        self.best_loss_value = INF
        logger.info('training start')
        for epoch in range(num_of_epochs):
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kld_loss = 0.0
            train_batches = 0
            logger.info('training phase in epoch {}'.format(epoch + 1))
            self.model.train()  # train mode
            enum_loader = make_enum_loader(self.train_loader, is_quiet)
            for i, data in enum_loader:  # load every batch
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # data は [inputs, labels] のリスト
                # reset gradients
                self.optimizer.zero_grad()
                # calculation
                loss, recon_loss, kld_loss = self.calculate_loss(inputs, labels)
                # accumulate loss
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kld_loss += kld_loss.item()
                train_batches += 1
                # update
                loss.backward()  # backpropagation
                self.optimizer.step()  # update parameters

            # output history
            logger.info('epoch {0} train_loss: {1}'.format(epoch + 1, train_loss / train_batches))
            with open("history.csv", 'a') as f:
                print(epoch + 1, train_loss / train_batches, train_recon_loss / train_batches,
                      train_kld_loss / train_batches, sep=',', file=f)

            # save the best model
            if self.best_loss_value > train_loss / train_batches:
                self.best_loss_value = train_loss / train_batches
                PATH = self.output_model_file_name
                if parallel_gpu:
                    torch.save(self.model.module.state_dict(),
                               PATH)  # see https://qiita.com/conta_/items/c3e173e891145e87e668
                else:
                    torch.save(self.model.state_dict(), PATH)

            # update learning rate
            scheduler.step()
            logger.info('learning rate adaption: {}'.format(scheduler.get_lr()[0]))

        logger.info('training end')

    def test(self, is_quiet, pre_trained_file=None):
        """run test.
        Note: In now VAE implementation, this function calculates loss for training data \
        since our model only treat with training data.
        :param is_quiet:
        :param pre_trained_file:
        :return:
        """
        self.set_evaluation_mode_with_pre_trained_weight(pre_trained_file)
        test_loss = 0.0
        test_batches = 0.0
        logger.info('running test...')

        # test
        enum_loader = make_enum_loader(self.train_loader, is_quiet)
        with torch.no_grad():
            for i, data in enum_loader:  # load every batch
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # data は [inputs, labels] のリスト
                loss, recon_loss, kld_loss = self.calculate_loss(inputs, labels)
                # accumulate loss
                test_loss += loss.item()
                test_batches += 1

        logger.info('test loss: {}'.format(test_loss / test_batches))
        logger.info('test end')
        self.best_loss_value = test_loss / test_batches

    def generate_image_from_data(self, img_data_path, num, pre_trained_file=None):
        self.set_evaluation_mode_with_pre_trained_weight(pre_trained_file)
        # load img
        images_numpy = np.load(img_data_path)
        transform = transforms.Compose([transforms.ToTensor(), ])
        img = (transform(images_numpy[num, :, :, :]).view(1, i_channel, width, height)).to(self.device)
        _, _, z = self.model.encode(img)
        _, _, outputs = self.model.decode(z)
        imshow(img[0], 'ex_in.png')
        imshow(outputs[0], 'ex_out.png')

    def generate_image_grid(self, is_quiet, pre_trained_file=None):
        self.set_evaluation_mode_with_pre_trained_weight(pre_trained_file)
        range_iter1 = np.arange(-1.0, 1.0, 0.2)
        range_iter2 = np.arange(-1.0, 1.0, 0.2)
        range_iter3 = np.arange(-1.0, 1.0, 0.2)
        col = len(range_iter1)
        row = len(range_iter2)
        enum_loader = make_enum_loader(range_iter3, is_quiet)
        for pp, axis3 in enum_loader:
            idx = 1
            plt.figure(figsize=(20, 20))
            for axis2 in range_iter2:
                for axis1 in range_iter1:
                    vec = torch.zeros(dim_latent).to(self.device)
                    vec[0] += axis1
                    vec[1] += axis2
                    vec[2] += axis3
                    img, _, _ = self.model.decode(vec)
                    img = convert_np(img[0])
                    plt.subplot(row, col, idx)
                    plt.imshow(img)
                    plt.axis('off')
                    idx += 1
            plt.savefig('test_grid_' + str(pp + 1) + '.png')
            plt.cla()
        return

    def show_latent_distribution(self, is_quiet, pre_trained_file=None):
        self.set_evaluation_mode_with_pre_trained_weight(pre_trained_file)
        latent_vec_batches = []
        enum_loader = make_enum_loader(self.train_loader, is_quiet)
        with torch.no_grad():
            for i, data in enum_loader:  # バッチ毎に読み込む
                inputs = data[0].to(self.device)  # data は [inputs, labels] のリスト
                mean, log_var, z = self.model.encode(inputs)
                latent_vec_batches.append(z)
        latent_vec_numpy = torch.cat(latent_vec_batches, dim=0).detach().to('cpu').numpy().copy()
        pg = sns.pairplot(pd.DataFrame(latent_vec_numpy), plot_kws={'alpha': 0.1}, )
        pg.savefig('latent_distribution.png')

    def generate_image_from_vector(self, z, pre_trained_file=None):
        self.set_evaluation_mode_with_pre_trained_weight(pre_trained_file)
        z_t = torch.from_numpy(z.copy().astype(np.float32)).clone()
        with torch.no_grad():
            img, _, _ = self.model.decode(z_t.to(self.device))
        for idx in range(img.shape[0]):
            imshow(img[idx], 'motor_img_history/' + str(idx) + '.png')
        return img

    def visualize_label_in_latent_space(self, pre_trained_file=None):
        import sklearn
        from sklearn.decomposition import PCA

        self.set_evaluation_mode_with_pre_trained_weight(pre_trained_file)

        # load data
        all_data = []
        all_data_target = []
        with torch.no_grad():
            for i, data in enumerate(self.train_loader):  # バッチ毎に読み込む
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # data は [inputs, labels] のリスト
                mean, log_var, z = self.model.encode(inputs)
                all_data.append(z.detach().to('cpu').numpy().copy())
                all_data_target.append(labels.detach().to('cpu').numpy().copy())
        all_data = np.concatenate(all_data)
        all_data_target = np.concatenate(all_data_target)
        # targetがnanである行を落とす
        all_data_ = all_data[~np.isnan(all_data_target).any(axis=1), :]
        all_data_target_ = all_data_target[~np.isnan(all_data_target).any(axis=1), :]
        pca = PCA()
        pca.fit(all_data)
        feature = pca.transform(all_data_)
        plt.figure()
        sc = plt.scatter(feature[:, 0], feature[:, 1], alpha=0.5,
                         vmin=-3, vmax=3, c=all_data_target_[:])
        plt.colorbar(sc)
        plt.xlabel("z_pc1")
        plt.ylabel("z_pc2")
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.savefig("PCA_latent_label.png")
        plt.cla()
        plt.clf()
        return pca
