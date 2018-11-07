"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

from os.path import join
import os

#from dataset import Dataset
import dataset

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, mapk3
#from utils.misc import print_cuda_statistics
import logging

from math import log2

class MnistAgent:

    def __init__(self, config):

        self.config = config
        self.logger = logging.getLogger("Agent")
        # define models
        self.model = config.model


        # define data_loader
        bs = config.batch_size
        num_workers = config["num_workers"] if "num_workers" in config else 8
        data_root = config.data_root
        size = config.image_size
        images_per_class = config.images_per_class

        self.train_data_loader, self.train_dataset = dataset.train(data_root, bs, images_per_class, size, num_workers)
        self.valid_data_loader = dataset.valid(data_root, bs, size, num_workers)

        # define loss
        self.loss = nn.CrossEntropyLoss()

        # define optimizer
        if self.config.optim == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.config.learning_rate,
                                       nesterov=self.config.nesterov,
                                       momentum=self.config.momentum,
                                       weight_decay=self.config.w_decay)
        elif self.config.optim == "ADAM":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.99))

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1)


        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        self.summary_writer = None

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            #            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        if self.config.checkpoint:
            self.load_checkpoint(self.config.checkpoint)

        # Summary Writer
        run_name = f'{config.exp_name}'
        self.summary_writer = SummaryWriter(self.config.summary_dir, run_name)

    def load_checkpoint(self, filename = "checkpoint.pth.tar"):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint)


    def save_checkpoint(self, file_name=None, is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        if not file_name:
            file_name = f"{self.current_epoch}_{self.current_iteration}.pth"
        PATH = join(self.config.checkpoint_dir, file_name)
        if not self.config.dry_run:
            torch.save(self.model.state_dict(), PATH)
            self.logger.info(f"model saved in {PATH}")

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.finalize()
            self.logger.info("You have entered CTRL+C.. Wait to finalize")


    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()

            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.loss_train_avg = AverageMeter()
        self.mapk_train_avg = AverageMeter()

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)

            loss.backward()
            self.optimizer.step()

            self.loss_train_avg.update(loss.item())

            ## Logging
            if batch_idx % self.config.log_interval == 0:
                mapk3_metric = mapk3(output, target)
                self.mapk_train_avg.update(mapk3_metric)


                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMapk3: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.train_data_loader.dataset),
                           100. * batch_idx / len(self.train_data_loader), loss.item(), mapk3_metric))
                if self.summary_writer:
                    iteration = self.current_epoch * 1000 + int(1000. * batch_idx / len(self.train_data_loader))

                    self.summary_writer.add_scalar("train_loss", loss.item(), iteration)
                    self.summary_writer.add_scalar("train_mapk", mapk3_metric, iteration)

            self.current_iteration += 1
        self.save_checkpoint()

    def validate(self, step = False):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        correct = 0
        loss_avg = AverageMeter()
        mapk_avg = AverageMeter()

        with torch.no_grad():
            for data, target in self.valid_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                loss_value = self.loss(output, target).item()  # sum up batch loss
                loss_avg.update(loss_value)

                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                mapk3_metric = mapk3(output, target)
                mapk_avg.update(mapk3_metric)

        self.logger.info("Epoch, LossV, LossT, Mapk3V, Mapk3T")
        self.logger.info("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {}".format(
                         self.current_epoch,
                         loss_avg.val,
                         self.loss_train_avg.val,
                         mapk_avg.val,
                         self.mapk_train_avg.val,
                         step))


        if self.summary_writer:
            iteration = (self.current_epoch + 1) * 1000

            self.summary_writer.add_scalar("valid_loss", loss_avg.val, iteration)
            self.summary_writer.add_scalar("valid_mapk", mapk_avg.val, iteration)

        if self.scheduler:
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(loss_avg.val)
            new_lr = self.optimizer.param_groups[0]['lr']
            if old_lr > new_lr:
                self.logger.info(f"Changing LR from {old_lr} to {new_lr}")

        self.model.train()

    def test(self):
        self.model.eval()


        test_data_loader = dataset.test(self.config.testcsv, self.config.batch_size, self.config.image_size, num_workers=8)

        self.load_checkpoint(self.config.checkpoint)

        cls_to_idx = {cls:idx for idx,cls in enumerate(self.train_dataset.classes)}
        print(cls_to_idx)
        idx_to_cls =  {cls_to_idx[c]:c for c in cls_to_idx}

        def row2string(r):
            v = [r[-1], r[-2], r[-3]]
            v = [v.item() for v in v]
            v = [idx_to_cls[v].replace(' ', '_') for v in v]

            return ' '.join(v)

        labels = []
        key_ids = []

        with torch.no_grad():
            for idx, (data, target) in enumerate(test_data_loader):
                data = data.to(self.device)
                output = self.model(data)

                n = output.detach().cpu().numpy()

                order = np.argsort(n, 1)[:, -3:]

                predicted_y = [row2string(o) for o in order]
                labels = labels + predicted_y
                key_ids = key_ids + target.numpy().tolist()

                if idx % 10 == 0:
                    print(f"{idx} of  {len(test_data_loader)}")



        d = {'key_id': key_ids,
             'word': labels}
        df = pd.DataFrame.from_dict(d)
        df.to_csv('submission.csv', index=False)


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("saving before finalize")
        file_name = f"killed_{self.current_epoch}_{self.current_iteration}.pth"
        self.save_checkpoint(file_name=file_name)
        self.logger.info(f"saved! with name {file_name}")
