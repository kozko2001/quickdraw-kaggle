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


from os.path import join

from dataset import Dataset

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, mapk3
#from utils.misc import print_cuda_statistics
import logging

import calendar
import time
from time import gmtime, strftime

class MnistAgent:

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        # define models
        self.model = config.model

        # define data_loader
        self.dataset = Dataset(config)

        self.train_data_loader = self.dataset.train()
        self.valid_data_loader = self.dataset.validate()

        # define loss
        self.loss = nn.CrossEntropyLoss()

        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.99))

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1, verbose=True)

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
        self.load_checkpoint()

        # Summary Writer
        if not self.config.dry_run:
            run_name = f'{config.exp_name}_{strftime("%Y-%m-%d %H:%M:%S", gmtime())}'
            self.summary_writer = SummaryWriter(self.config.summary_dir, run_name)

    def load_checkpoint(self, file_name = "checkpoint.pth.tar"):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name=None, is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        if not file_name:
            file_name = f"{calendar.timegm(time.gmtime())}.pth"

        PATH = join(self.config.checkpoint_dir, file_name)
        if not self.config.dry_run:
            torch.save(self.model.state_dict(), PATH)

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
                    self.summary_writer.add_scalar("train_loss", loss.item(), self.current_iteration)
                    self.summary_writer.add_scalar("train_mapk", mapk3_metric, self.current_iteration)


            self.current_iteration += 1
        self.save_checkpoint(f"")

    def validate(self):
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

        self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), mapk3: {:.4f}, TrainLoss: {:.4f}, TrainMapk3: {:.4f}\n'.format(
            loss_avg.val, correct, len(self.valid_data_loader.dataset),
            100. * correct / len(self.valid_data_loader.dataset), mapk_avg.val, self.loss_train_avg.val, self.mapk_train_avg.val))

        if self.summary_writer:
            self.summary_writer.add_scalar("valid_loss", loss_avg.val, self.current_epoch)
            self.summary_writer.add_scalar("valid_mapk", mapk_avg.val, self.current_epoch)

        if self.scheduler:
            self.scheduler.step(loss_avg.val)


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("saving before finalize")
        file_name = f"killed_{calendar.timegm(time.gmtime())}.pth"
        self.save_checkpoint(file_name=file_name)
        self.logger.info(f"saved! with name {file_name}")
