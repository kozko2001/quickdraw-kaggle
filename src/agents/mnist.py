"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np
from tqdm import tqdm

from glob import glob
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

from os.path import join, isfile
from os import listdir
import os

from utils.config import *

from finegrain import refine

#from dataset import Dataset
import dataset

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, mapk3
from scheduler.Cyclic import CyclicScheduler, adjust_learning_rate, get_learning_rate, CyclicLR
#from utils.misc import print_cuda_statistics
import logging

from math import log2
from losses.svm import SmoothSVM

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
        specific_classes = Config['specific_classes'] if 'specific_classes' in config else None


        # define loss
        loss_name = config.loss.name if "loss" in config else "cross_entropy"

        if loss_name == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        elif loss_name == "smooth_svm":
            tau = float(config.loss.tau)
            alpha = float(config.loss.alpha)
            k = int(config.loss.k)
            self.loss = SmoothSVM(config.num_classes, alpha, tau, 3)

        if specific_classes:
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

        self.acum_batches = int(config["acum_batches"]) if "acum_batches" in config else 0


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

        input_channels = int(config.input_channels) if "input_channels" in config else 1
        prob_drop_stroke = float(config.prob_drop_stroke) if "prob_drop_stroke" in config else 0.0
        self.input_channels = input_channels
        self.train_data_loader, self.train_dataset = dataset.train(data_root,
                                                                   bs,
                                                                   images_per_class,
                                                                   size,
                                                                   num_workers,
                                                                   input_channels=input_channels,
                                                                   prob_drop_stroke=prob_drop_stroke,
                                                                   specific_folders= specific_classes)
        self.valid_data_loader, self.valid_dataset = dataset.valid(data_root, bs, size, num_workers, input_channels=input_channels, specific_folders= specific_classes)


        if "scheduler" in self.config:
            if self.config.scheduler.type == "Cyclic":
                self.scheduler = CyclicScheduler(self.optimizer,
                                                 min_lr = config.learning_rate,
                                                 max_lr = config.scheduler.max_lr,
                                                 period = config.scheduler.period,
                                                 warm_start = config.scheduler.warm_start)
            elif self.config.scheduler.type == "Cyclic2":
                epoch_step = config.scheduler.step_size

                minibatches_in_epoch = int(len(self.train_dataset) / bs)
                steps_up = int(minibatches_in_epoch * epoch_step / 2.0)

                self.scheduler = CyclicLR(
                    self.optimizer,
                    base_lr= config.learning_rate,
                    max_lr = config.scheduler.max_lr,
                    step_size_up = steps_up,
                    mode = config.scheduler.mode,
                    gamma = config.scheduler.gamma,
                )
            elif self.config.scheduler.type == "LROnPlateau":
                patience = self.config.scheduler.patience
                factor = self.config.scheduler.factor
                min_lr = self.config.scheduler.min_lr
                threshold = self.config.scheduler.threshold
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=factor, min_lr=min_lr, threshold=threshold)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1)


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

    def run_validation_only(self):
        print("Validation oonlyyyy")

        self.validate()

    def run_create_concrete():
        path = "./configs/pairs/"
        config_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) if ".py" not in f and "generic" not in f and "__" not in f]

        for config_file in config_files:
            print("MMMMMMMMMMMMMMMMMMMMMMMMMM", config_file)
            _config = process_config(config_file, create_folders=False)

            data_root = join(_config.data_root, "valid")

            ds = dataset.Dataset(data_root, specific_folders=_config.specific_classes)
            classes = ds.classes

            ## Find checkpoint
            print(_config.exp_name)
            checkpoint_file = glob(f"./experiments/{_config.exp_name}*/checkpoints/killed*.pth")[0]
            self.load_checkpoint(checkpoint_file)
            print("CHECKPOINT!", checkpoint_file)

            input_channels = int(_config.input_channels)

            loader, test_ds = dataset.test("./input/quickdraw/test_simplified.csv", 200, _config.image_size, num_workers=8, input_channels=input_channels)

            self.validate_concrete("test", classes, loader, ds)

            ## valid
            ds = self.valid_dataset
            loader = self.valid_data_loader

            self.validate_concrete("valid", classes, loader, ds)


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

        count = 0

        def logging(output, target, loss, batch_idx):
            mapk3_metric = mapk3(output, target)
            self.mapk_train_avg.update(mapk3_metric)


            self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMapk3: {:.6f}'.format(
                self.current_epoch, batch_idx * len(data), len(self.train_data_loader.dataset),
                100. * batch_idx / len(self.train_data_loader), loss.item(), mapk3_metric))
            if self.summary_writer:
                iteration = self.current_epoch * 1000 + int(1000. * batch_idx / len(self.train_data_loader))

                self.summary_writer.add_scalar("train_loss", loss.item(), iteration)
                self.summary_writer.add_scalar("train_mapk", mapk3_metric, iteration)
                self.summary_writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], iteration)


        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            if count <= 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                count = self.acum_batches
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)

            loss = self.loss(output, target)

            ## Loggin and gradient accum
            if batch_idx % self.config.log_interval == 0:
                logging(output, target, loss, batch_idx)

            if self.acum_batches >= 2:
                loss = loss / self.acum_batches

            loss.backward()

            self.loss_train_avg.update(loss.item())


            self.current_iteration += 1
            count = count -1
            if self.scheduler and "step_batch" in dir(self.scheduler):
                self.scheduler.step_batch(self.current_iteration)
        self.save_checkpoint()


    def validate_concrete(self, mode, classes, loader, ds):
        self.model.eval()

        r = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                print(batch_idx, len(loader))
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                x = output.max(1, keepdim=True)[1].cpu().numpy()
                r = r + [x]

        df = pd.DataFrame(np.concatenate(r))
        df[0] = df[0].apply(lambda x: classes[x])
        df.to_csv(f"./concrete/{mode}-{classes[0]}-{classes[1]}.csv", header=False)



    def validate(self, step = False, calc_confusion=False):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        correct = 0
        loss_avg = AverageMeter()
        mapk_avg = AverageMeter()

        if calc_confusion:
            num_categories = 340
            confusion = np.zeros((num_categories, num_categories), dtype=np.float32)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                print(f"validation calc: {batch_idx} of {len(self.valid_data_loader)}")

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                if calc_confusion:
                    s_output = F.softmax(output, dim=1)
                    _, prediction_categories = s_output.topk(3, dim=1, sorted=True)

                    for bpc, bc in zip(prediction_categories[:, 0], target):
                        confusion[bpc, bc] += 1

                loss_value = self.loss(output, target).item()  # sum up batch loss
                loss_avg.update(loss_value)

                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#                pred = refine(pred.cpu().numpy(), "valid", batch_idx * self.config.batch_size)
#                pred = torch.from_numpy(pred).cuda()

                correct += pred.eq(target.view_as(pred)).sum().item()

                mapk3_metric = mapk3(output, target, "valid", batch_idx * self.config.batch_size)
                mapk_avg.update(mapk3_metric)

        if hasattr(self, "loss_train_avg") and hasattr(self, "mapk_train_avg"):
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


            if self.scheduler and "step" in dir(self.scheduler):
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(loss_avg.val)
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr > new_lr:
                    self.logger.info(f"Changing LR from {old_lr} to {new_lr}")


        self.model.train()

        if calc_confusion:
            for c in range(confusion.shape[0]):
                category_count = confusion[c, :].sum()
                if category_count != 0:
                    confusion[c, :] /= category_count

            np.save("confusion.np", confusion)

    def test(self):
        self.model.eval()


        test_data_loader, test_dataset = dataset.test(self.config.testcsv, self.config.batch_size, self.config.image_size, num_workers=8, input_channels=self.input_channels)

        self.load_checkpoint(self.config.checkpoint)

        cls_to_idx = {cls:idx for idx,cls in enumerate(self.train_dataset.classes)}
        print(cls_to_idx)
        idx_to_cls =  {cls_to_idx[c]:c for c in cls_to_idx}

        def row2string(r):
            v = [r[-1], r[-2], r[-3]]
            v = [v.item() for v in v]
            v = map(lambda v: v if v < 340 else 0, v)

            v = [idx_to_cls[v].replace(' ', '_') for v in v]

            return ' '.join(v)

        labels = []
        key_ids = []
        outputs = []

        with torch.no_grad():
            for idx, (data, target) in enumerate(test_data_loader):
                data = data.to(self.device)
                output = self.model(data)

                n = output.detach().cpu().numpy()
                outputs.append(n)

                order = np.argsort(n, 1)[:, -3:]
#                order = refine(order, "test", idx * self.config.batch_size)

                predicted_y = [row2string(o) for o in order]
                labels = labels + predicted_y
                key_ids = key_ids + target.numpy().tolist()

                if idx % 10 == 0:
                    print(f"{idx} of  {len(test_data_loader)}")

        import pickle

        with open('labels', 'wb') as fp:
            pickle.dump(key_ids, fp)
        with open('idx_to_cls', 'wb') as fp:
            pickle.dump(idx_to_cls, fp)

        _all = np.concatenate(outputs)
        np.save("output.npy", _all)

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
