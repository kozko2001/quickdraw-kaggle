from torchvision.transforms import ToTensor
from os.path import join, isdir, isfile
from os import listdir
from torch import randperm
from torch.utils.data.dataloader import DataLoader
import glob
import json
import cv2
import numpy as np
from PIL import Image
import random
import pandas as pd
from torchvision.transforms.functional import to_tensor, to_pil_image, normalize

import torch
import ast


BASE_SIZE = 256

mean = [0.09934586, 0.09934586, 0.09934586]
std = [0.27682555, 0.27682555, 0.27682555]


def draw_cv2(raw_strokes, size, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img



class Dataset():

    def __init__(self, folder, mode="train", images_per_class = 1500, size = 80, input_channels = 1):
        self.folder = folder
        self.mode = mode
        self.size = size

        self.stats = self.loadStats(folder)
        self.images_per_class = images_per_class
        self.classes = self.stats[0]
        self.r = None
        self.random = []
        self.draws_per_class = None
        self.input_channels = input_channels

    def loadStats(self, folder):
        classes = sorted([d for d in listdir(folder) if isdir(join(folder, d))])
        def filesInFolder(folder):
            return [f for f  in listdir(folder) if isfile(join(folder, f))]

        buckets_per_class = {c: len(filesInFolder(join(folder, c))) for c in classes }

        sample_bucket = join(folder, classes[0], "0.csv")
        with open(sample_bucket, "r") as f:
            data = json.load(f)
            draws_per_bucket = len(data)

        total_buckets = sum([buckets_per_class[k] for k in buckets_per_class])

        return (classes, buckets_per_class, draws_per_bucket, total_buckets)

    def getStroke(self, i):
        classes, buckets, draws, total_buckets = self.stats
        if not self.draws_per_class:
            self.draws_per_class = [buckets[k] * draws for k in buckets]
        draws_per_class = self.draws_per_class

        if not self.r:
            r = []
            sum = 0
            for dr in draws_per_class:
                r.append(sum)
                sum = sum + dr
            r.append(sum)
            self.r = r
        else:
            r = self.r

        ## select class
        p = [i < r for r in r].index(True) -1
        class_idx = p

        i = i - r[class_idx]

        bucket_idx = i // draws
        draw_idx = i % draws

        filename = join(self.folder, classes[class_idx], f"{bucket_idx}.csv")
        with open(filename, "r") as f:
            list = json.load(f)
            return list[draw_idx], class_idx


    def __len__(self):
        classes, buckets, draws, total_buckets = self.stats

        if self.mode == "valid" or self.mode == "test":
            return total_buckets * draws
        else:
            return len(classes) * self.images_per_class


    def __getitem__(self, i):
        if self.mode == "train":
            classes, buckets, draws, total_buckets = self.stats
            if len(self.random) == 0:
                self.random = random.sample(range(total_buckets * draws), 200000)
                print("GENERATING...")
            i = self.random.pop()

        strokes, class_idx = self.getStroke(i)
        arr = draw_cv2(strokes, size = self.size) ## shape (size, size)
        if self.input_channels > 1:
            arr = np.stack((arr,)*3, axis=0)
        arr = (arr / 255).astype('f')
        t = torch.from_numpy(arr)
        if self.input_channels == 1:
            t = torch.unsqueeze(t, 0)

#        t = normalize(t, mean, std)
        return t, class_idx

class TestDataset(Dataset):
    def __init__(self, csv, size, input_channels = 1):
        self.mode = "test"
        self.size = size
        self.input_channels = input_channels

        self.df = pd.read_csv(csv)
        self.strokes = self.df['drawing'].apply(ast.literal_eval)
        self.ids = self.df["key_id"].tolist()

    def __len__(self):
        return len(self.df)

    def getStroke(self, i):
        return self.strokes[i], self.ids[i]

def train(folder, bs, images_per_class, size, num_workers = 4, input_channels = 1):
    folder = join(folder, "train")
    d = Dataset(folder, mode="train", images_per_class=images_per_class, size=size, input_channels = input_channels)

    return DataLoader(d, batch_size=bs, num_workers=num_workers, shuffle=False), d

def valid(folder, bs, size, num_workers = 4, input_channels = 1):
    folder = join(folder, "valid")
    d = Dataset(folder, mode="valid", images_per_class=None, size=size, input_channels = input_channels)

    return DataLoader(d, batch_size=bs, num_workers=num_workers, shuffle=False)

def test(test_csv, bs, size, num_workers = 4, input_channels = 1):
    d = TestDataset(test_csv, size, input_channels = input_channels)
    return DataLoader(d, batch_size=bs, num_workers=num_workers, shuffle=False)

def getImagesStats(folder):
    d = Dataset(folder, 1000, 101, 80)

    means = []
    stds = []
    for base in range(0, 30):
        images = [d[(base * 1000) + i][0] for i in range(1000)] ## get 2000 images

        subimgs = images
        xx = np.stack(subimgs)


        means.append(xx.mean(axis=(0, 2, 3)))
        stds.append(xx.std(axis=(0, 2, 3)))

    nn = np.stack(means)
    ss = np.stack(stds)
    print(nn.mean(axis=0))
    print(ss.mean(axis=0))

    return nn, ss




if __name__ == "__main__":
#    valid_folder = "/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-dataset/valid/"
#    getImagesStats(valid_folder)
    d = Dataset("/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-dataset/valid/", mode="valid")
    print(len(d))

    im, cls = d[0]
    im = torch.unsqueeze(im, 0)
    print(im.shape)
    # print(im.shape, cls)
    # print(im)
    # im = to_pil_image(im)
    # im.save("test.png")

    # dl,d  = train("/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-dataset/", 1400, 1500, 80, num_workers=8)
    # print(len(d))

    # im,cls = d[0]
    # print(im.shape, cls)

    # import time
    # print("valid num mini-batches", len(dl))
    # start_time = time.time()
    # s = next(iter(dl))
    # for x in s:
    #     print(f" shape: {x.shape}")

    # print(f"elapsed {time.time() - start_time}")

    # d = TestDataset("/home/kozko/tmp/kaggle/quickdraw/input/quickdraw/test_simplified.csv", 80)
    # print("len test dataset" , len(d))

    # print(d[0])
