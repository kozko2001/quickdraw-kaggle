from torchvision.transforms import ToTensor
from os.path import join, isdir
from os import listdir
from torch import randperm
from torch.utils.data.dataloader import DataLoader
import glob
import json
import cv2
import numpy as np
from PIL import Image
from random import random
import pandas as pd
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import ast


BASE_SIZE = 256

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

    def __init__(self, folder, mode="train", images_per_class = 1500, size = 80):
        self.folder = folder
        self.mode = mode
        self.size = size

        self.stats = self.loadStats(folder)
        self.images_per_class = images_per_class
        self.classes = self.stats[0]

    def loadStats(self, folder):
        classes = sorted([d for d in listdir(folder) if isdir(join(folder, d))])
        buckets_per_class = len(listdir(join(folder, classes[0])))

        sample_bucket = join(folder, classes[0], "0.csv")
        with open(sample_bucket, "r") as f:
            data = json.load(f)
            draws_per_bucket = len(data)

        return (classes, buckets_per_class, draws_per_bucket)

    def getStroke(self, i):
        classes, buckets, draws = self.stats
        draws_per_class = buckets * draws

        ## select class
        class_idx = i // draws_per_class
        i = i % draws_per_class

        ## select bucket
        bucket_idx = i // draws
        i = i % draws

        draw_idx = i

        filename = join(self.folder, classes[class_idx], f"{bucket_idx}.csv")
        with open(filename, "r") as f:
            list = json.load(f)
            return list[draw_idx], class_idx


    def __len__(self):
        classes, buckets, draws = self.stats

        if self.mode == "valid" or self.mode == "test":
            return len(classes) * buckets * draws
        else:
            return len(classes) * self.images_per_class


    def __getitem__(self, i):
        if self.mode == "train":
            classes, buckets, draws = self.stats
            i = int(random() * (len(classes) * buckets * draws))

        strokes, class_idx = self.getStroke(i)
        arr = draw_cv2(strokes, size = self.size) ## shape (size, size)
        arr = np.stack((arr,)*3, axis=0)
        arr = (arr / 255).astype('f')
        return torch.from_numpy(arr), class_idx

class TestDataset(Dataset):
    def __init__(self, csv, size):
        self.mode = "test"
        self.size = size

        self.df = pd.read_csv(csv)
        self.strokes = self.df['drawing'].apply(ast.literal_eval)
        self.ids = self.df["key_id"].tolist()

    def __len__(self):
        return len(self.df)

    def getStroke(self, i):
        return self.strokes[i], self.ids[i]

def train(folder, bs, images_per_class, size, num_workers = 4):
    folder = join(folder, "train")
    d = Dataset(folder, mode="train", images_per_class=images_per_class, size=size)

    return DataLoader(d, batch_size=bs, num_workers=num_workers, shuffle=False), d

def valid(folder, bs, size, num_workers = 4):
    folder = join(folder, "valid")
    d = Dataset(folder, mode="valid", images_per_class=None, size=size)

    return DataLoader(d, batch_size=bs, num_workers=num_workers, shuffle=False)

def test(test_csv, bs, size, num_workers = 4):
    d = TestDataset(test_csv, size)
    return DataLoader(d, batch_size=bs, num_workers=num_workers, shuffle=False)




if __name__ == "__main__":
    d = Dataset("/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-dataset/valid/", mode="valid")
    print(len(d))

    im, cls = d[0]
    print(im.shape, cls)
    print(im)
    im = to_pil_image(im)
    im.save("test.png")

    d = Dataset("/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-dataset/train/", mode="train")
    print(len(d))

    im,cls = d[0]
    print(im.shape, cls)

    import time
#    dl = valid("/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-dataset/", 1400, 80, num_workers=0)
    dl = train("/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-dataset/", 1400, 1500, 80, num_workers=8)
    print("valid num mini-batches", len(dl))
    start_time = time.time()
    s = next(iter(dl))
    for x in s:
        print(f" shape: {x.shape}")

    print(f"elapsed {time.time() - start_time}")

    d = TestDataset("/home/kozko/tmp/kaggle/quickdraw/input/quickdraw/test_simplified.csv", 80)
    print("len test dataset" , len(d))

    print(d[0])
