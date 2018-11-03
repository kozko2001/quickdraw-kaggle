
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import ToTensor
from os.path import join
from torch import randperm


class Dataset():

    def __init__(self, config):
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"] if "num_workers" in config else 8
        self.root = config["data_root"]
        self.pct_data = config["pct_data"] if "pct_data" in config else 1.0

    def createDataloader(self, folder):
        data = ImageFolder(folder, transform=ToTensor())

        indices = randperm(len(data))
        indices = indices[0: int(len(data) * self.pct_data)]
        data = Subset(data, indices) ## Make dataset smaller

        return DataLoader(data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def train(self):
        folder = join(self.root, "train")
        if self.config.use_valid:
            folder = join(self.root, "valid")

        return self.createDataloader(folder)

    def validate(self):
        folder = join(self.root, "valid")
        return self.createDataloader(folder)

    def test(self):
        folder = join(self.root, "test")
        return self.createDataloader(folder)



if __name__ == "__main__":

    config = {
        "data_root": "/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-fast/",
        "batch_size": 4,
        "num_workers": 8,
        "pct_data": 0.05,
    }

    d = Dataset(config)
    train = d.train()
    print(train)
