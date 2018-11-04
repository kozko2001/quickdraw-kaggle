
from os.path import join, isdir, isfile
from os import listdir, makedirs
from shutil import move
import random

config = "small"
DATA_DIR = f"/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-{config}/"

if config == "small":
    VALID_SIZE = 100
else:
    VALID_SIZE = 500


train = join(DATA_DIR, "train")
valid = join(DATA_DIR, "valid")

onlyfolder = [f for f in listdir(train) if isdir(join(train, f))]

for f in onlyfolder:

    makedirs(join(valid, f), exist_ok=True)
    train_class = join(train, f)
    onlyfile = [f for f in listdir(train_class) if isfile(join(train_class, f))]
    random.shuffle(onlyfile)
    tomovefiles = onlyfile[0:VALID_SIZE]

    _from = [join(train_class, p) for p in tomovefiles]
    _to = [join(valid, f, p) for p in tomovefiles]

    for f,t in zip(_from, _to):
        print(f, t)
        move(f, t)
