import os
import glob
import pandas as pd
import ast
from PIL import Image
import numpy as np
import json
from multiprocessing import Pool

INPUT_DIR = './input/quickdraw/'
OUTPUT_DIR = './input/quickdraw-dataset/train/'
VALID_DIR = './input/quickdraw-dataset/valid'
TEST_DIR = './input/quickdraw-dataset/test'


BUCKET_SIZE = 10

VALID_NUM_BUCKETS = 20
VALID_SIZE =  VALID_NUM_BUCKETS * BUCKET_SIZE


def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'input/simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

csv_files = glob.glob(os.path.join(INPUT_DIR, 'input/simplified/*.csv'))

## 1. Load csv
#for file in csv_files:
def process_csv(file):
    category = os.path.basename(file)[0:-4]
    print(f"loading {category}")
    folder = os.path.join(OUTPUT_DIR, category)
    os.makedirs(folder, exist_ok=True)

    df = pd.read_csv(file)
    df_train = df[0:-VALID_SIZE]
    df_valid = df[-VALID_SIZE:]

    strokes = df_train['drawing'].apply(ast.literal_eval)

    NUM_BUCKETS = len(df_train) // BUCKET_SIZE
    for bucket in range(NUM_BUCKETS):
        s = strokes[BUCKET_SIZE * bucket: BUCKET_SIZE * (bucket + 1)].tolist()

        bucket_filepath = os.path.join(folder, f"{bucket}.csv")
        with open(bucket_filepath, "w") as f:
            json.dump(s, f)


    ## valid
    folder = os.path.join(VALID_DIR, category)
    os.makedirs(folder, exist_ok=True)

    strokes = df_valid['drawing'].apply(ast.literal_eval)
    for bucket in range(VALID_NUM_BUCKETS):
        s = strokes[(BUCKET_SIZE * bucket): (BUCKET_SIZE * (bucket + 1))].tolist()
        bucket_filepath = os.path.join(folder, f"{bucket}.csv")
        with open(bucket_filepath, "w") as f:
            json.dump(s, f)
p = Pool(8)
p.map(process_csv, csv_files)
