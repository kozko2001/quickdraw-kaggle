import numpy as np
import os
import pandas as pd
import cv2
import glob
from PIL import Image
import ast

INPUT_DIR = './input/quickdraw/'
OUTPUT_DIR = './input/quickdraw-fast/train/'
TEST_DIR = './input/quickdraw-fast/test/'

BASE_SIZE = 256
NCSVS = 100
NCATS = 340
NUM_IMAGES=25000

np.random.seed(seed=1987)

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'input/simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)

def draw_cv2(raw_strokes, size=128, lw=6, time_color=True):
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


csv_files = glob.glob(os.path.join(INPUT_DIR, 'input/simplified/*.csv'))

draw = lambda strokes: draw_cv2(strokes, lw=6)

print("TRAIN IMAGES")

## 1. Load csv
for file in csv_files:
    category = os.path.basename(file)[0:-4]
    print(category)
    folder = os.path.join(OUTPUT_DIR, category)
    os.makedirs(folder, exist_ok=True)

    df = pd.read_csv(file, nrows=NUM_IMAGES)
    ## 2. Load paths
    strokes = df['drawing'].apply(ast.literal_eval)

    ## 3. Draw them to numpy array
    images = strokes.apply(draw)
    images = images.values

    for i, image in enumerate(images):
        im = Image.fromarray(image)
        file = os.path.join(folder, f'{i}.png')
        im.save(file)


print("TEST IMAGES")
csv_file = os.path.join(INPUT_DIR, 'test_simplified.csv')

folder = os.path.join(TEST_DIR)
os.makedirs(folder, exist_ok=True)

df = pd.read_csv(csv_file)
df.head()

df['drawing']  = df['drawing'].apply(ast.literal_eval).apply(draw)

for index, row in df.iterrows():
    _id = row['key_id']
    file = os.path.join(folder, f'{_id}.png')
    im = Image.fromarray(row['drawing'])
    im.save(file)
