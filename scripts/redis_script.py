import redis
import pickle
database = redis.Redis(host='localhost', port=6379)
from PIL import Image
import os
from tqdm import tqdm

img_path = '/remote-home/share/DATA/HISMIL/5_folder/0/train/pos'
for i in tqdm(os.listdir(img_path)):
    for j in os.listdir(os.path.join(img_path,i)):
        if j.endswith('.png'):
            img = Image.open(os.path.join(img_path,i,j)).convert('RGB')
            image_data = pickle.dumps(img)
            key = j
            database.set(key, image_data)
img_path = '/remote-home/share/DATA/HISMIL/5_folder/0/train/neg'
for i in tqdm(os.listdir(img_path)):
    for j in os.listdir(os.path.join(img_path,i)):
        if j.endswith('.png'):
            img = Image.open(os.path.join(img_path,i,j)).convert('RGB')
            image_data = pickle.dumps(img)
            key = j
            database.set(key, image_data)
img_path = '/remote-home/share/DATA/HISMIL/5_folder/0/test/pos'
for i in tqdm(os.listdir(img_path)):
    for j in os.listdir(os.path.join(img_path,i)):
        if j.endswith('.png'):
            img = Image.open(os.path.join(img_path,i,j)).convert('RGB')
            image_data = pickle.dumps(img)
            key = j
            database.set(key, image_data)
img_path = '/remote-home/share/DATA/HISMIL/5_folder/0/test/neg'
for i in tqdm(os.listdir(img_path)):
    for j in os.listdir(os.path.join(img_path,i)):
        if j.endswith('.png'):
            img = Image.open(os.path.join(img_path,i,j)).convert('RGB')
            image_data = pickle.dumps(img)
            key = j
            database.set(key, image_data)