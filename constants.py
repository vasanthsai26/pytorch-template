import os
from pathlib import Path

NUM_WORKERS = os.cpu_count() 
IMG_SIZE = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 196
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SCALE = 1.1 

#DIR CONFIGS
DATA_PATH = Path(os.path.join(os.getcwd(),"stanford-cars-dataset"))
META_PATH = Path(os.path.join(DATA_PATH,"car_devkit","devkit"))
CARS_TRAIN_PATH = Path(os.path.join(DATA_PATH,"cars_train","cars_train"))
CARS_TEST_PATH = Path(os.path.join(DATA_PATH,"cars_test","cars_test")) 

URL = "https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset"





