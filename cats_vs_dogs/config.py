from pathlib import Path

import torch

DATA_URL = "https://www.dropbox.com/s/gqdo90vhli893e0/data.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

IMAGE_SIZE_H = 224
IMAGE_SIZE_W = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 128
EPOCH_NUM = 5
LR = 3e-4

NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
