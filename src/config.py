import os
import torch

DEVICE = "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

DATASET_PATH = "C:\\Users\\Admin\\SaltSeg\\NEW_dataset\\train"
IMAGE_DATASET_PATH= "C:\\Users\\Admin\\SaltSeg\\NEW_dataset\\train\\images"
MASK_DATASET_PATH= "C:\\Users\\Admin\\SaltSeg\\NEW_dataset\\train\\masks"

TEST_SPLIT= 0.3

NUM_CHANNELS = 1
NUM_CLASSES = 3 #this changes to 3 classes
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 30
BATCH_SIZE = 64
MOMENTUM = 0.999
# gradient clipping value (for stability while training)
GRADIENT_CLIPPING = 1.0
# weight decay (L2 regularization) for the optimizer
WEIGHT_DECAY = 1e-8
# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
