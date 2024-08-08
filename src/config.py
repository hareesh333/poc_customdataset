import torch
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

BATCH_SIZE = 20 # increase / decrease according to GPU memeory
RESIZE_TO = 128 # resize the image for training and transforms
NUM_EPOCHS = 2000 # number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = '../Microcontroller Detection/train'
# validation images and XML files directory
VALID_DIR = '../Microcontroller Detection/test'

# classes: 0 index is reserved for background
CLASSES = [
    'Sealant','Opening direction to the left','Opening direction to the right','Packing unit'
,'Side-hung window (SH)','background'
]
NUM_CLASSES = 6

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 200 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 200 # save model after these many epochs

# Early stopping settings
EARLY_STOPPING = True  # Enable early stopping
PATIENCE = 200  # Number of epochs with no improvement after which training will be stopped
EARLY_STOPPING_METRIC = 'precision'  # Metric to monitor for early stopping: 'precision' or 'recall'

# config.py
GRADIENT_ACCUMULATION_STEPS = 4  # or whatever value you need
