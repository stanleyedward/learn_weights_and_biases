# Training Hyperparams
INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
MIN_NUM_EPOCHS = 1
MAX_NUM_EPOCHS = 2

# Logging
LOGS_DIR = "1. Image Classification/logs/"

# Dataset
DATA_DIR = "1. Image Classification/dataset/"
NUM_WORKERS = 4

# Compute relatied
PROFILER = None
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "16-mixed"
STRATEGY = "auto"
