# Training Hyperparams
H_LAYER_1 = 128
H_LAYER_2 = 256
NUM_CLASSES = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MIN_NUM_EPOCHS = 1
MAX_NUM_EPOCHS = 5

# Logging
LOGS_DIR = "2. Optimize Lightning/logs/"

# Dataset
DATA_DIR = "2. Optimize Lightning/dataset/"
NUM_WORKERS = 6

# Compute relatied
PROFILER = None
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = "16-mixed"
STRATEGY = "auto"