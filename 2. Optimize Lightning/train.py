import wandb
import lightning as L
from model import MNIST_LitModule
from dataset import MNISTDataModule
from lightning.pytorch.loggers import WandbLogger
from callbacks import MyPrintingCallback, LogPredictionsCallback
from lightning.pytorch.callbacks import ModelCheckpoint
import config

# initialize datamodule
datamodule = MNISTDataModule(
    batch_size=config.BATCH_SIZE, data_dir=config.DATA_DIR, num_workers=config.NUM_WORKERS
)
datamodule.prepare_data()
datamodule.setup()

# initialize model
model = MNIST_LitModule(
    n_layer_1=config.H_LAYER_1, n_layer_2=config.H_LAYER_2, num_classes=datamodule.num_classes, learning_rate=config.LEARNING_RATE
)

# initialize wandb logger
wandb_logger = WandbLogger(
    project="first-MNIST",
    log_model="all",
    save_dir=config.LOGS_DIR,
    name="my_custom_run_name_2!",
)

# initialize callbacks
checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
log_predictions_callback = LogPredictionsCallback(wandb_logger=wandb_logger)

# initialize trainer
trainer = L.Trainer(
    logger=wandb_logger,
    min_epochs=config.MIN_NUM_EPOCHS,
    max_epochs=config.MAX_NUM_EPOCHS,
    accelerator=config.ACCELERATOR,
    devices=config.DEVICES,
    callbacks=[MyPrintingCallback(), log_predictions_callback, checkpoint_callback],
    precision=config.PRECISION,
    strategy=config.STRATEGY,
    fast_dev_run=False,
    profiler=config.PROFILER,
)

trainer.fit(model=model, datamodule=datamodule)
wandb.finish()
