import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from dataset import CIFAR10DataModule
from model import LitModel
from callbacks import ImagePredictionLogger, MyPrintingCallback
import config

datamodule = CIFAR10DataModule(
    batch_size=config.BATCH_SIZE,
    data_dir=config.DATA_DIR,
    num_workers=config.NUM_WORKERS,
)
# To access the x_dataloader we need to call prepare_data and setup.
datamodule.prepare_data()
datamodule.setup()

# Samples required by the custom ImagePredictionLogger Callback to log image predictions
val_samples = next(iter(datamodule.val_dataloader()))
val_imgs, val_labels = val_samples[0], val_samples[1]
print(f"val img shape: {val_imgs.shape}, val_label_shape: {val_labels.shape}")

model = LitModel(
    input_shape=config.INPUT_SHAPE,
    num_classes=datamodule.num_classes,
    learning_rate=config.LEARNING_RATE,
)

# initialize the wandb logger
wandb_logger = WandbLogger(
    project="first-CIFAR10", job_type="train", save_dir=config.LOGS_DIR
)

# initialize callbacks
early_stop_callback = L.pytorch.callbacks.EarlyStopping(monitor="val_loss")
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint()

# initialzize the trainer
trainer = L.Trainer(
    min_epochs=config.MIN_NUM_EPOCHS,
    max_epochs=config.MAX_NUM_EPOCHS,
    logger=wandb_logger,
    callbacks=[
        early_stop_callback,
        ImagePredictionLogger(val_samples=val_samples),
        checkpoint_callback,
        MyPrintingCallback(),
    ],
    accelerator=config.ACCELERATOR,
    devices=config.DEVICES,
    precision=config.PRECISION,
    strategy=config.STRATEGY,
    profiler=config.PROFILER,
    fast_dev_run=False,
)

# train the model

trainer.fit(model=model, datamodule=datamodule)
trainer.test(dataloaders=datamodule.test_dataloader())

wandb.finish()
