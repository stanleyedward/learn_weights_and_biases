import wandb
import lightning as L
from model import MNIST_LitModule
from dataset import MNISTDataModule
from lightning.pytorch.loggers import WandbLogger
from callbacks import MyPrintingCallback, LogPredictionsCallback
from lightning.pytorch.callbacks import ModelCheckpoint

#initialize datamodule
datamodule = MNISTDataModule(batch_size=64,
                             data_dir="2. Optimize Lightning/dataset/",
                             num_workers=4
                             )
datamodule.prepare_data()
datamodule.setup()

#initialize model
model = MNIST_LitModule(n_layer_1=128,
                        n_layer_2=128,
                        num_classes=datamodule.num_classes,
                        learning_rate=1e-3)

#initialize wandb logger
wandb_logger = WandbLogger(project='first-MNIST', 
                           log_model='all',
                           save_dir="2. Optimize Lightning/logs/",
                           name='my_custom_run_name!')

#initialize callbacks
checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')
log_predictions_callback = LogPredictionsCallback(wandb_logger=wandb_logger)

trainer = L.Trainer(
    min_epochs=1,
    max_epochs=5,
    accelerator='gpu',
    devices=[0],
    callbacks=[MyPrintingCallback(),
               log_predictions_callback,
               checkpoint_callback],
    precision='16-mixed',
    strategy='auto',
    fast_dev_run=False,
    profiler=None
)

trainer.fit(model=model, datamodule=datamodule)
wandb.finish()

