import lightning as L
from lightning.pytorch.loggers import WandbLogger


import wandb
from model import LitModel
from dataset import StanfordCarsDataModule

model = LitModel((3, 300, 300), 196)
dm = StanfordCarsDataModule(
    batch_size=32,
    model_transforms=model.resnet18_transforms,
    data_dir="3. Transfer Learning/dataset/",
)
dm.prepare_data()
dm.setup()

trainer = L.Trainer(
    logger=WandbLogger(
        project="TransferLearning", save_dir="3. Transfer Learning/logs/"
    ),
    max_epochs=10,
    accelerator="gpu",
)
trainer.fit(model, dm)
trainer.test(model, dm)
wandb.finish()
