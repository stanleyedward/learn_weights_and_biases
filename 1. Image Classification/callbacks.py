from lightning.pytorch.callbacks import Callback
import lightning as L
import torch
import wandb


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # bring the tensors to the CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # get model preds
        logits = pl_module(val_imgs)
        preds = torch.argmax(input=logits, dim=-1)
        # log the images as wandb image
        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(
                        val_imgs[: self.num_samples],
                        preds[: self.num_samples],
                        val_labels[: self.num_samples],
                    )
                ]
            }
        )


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        print("Starting to train!")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        print("Training is done.")
