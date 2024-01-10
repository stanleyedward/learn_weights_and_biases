from typing import Any
from lightning.pytorch.callbacks import Callback
import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger


class MyPrintingCallback(Callback):
    def __init(self):
        super().__init__()

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        print("Starting to train!")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        print("Training is done.")
        
class LogPredictionsCallback(Callback):
    
    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule,
                                outputs, batch, batch_idx, wandb_logger: WandbLogger):
        '''called when the validation batch ends.'''
        
        #'outputs' comes from 'LightningModule.validation_step'
        #which corresponds to urmodel preds from first batch
        
        #lets log 20 samples image preds from first batch
        if batch_idx == 20:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            #option 1:log iamges iwth wandblogger.log_image'
            wandb_logger.log_image(key='sample_images', images=images, captions=captions)
            
            #option 2: log predictions as a table
            columns = ['image', 'ground_truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(key='sample_table', columns=columns, data=data)
            
