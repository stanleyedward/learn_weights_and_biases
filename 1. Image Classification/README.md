<img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

<!--- @wandbcode{pytorch-lightning-colab} -->

# Image Classification using PyTorch Lightning and W&B⚡️

We will build an image classification pipeline using PyTorch Lightning. We will follow this [style guide](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html) to increase the readability and reproducibility of our code. A cool explanation of this available [here](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY).


## Setting up PyTorch Lightning and W&B

For this tutorial, we need PyTorch Lightning(ain't that obvious!) and Weights and Biases.

```sh
$ pip install -r requirements.txt
```

## 🔧 DataModule - The Data Pipeline we Deserve

DataModules are a way of decoupling data-related hooks from the LightningModule so you can develop dataset agnostic models.

It organizes the data pipeline into one shareable and reusable class. A datamodule encapsulates the five steps involved in data processing in PyTorch:
- Download / tokenize / process.
- Clean and (maybe) save to disk.
- Load inside Dataset.
- Apply transforms (rotate, tokenize, etc…).
- Wrap inside a DataLoader.

Learn more about datamodules [here](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). Let's build a datamodule for the Cifar-10 dataset.

### DataModule code: [dataset.py](dataset.py)

## 📱 Callbacks

A callback is a self-contained program that can be reused across projects. PyTorch Lightning comes with few [built-in callbacks](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks) which are regularly used.
Learn more about callbacks in PyTorch Lightning [here](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html).

### Built-in Callbacks

In this tutorial, we will use [Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping) and [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) built-in callbacks. They can be passed to the `Trainer`.

### Custom Callbacks
If you are familiar with Custom Keras callback, the ability to do the same in your PyTorch pipeline is just a cherry on the cake.

Since we are performing image classification, the ability to visualize the model's predictions on some samples of images can be helpful. This in the form of a callback can help debug the model at an early stage.

### Callbacks code: [callbacks.py](callbacks.py)

## 🎺 LightningModule - Define the System

The LightningModule defines a system and not a model. Here a system groups all the research code into a single class to make it self-contained. `LightningModule` organizes your PyTorch code into 5 sections:
- Computations (`__init__`).
- Train loop (`training_step`)
- Validation loop (`validation_step`)
- Test loop (`test_step`)
- Optimizers (`configure_optimizers`)

One can thus build a dataset agnostic model that can be easily shared. Let's build a system for Cifar-10 classification.

### Model Code: [model.py](model.py)

## 🚋 Train and Evaluate

Now that we have organized our data pipeline using `DataModule` and model architecture+training loop using `LightningModule`, the PyTorch Lightning `Trainer` automates everything else for us.

The Trainer automates:
- Epoch and batch iteration
- Calling of `optimizer.step()`, `backward`, `zero_grad()`
- Calling of `.eval()`, enabling/disabling grads
- Saving and loading weights
- Weights and Biases logging
- Multi-GPU training support
- TPU support
- 16-bit training support

### Training Code: [train.py](train.py)

<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>epoch</td><td>▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅█</td></tr><tr><td>test_acc</td><td>▁</td></tr><tr><td>test_loss</td><td>▁</td></tr><tr><td>train_acc_epoch</td><td>▁█</td></tr><tr><td>train_acc_step</td><td>▁▂▂▃▃▃▅▄▄▆▅▄▃▄▅▄▆▅▅▆▄▆▄▅▃▆▆▆▅▄▅▅▆▇▆▆▆██▅</td></tr><tr><td>train_loss_epoch</td><td>█▁</td></tr><tr><td>train_loss_step</td><td>█▇▇▆▅▆▅▅▄▃▄▅▄▅▄▅▃▄▅▃▅▃▄▃▄▃▂▃▃▄▂▃▄▃▂▃▃▁▁▃</td></tr><tr><td>trainer/global_step</td><td>▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇█████</td></tr><tr><td>val_acc</td><td>▁█</td></tr><tr><td>val_loss</td><td>█▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>epoch</td><td>2</td></tr><tr><td>test_acc</td><td>0.5488</td></tr><tr><td>test_loss</td><td>1.25715</td></tr><tr><td>train_acc_epoch</td><td>0.51124</td></tr><tr><td>train_acc_step</td><td>0.5</td></tr><tr><td>train_loss_epoch</td><td>1.34988</td></tr><tr><td>train_loss_step</td><td>1.25217</td></tr><tr><td>trainer/global_step</td><td>2814</td></tr><tr><td>val_acc</td><td>0.5468</td></tr><tr><td>val_loss</td><td>1.26892</td></tr></table><br/></div></div>


## Final Thoughts
I come from the TensorFlow/Keras ecosystem and find PyTorch a bit overwhelming even though it's an elegant framework. Just my personal experience though. While exploring PyTorch Lightning, I realized that almost all of the reasons that kept me away from PyTorch is taken care of. Here's a quick summary of my excitement:
- Then: Conventional PyTorch model definition used to be all over the place. With the model in some `model.py` script and the training loop in the `train.py `file. It was a lot of looking back and forth to understand the pipeline.
- Now: The `LightningModule` acts as a system where the model is defined along with the `training_step`, `validation_step`, etc. Now it's modular and shareable.
- Then: The best part about TensorFlow/Keras is the input data pipeline. Their dataset catalog is rich and growing. PyTorch's data pipeline used to be the biggest pain point. In normal PyTorch code, the data download/cleaning/preparation is usually scattered across many files.
- Now: The DataModule organizes the data pipeline into one shareable and reusable class. It's simply a collection of a `train_dataloader`, `val_dataloader`(s), `test_dataloader`(s) along with the matching transforms and data processing/downloads steps required.
- Then: With Keras, one can call `model.fit` to train the model and `model.predict` to run inference on. `model.evaluate` offered a good old simple evaluation on the test data. This is not the case with PyTorch. One will usually find separate `train.py` and `test.py` files.
- Now: With the `LightningModule` in place, the `Trainer` automates everything. One needs to just call `trainer.fit` and `trainer.test` to train and evaluate the model.
- Then: TensorFlow loves TPU, PyTorch...well!
- Now: With PyTorch Lightning, it's so easy to train the same model with multiple GPUs and even on TPU. Wow!
- Then: I am a big fan of Callbacks and prefer writing custom callbacks. Something as trivial as Early Stopping used to be a point of discussion with conventional PyTorch.
- Now: With PyTorch Lightning using Early Stopping and Model Checkpointing is a piece of cake. I can even write custom callbacks.


## 🎨 Conclusion and Resources

I hope you find this report helpful. I will encourage to play with the code and train an image classifier with a dataset of your choice.

Here are some resources to learn more about PyTorch Lightning:
- [Step-by-step walk-through](https://lightning.ai/docs/pytorch/latest/starter/introduction.html) - This is one of the official tutorials. Their documentation is really well written and I highly encourage it as a good learning resource.
- [Use Pytorch Lightning with Weights & Biases](https://wandb.me/lightning) - This is a quick colab that you can run through to learn more about how to use W&B with PyTorch Lightning.
