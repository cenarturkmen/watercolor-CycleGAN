from CycleGAN_ls import CycleGAN_LightningSystem
from dataModule import ImageTransform, WatercolorDataset, WatercolorDataModule
from discriminator import CycleGAN_Discriminator
from generator import CycleGAN_Unet_Generator
import os
import glob
import  random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Config  -----------------------------------------------------------------
data_dir = '/content/drive/MyDrive/data/'
transform = ImageTransform(img_size=256)
batch_size = 1
lr = {
    'G': 0.0002,
    'D': 0.0002
}
epoch = 160
seed = 42
reconstr_w = 10
id_w = 5
seed_everything(seed)

# DataModule  -----------------------------------------------------------------
dm = WatercolorDataModule(data_dir, transform, batch_size, seed=seed)

G_basestyle = CycleGAN_Unet_Generator()
G_stylebase = CycleGAN_Unet_Generator()
D_base = CycleGAN_Discriminator()
D_style = CycleGAN_Discriminator()

# LightningModule  --------------------------------------------------------------
model = CycleGAN_LightningSystem(G_basestyle, G_stylebase, D_base, D_style, 
                                 lr, transform, reconstr_w, id_w)
# Callback
checkpoint_callback = ModelCheckpoint(dirpath="/content/drive/MyDrive/checkpoint",
                                      period=10)
# Trainer  --------------------------------------------------------------
trainer = Trainer(
    logger=False,
    max_epochs=epoch,
    gpus=1,
    checkpoint_callback=checkpoint_callback,
    reload_dataloaders_every_epoch=True,
    num_sanity_val_steps=0,
    # resume_from_checkpoint='/content/drive/MyDrive/checkpoint/epoch=279-step=100799.ckpt' 
)

# Train ------------------------------------------------------------------------
trainer.fit(model, datamodule=dm)


