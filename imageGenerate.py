####### not so good right now, will be fixed ##########
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

#  this is my local checkpoint path
c = "/home/caner/Downloads/epoch=349-step=125999.ckpt"
# THIS PART SAME AS THE train.py,
########################################################################################################
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Config  -----------------------------------------------------------------
data_dir = "/home/caner/Desktop/watercolor-CycleGAN/data/"
transform = ImageTransform(img_size=256)
batch_size = 1
lr = {
    "G": 0.0002,
    "D": 0.0002
}
epoch = 400
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

model = CycleGAN_LightningSystem(G_basestyle, G_stylebase, D_base, D_style, 
                                 lr, transform, reconstr_w, id_w)
trainer = Trainer(logger=False,
    max_epochs=350, # I couldn"t implement the pl.load_from_checkpoint so I went the log way 
    gpus=1,
    reload_dataloaders_every_epoch=True,
    num_sanity_val_steps=0,
    resume_from_checkpoint=c)
trainer.fit(model, datamodule=dm)
##################################################################################################3
#generate image part

net = model.G_basestyle
photo_path = "/home/caner/Desktop/predict/"
photos = os.listdir(photo_path)
for photo in photos:
    img = transform(Image.open(photo_path + photo), phase="test")
    device = torch.device("cpu")
    print(device)
    img = img.to(device)
    gen_img = net(img.unsqueeze(0))[0]
    gen_img = gen_img * 0.5 + 0.5
    gen_img = gen_img * 255
    gen_img = gen_img.detach().cpu().numpy().astype(np.uint8)
    gen_img = np.transpose(gen_img, [1,2,0])
    gen_img = Image.fromarray(gen_img)
    gen_img.save("g"+ photo)


