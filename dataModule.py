from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import glob
import os
import random


# Resize the image (256,256) and normalize image.
class ImageTransform:
    def __init__(self, img_size=256):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])}

    def __call__(self, img, phase="train"):
        img = self.transform[phase](img)

        return img

# Define dataset
class WatercolorDataset(Dataset):
    def __init__(self, base_img_paths, style_img_paths,  transform, phase="train"):
        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.base_img_paths), len(self.style_img_paths)])

    def __getitem__(self, idx):        
        base_img_path = self.base_img_paths[idx]
        style_img_path = self.style_img_paths[idx]
        base_img = Image.open(base_img_path)
        style_img = Image.open(style_img_path)

        base_img = self.transform(base_img, self.phase)
        style_img = self.transform(style_img, self.phase)

        return base_img, style_img

# Define data module 
class WatercolorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, phase="train", seed=0):
        super(WatercolorDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.phase = phase
        self.seed = seed

    def prepare_data(self):
        self.base_img_paths = glob.glob(os.path.join(self.data_dir, "photo_jpg_less/", "*.jpg"))
        self.style_img_paths = glob.glob(os.path.join(self.data_dir, "watercolor_jpg/", "*.jpg"))

    def train_dataloader(self):
        random.seed()
        random.shuffle(self.base_img_paths)
        random.shuffle(self.style_img_paths)
        random.seed(self.seed)
        self.train_dataset = WatercolorDataset(self.base_img_paths, self.style_img_paths, self.transform, self.phase)
        
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True
                         )