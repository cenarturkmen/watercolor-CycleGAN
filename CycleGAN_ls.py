import torch
import pytorch_lightning as pl 
from torch import optim, nn


class CycleGAN_LightningSystem(pl.LightningModule):
    def __init__(self, G_basestyle, G_stylebase, D_base, D_style, lr, transform, reconstr_w=10, id_w=2):
        super(CycleGAN_LightningSystem, self).__init__()
        self.G_basestyle = G_basestyle
        self.G_stylebase = G_stylebase
        self.D_base = D_base
        self.D_style = D_style
        self.lr = lr
        self.transform = transform
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.cnt_train_step = 0
        self.step = 0

        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()
        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []
        self.validity = []
        self.reconstr = []
        self.identity = []

    def configure_optimizers(self):
        self.g_basestyle_optimizer = optim.Adam(self.G_basestyle.parameters(), lr=self.lr["G"], betas=(0.5, 0.999))
        self.g_stylebase_optimizer = optim.Adam(self.G_stylebase.parameters(), lr=self.lr["G"], betas=(0.5, 0.999))
        self.d_base_optimizer = optim.Adam(self.D_base.parameters(), lr=self.lr["D"], betas=(0.5, 0.999))
        self.d_style_optimizer = optim.Adam(self.D_style.parameters(), lr=self.lr["D"], betas=(0.5, 0.999))

        return [self.g_basestyle_optimizer, self.g_stylebase_optimizer, self.d_base_optimizer, self.d_style_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        base_img, style_img = batch
        b = base_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            # Validity
            # MSELoss
            val_base = self.generator_loss(self.D_base(self.G_stylebase(style_img)), valid)
            val_style = self.generator_loss(self.D_style(self.G_basestyle(base_img)), valid)
            val_loss = (val_base + val_style) / 2

            # Reconstruction
            reconstr_base = self.mae(self.G_stylebase(self.G_basestyle(base_img)), base_img)
            reconstr_style = self.mae(self.G_basestyle(self.G_stylebase(style_img)), style_img)
            reconstr_loss = (reconstr_base + reconstr_style) / 2

            # Identity
            id_base = self.mae(self.G_stylebase(base_img), base_img)
            id_style = self.mae(self.G_basestyle(style_img), style_img)
            id_loss = (id_base + id_style) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss

            return {"loss": G_loss, "validity": val_loss, "reconstr": reconstr_loss, "identity": id_loss}

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            # MSELoss
            D_base_gen_loss = self.discriminator_loss(self.D_base(self.G_stylebase(style_img)), fake)
            D_style_gen_loss = self.discriminator_loss(self.D_style(self.G_basestyle(base_img)), fake)
            D_base_valid_loss = self.discriminator_loss(self.D_base(base_img), valid)
            D_style_valid_loss = self.discriminator_loss(self.D_style(style_img), valid)
            
            D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2
            
            # Loss Weight
            D_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3

            # Count up
            self.cnt_train_step += 1

            return {"loss": D_loss}

    def training_epoch_end(self, outputs):
        self.step += 1
        
        avg_loss = sum([torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 4 for i in range(4)])
        G_mean_loss = sum([torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        D_mean_loss = sum([torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2 for i in [2, 3]])
        validity = sum([torch.stack([x["validity"] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        reconstr = sum([torch.stack([x["reconstr"] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
        identity = sum([torch.stack([x["identity"] for x in outputs[i]]).mean().item() / 2 for i in [0, 1]])
            
        self.losses.append(avg_loss)
        self.G_mean_losses.append(G_mean_loss)
        self.D_mean_losses.append(D_mean_loss)
        self.validity.append(validity)
        self.reconstr.append(reconstr)
        self.identity.append(identity)
        
        if self.step % 10 == 0:
            # Display Model Output
            target_img_paths = glob.glob("/content/drive/MyDrive/data/photo_jpg_less/*.jpg")[:5]
            target_imgs = [self.transform(Image.open(path), phase="test") for path in target_img_paths]
            target_imgs = torch.stack(target_imgs, dim=0)
            target_imgs = target_imgs.cuda()

            gen_imgs = self.G_basestyle(target_imgs)
            gen_img = torch.cat([target_imgs, gen_imgs], dim=0)

            # Reverse Normalization
            gen_img = gen_img * 0.5 + 0.5
            gen_img = gen_img * 255

            joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)

            joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
            joined_images = np.transpose(joined_images, [1,2,0])

            # Visualize
            fig = plt.figure(figsize=(18, 8))
            plt.imshow(joined_images)
            plt.axis("off")
            plt.title(f"Epoch {self.step}")
            plt.show()
            plt.clf()
            plt.close()

        return None