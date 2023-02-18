from Models.MONAImodels import SwinUNETR
#from monai.networks.nets import SwinUNETR

from torch.nn import L1Loss, MSELoss
from monai.losses import ContrastiveLoss, DiceCELoss, SSIMLoss

from Models.Optimizer import LinearWarmupCosineAnnealingLR

from monai.visualize.img2tensorboard import plot_2d_or_3d_image
import numpy as np
from pytorch_lightning import LightningModule
import torch

class swinUNETR(LightningModule):
    def __init__(self, SWIN_size,
                 img_size=(1, 1, 128, 128, 128), in_channels=4, batch_size=1, feature_size=48,
                 lr=1e-4, wd=1e-5):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [
            torch.zeros(self.hparams.img_size)]

        self.model = SwinUNETR(
            img_size=(SWIN_size),
            in_channels=1,
            feature_size=48,
            use_checkpoint=True)

        #self.model = ViTA(self.hparams.in_channels, self.hparams.img_size, self.hparams.patch_size)
        #for CT recon, note L1 = MAE
        self.L1 = L1Loss()
        self.L2 = MSELoss()
        self.contrast = ContrastiveLoss(temperature=0.05)
        self.SSIM = SSIMLoss(spatial_dims=3)

        #for skull seg, try regular dice CE for now: change to squared later?
        self.DSCE_Loss = DiceCELoss(to_onehot_y=False, softmax=True, smooth_dr=1e-6, smooth_nr=0.0)


    def forward(self, inputs):
        CToutput, SegOutput = self.model(inputs)
        return CToutput, SegOutput

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        lr_scheduler = {
            #warmup is 1% of max epochs
            'scheduler': LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=5000),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    def _prepare_batch(self, batch):
        MR_batch = batch['MR']
        CT_batch = batch['CT']
        Seg_batch = batch['Segs']
        return MR_batch, CT_batch, Seg_batch

    def _common_step(self, batch, batch_idx, stage: str):
        input_MR, gt_CT, gt_Seg = self._prepare_batch(batch)

        CT_recon, skull_Seg = self.forward(input_MR)

        gt_CT_flat_out = gt_CT.flatten(start_dim=1, end_dim=4)
        CT_recon_flat_out = CT_recon.flatten(start_dim=1, end_dim=4)

        r1_loss = self.L1(CT_recon, gt_CT)
        r2_loss = self.L2(CT_recon, gt_CT)
        cl_loss = self.contrast(CT_recon_flat_out, gt_CT_flat_out)
        ssim_loss = self.SSIM(CT_recon, gt_CT, data_range=gt_CT.max().unsqueeze(0))

        DSCE_loss = self.DSCE_Loss(skull_Seg, gt_Seg)

        # Adjust the CL loss by Recon Loss
        total_loss = r1_loss + cl_loss * r1_loss + r2_loss + ssim_loss + DSCE_loss
        train_steps = self.current_epoch + batch_idx

        self.log_dict({
            f'{stage}_loss': total_loss.item(),
            'step': float(train_steps),
            'epoch': float(self.current_epoch)}, batch_size=self.hparams.batch_size)

        self.log_dict({
            f'{stage}_loss': total_loss.item(),
            'step': float(train_steps),
            'epoch': float(self.current_epoch)}, batch_size=self.hparams.batch_size)

        if train_steps % 10 == 0:
            self.log_dict({
                'L1': r1_loss.item(),
                'L2': r2_loss.item(),
                'Contrastive': cl_loss.item(),
                'SSIM': ssim_loss.item(),
                'DSCE_Loss': DSCE_loss.item(),
                'epoch': float(self.current_epoch),
                'step': float(train_steps)}, batch_size=self.hparams.batch_size)

            self.logger.log_image(key="Input", images=[
                (input_MR.detach().cpu().numpy() * 255)[0, 0, :, :, 64]],
                                  caption=["MRI"])

            self.logger.log_image(key="Ground Truths", images=[
                (gt_CT.detach().cpu().numpy() * 255)[0, 0, :, :, 64],
                (gt_Seg.detach().cpu().numpy() * (255/7))[0, 0, :, :, 64]],
                                  caption=["GT CT", "GT Seg"])

            CT_recon = CT_recon.to(dtype=torch.float16)
            CT_recon_array = np.clip(CT_recon.detach().cpu().numpy(), 0, 1)
            skull_Seg = skull_Seg.to(dtype=torch.float16)

            self.logger.log_image(key="Predictions", images=[
                (CT_recon_array * 255)[0, 0, :, :, 64],
                ((skull_Seg.detach().cpu().numpy().argmax(1)) * (255/7))[0, :, :, 64]],
                                  caption=["CT Recon", "Skull Seg Prediction"])

        return total_loss


