#from Models.MONAImodels import SwinUNETR
from monai.networks.nets import SwinUNETR

from torch.nn import L1Loss, MSELoss
from monai.losses import ContrastiveLoss, DiceCELoss, DiceLoss, SSIMLoss, MaskedDiceLoss
from monai.transforms import Activations, AsDiscrete
from monai.data import decollate_batch

from Models.Optimizer import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from monai.visualize.img2tensorboard import plot_2d_or_3d_image
import numpy as np
from pytorch_lightning import LightningModule
import torch
import warnings

class swinUNETR(LightningModule):
    def __init__(self, SWIN_size,
                 img_size=(1, 1, 96, 96, 96), in_channels=1, out_channels=1, batch_size=1, feature_size=48, drop_rate=0.0,
                 attn_drop_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False, lr=1e-4, wd=1e-5):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [
            torch.zeros(self.hparams.img_size)]

        self.model = SwinUNETR(
            img_size=(SWIN_size),
            in_channels=1,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True)

        #self.model = ViTA(self.hparams.in_channels, self.hparams.img_size, self.hparams.patch_size)
        #for CT recon, note L1 = MAE
        self.L1 = L1Loss()
        self.L2 = MSELoss()
        self.contrast = ContrastiveLoss(temperature=0.05)
        self.SSIM = SSIMLoss(spatial_dims=3)


    def forward(self, inputs):
        CToutput = self.model(inputs)
        return CToutput

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        lr_scheduler = {
            #decay at plataeu
            #warmup is 1% of max epochs
            'scheduler': ReduceLROnPlateau(optimizer, mode='min'),
            #'scheduler': LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=5000),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    def _prepare_batch(self, batch):
        MR_batch = torch.cat([batch[i]['MR'] for i in range(len(batch))])
        CT_batch = torch.cat([batch[i]['CT'] for i in range(len(batch))])
        return MR_batch, CT_batch

    def _common_step(self, batch, batch_idx, stage: str):
        input_MR, gt_CT = self._prepare_batch(batch)

        CT_recon = self.forward(input_MR)

        gt_CT_flat_out = gt_CT.flatten(start_dim=1, end_dim=4)
        CT_recon_flat_out = CT_recon.flatten(start_dim=1, end_dim=4)

        r1_loss = self.L1(CT_recon, gt_CT)
        r2_loss = self.L2(CT_recon, gt_CT)

        cl_loss = self.contrast(gt_CT_flat_out, CT_recon_flat_out)

        ssim_loss_0 = self.SSIM(CT_recon[0:1, ...], gt_CT[0:1, ...], data_range=gt_CT.max().unsqueeze(0))
        ssim_loss_1 = self.SSIM(CT_recon[1:2, ...], gt_CT[1:2, ...], data_range=gt_CT.max().unsqueeze(0))
        ssim_loss_2 = self.SSIM(CT_recon[2:3, ...], gt_CT[2:3, ...], data_range=gt_CT.max().unsqueeze(0))
        ssim_total = (ssim_loss_0 + ssim_loss_1 + ssim_loss_2)/3

        # Adjust the CL loss by Recon Loss
        total_loss = r1_loss + r2_loss + (cl_loss * r1_loss) + ssim_total
        train_steps = self.current_epoch + batch_idx

        self.log_dict({
            f'{stage}_loss': total_loss.item(),
            'step': float(train_steps),
            'epoch': float(self.current_epoch)}, batch_size=self.hparams.batch_size)

        self.log_dict({
            f'{stage}_loss': total_loss.item(),
            'step': float(train_steps),
            'epoch': float(self.current_epoch)}, batch_size=self.hparams.batch_size)

        if train_steps % 10000 == 0:
            try:
                self.log_dict({
                    'L1': r1_loss.item(),
                    'L2': r2_loss.item(),
                    'Contrastive': cl_loss.item(),
                    'SSIM': ssim_total.item(),
                    'epoch': float(self.current_epoch),
                    'step': float(train_steps)}, batch_size=self.hparams.batch_size)

                MRname = input_MR.meta['filename_or_obj'].split("\\")[-1].split("_")[0]
                CTname = gt_CT.meta['filename_or_obj'].split("\\")[-1].split("_")[0]
                sCTname = CT_recon.meta['filename_or_obj'].split("\\")[-1].split("_")[0]

                self.logger.log_image(key="Input", images=[
                    (input_MR.detach().cpu().numpy() * 255)[0, 0, :, :, 64]],
                                      caption=[f"MRI: c{MRname}"])

                self.logger.log_image(key="Ground Truths", images=[
                    (gt_CT.detach().cpu().numpy() * 255)[0, 0, :, :, 64]],
                                      caption=[f"CT: c{CTname}"])

                CT_recon = CT_recon.to(dtype=torch.float16)
                CT_recon_array = np.clip(CT_recon.detach().cpu().numpy(), 0, 1)

                self.logger.log_image(key="Predictions", images=[
                    (CT_recon_array * 255)[0, 0, :, :, 64]],
                                      caption=[f"CT Recon: c{sCTname}"])

            except FileNotFoundError as e:
                warnings.warn(str(e))

        return total_loss