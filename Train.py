import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

from Data.Dataloader import MRIdata
from Models.Training import swinUNETR

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="MR2CT")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="saved_models\\sCT\\", save_top_k=1, monitor="val_loss", save_on_train_epoch_end=True)
    #last_chpt = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\saved_models\\epoch=315-step=319.ckpt"

    SWIN_size = (96, 96, 96)

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        #add mutli-gpu
        devices=[0],
        max_epochs=5000,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,
        precision=32
    )

    trainer.fit(
        model=swinUNETR(batch_size=1, SWIN_size=SWIN_size),
        #ckpt_path=last_chpt,
        datamodule=MRIdata(batch_size=1, SWIN_size=SWIN_size)
    )
