import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Data.Dataloader import MRIdata
from Models.Training import swinUNETR

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="MR2CT")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="saved_models\\MR2CT", save_top_k=1, monitor="val_loss", save_on_train_epoch_end=True)
    #last_chpt = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\saved_models\\epoch=315-step=319.ckpt"

    SWIN_size = (128, 128, 128)

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
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