import os
import glob
from functools import partial

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import SimpleITK as sitk
import numpy as np
import torch
import pytorch_lightning

from tqdm import tqdm
from Models.Training import swinUNETR
from Data.Data import NiftiData, test_transform
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

if __name__ == "__main__":
    model_path = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\saved_models\\sCT\\epoch=1234-step=1239.ckpt"
    input_predict_path = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\Data\\MR"
    input_data = NiftiData(input_predict_path)

    output_path = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\Data\\sCT"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    device = torch.device(1)
    model = swinUNETR(
        (96, 96, 96),
        img_size=(1, 1, 96, 96, 96),
        in_channels=1,
        out_channels=1,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
        lr=1e-4,
        wd=1e-5
    )
    model.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[96, 96, 96],
        sw_batch_size=12,
        predictor=model,
        overlap=0.0,
        mode='gaussian',
    )

    dict_keys = ["MR"]
    transform = test_transform(dict_keys)

    input_data.transform = transform
    content_Dataloader = DataLoader(input_data, batch_size=1, num_workers=12)

    with torch.no_grad():
        for i, content in enumerate(tqdm(content_Dataloader)):
            print(f"Inferring image {i}")
            image = content['MR'].to(device)
            CT_recon = model_inferer_test(image)
            CT_recon_array = np.clip(CT_recon.detach().cpu().numpy(), 0, 1)
            CT_array = (CT_recon_array * 255)[0, 0, :, :, 64]
            CT_image = sitk.GetImageFromArray(CT_array)
