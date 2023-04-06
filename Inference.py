import os
import glob
from functools import partial
import warnings

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

def realign(gt_CT, sCT):
    try:
        size = gt_CT.GetSize()
        origin = gt_CT.GetOrigin()
        spacing = gt_CT.GetSpacing()
        direction = gt_CT.GetDirection()
        center = [origin[i] + (size[i] - 1) * spacing[i] / 2.0 for i in range(3)]
        center = ' '.join([str(c) for c in center])

        RigidElastix = sitk.ElastixImageFilter()

        sCT.SetOrigin(origin)
        sCT.SetDirection(direction)

        RigidElastix.SetFixedImage(gt_CT)
        RigidElastix.SetMovingImage(sCT)
        RigidElastix.LogToConsoleOff()
        rigid_map = RigidElastix.ReadParameterFile("Parameters_Rigid.txt")

        rigid_map['CenterOfRotation'] = [center]
        rigid_map['ResultImageFormat'] = ['nii']
        RigidElastix.SetParameterMap(rigid_map)
        RigidElastix.Execute()
        sCTtoGT = RigidElastix.GetResultImage()

        sCTtoGT.SetOrigin(origin)
        sCTtoGT.SetDirection(gt_CT.GetDirection())

        #CT values are 0 where not aligned, address with intensity shift, may not need
        '''CT_to_T1_array = sitk.GetArrayFromImage(CT_to_T1_image)
        CT_to_T1_array[CT_to_T1_array == 0] = -1024
        CT_to_T1_array2im = sitk.GetImageFromArray(CT_to_T1_array)

        CT_to_T1_array2im.SetOrigin(CT_to_T1_image.GetOrigin())
        CT_to_T1_array2im.SetDirection(CT_to_T1_image.GetDirection())'''
        return sCTtoGT

    except RuntimeError as e:
        warnings.warn(str(e))

def rescale(gt_CT, sCT):
    gt_CT = sitk.GetArrayFromImage(gt_CT)
    GT_min, GT_max = np.min(gt_CT), np.max(gt_CT)
    sCT_min, sCT_max = np.min(sCT), np.max(sCT)
    target_range = GT_max - GT_min
    sCT_range = sCT_max - sCT_min
    scale_factor = target_range / sCT_range
    sCTrescale = (sCT - sCT_min) * scale_factor + GT_min
    return sCTrescale

if __name__ == "__main__":
    model_path = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\saved_models\\sCT\\epoch=2248-step=2253.ckpt"
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
    model = model.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
    #model = model.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[96, 96, 96],
        sw_batch_size=12,
        predictor=model,
        overlap=0.9,
        mode='gaussian',
    )

    dict_keys = ["MR", "CT"]
    transform = test_transform(dict_keys)

    input_data.transform = transform
    content_Dataloader = DataLoader(input_data, batch_size=1, num_workers=12)

    with torch.no_grad():
        for i, content in enumerate(tqdm(content_Dataloader)):
            name = content_Dataloader.dataset.CT_paths[i].split("\\")[-1].split("_")[0]
            print(f"\nInferring image {name}")
            gt = sitk.ReadImage(content_Dataloader.dataset.CT_paths[i])
            image = content['MR'].to(device)
            CT_recon = model_inferer_test(image)

            CT_recon_array = np.clip(CT_recon.detach().cpu().numpy(), 0, 1)
            sCT = CT_recon_array[0, 0, :, :, :]
            sCT = np.transpose(sCT)
            sCT = np.flip(sCT)

            CT_rescale = rescale(gt, sCT)
            sCT = sitk.GetImageFromArray(CT_rescale)
            #sitk.WriteImage(sCT, str(os.path.join(output_path, f"UnalignedsCT_{i}.nii.gz")))

            CT_recon = realign(gt, sCT)
            sitk.WriteImage(CT_recon, str(os.path.join(output_path, f"sCT_{name}.nii.gz")))