import os
import glob
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    Spacingd,
    ToTensord
)

class NiftiData(Dataset):
    def __init__(self, SWIN_size):
        # for image check
        # self.path = os.path.join(os.getcwd()+'\\Images')

        # for standard training
        self.MRpath = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\Data\\MR"
        self.MR_paths = sorted(glob.glob(self.MRpath + '\\*'))

        self.CTpath = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\Data\\CT"
        self.CT_paths = sorted(glob.glob(self.CTpath + '\\*'))

        self.Segpath = "C:\\Users\\Austin Tapp\\Documents\\swinUNETR\\Data\\SkullSegs"
        self.Seg_paths = sorted(glob.glob(self.Segpath + '\\*'))

        # for prediction
        # self.Testpath = ''
        # self.pred_path = sorted(glob.glob(self.Testpath + '\\*'))
        # print("testing: ", self.pred_path)

        self.transform = Compose(

            [
                LoadImaged(keys=["MR", "CT", "Segs"]),
                EnsureChannelFirstd(keys=["MR", "CT", "Segs"]),
                Orientationd(keys=["MR", "CT", "Segs"], axcodes='RAI'),
                # #redundent but still okay to do, segmentation should be to nearest, while others are bilinear
                Spacingd(keys=["MR", "CT", "Segs"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "nearest")),
                ScaleIntensityd(keys=["MR", "CT"], minv=0.0, maxv=1.0),
                CropForegroundd(keys=["MR", "CT", "Segs"], source_key="MR", k_divisible=SWIN_size),
                #RandSpatialCropd(keys=["MR", "CT", "Segs"], roi_size=SWIN_size, random_size=False),
                RandCropByPosNegLabeld(keys=["MR", "CT", "Segs"], spatial_size=SWIN_size, label_key="Segs",  neg=0),
                RandFlipd(keys=["MR", "CT", "Segs"], spatial_axis=[0], prob=0.25),
                RandFlipd(keys=["MR", "CT", "Segs"], spatial_axis=[1], prob=0.25),
                RandFlipd(keys=["MR", "CT", "Segs"], spatial_axis=[2], prob=0.25),
                RandRotate90d(keys=["MR", "CT", "Segs"], prob=0.25, max_k=3),

                #ToTensord(keys=["MR", "CT", "Segs"])
            ]
        )

        self.prediction_transform = Compose(

            [
                LoadImaged(keys=["MR", "CT", "Segs"]),
                EnsureChannelFirstd(keys=["MR", "CT", "Segs"]),
                Orientationd(keys=["MR", "CT", "Segs"], axcodes='RAI'),
                # redundent but still okay to do, segmentation should be to nearest, while others are bilinear
                Spacingd(keys=["MR", "CT", "Segs"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "nearest")),
                ScaleIntensityd(keys=["MR", "CT"], minv=0.0, maxv=1.0),
                CropForegroundd(keys=["MR", "CT", "Segs"], source_key="MR",
                                k_divisible=SWIN_size, margin="edge"),
                #ToTensord(keys=["MR", "CT", "Segs"])
            ]
        )

    def __len__(self):
        #cases are all paired so one path equals the size of all paths
        return len(self.MR_paths)

    def transform_data(self, data_dict: dict):
        return self.transform(data_dict)


    def __getitem__(self, index):
        # For training
        MR_image = self.MR_paths[index]
        CT_image = self.CT_paths[index]
        Seg_image = self.Seg_paths[index]
        images = {"MR": MR_image, "CT": CT_image, "Segs": Seg_image}
        image_transformed = self.transform_data(images)

        labels = []
        for i in range(8):
            zeros = torch.zeros_like(image_transformed[0]["Segs"])
            zeros[image_transformed[0]["Segs"] == i] = 1
            labels.append(zeros)
        modified_label = torch.stack(labels, dim=1)
        image_transformed[0]["Segs"] = torch.squeeze(modified_label, 0)
        return image_transformed

        # For prediction
        # image = {"image": image_path}
        # return self.prediction_transform(image)

    def get_sample(self, index):
        MR_image = self.MR_paths[index]
        CT_image = self.CT_paths[index]
        Seg_image = self.Seg_paths[index]
        images = {"MR": MR_image, "CT": CT_image, "Segs": Seg_image}
        return self.transform(images)
        #return self.prediction_transform(MR)

