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
    RandSpatialCropSamplesd,
    ScaleIntensityd,
    Spacingd,
    ToTensord
)

def test_transform(dict_keys):
    return Compose(
            [
                LoadImaged(keys=dict_keys, allow_missing_keys=True),
                EnsureChannelFirstd(keys=dict_keys, allow_missing_keys=True),
                Orientationd(keys=dict_keys, axcodes='RAI', allow_missing_keys=True),
                ScaleIntensityd(keys=dict_keys, allow_missing_keys=True),
            ]
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

        self.transform = Compose(

            [
                LoadImaged(keys=["MR", "CT"]),
                EnsureChannelFirstd(keys=["MR", "CT"]),
                Orientationd(keys=["MR", "CT"], axcodes='RAI'),
                ScaleIntensityd(keys=["MR", "CT"], minv=0.0, maxv=1.0),
                CropForegroundd(keys=["MR", "CT"], source_key="CT"),
                RandSpatialCropSamplesd(keys=["MR", "CT"], roi_size=(96, 96, 96), num_samples=3, random_size=False),
                RandFlipd(keys=["MR", "CT"], spatial_axis=[0], prob=0.25),
                RandFlipd(keys=["MR", "CT"], spatial_axis=[1], prob=0.25),
                RandFlipd(keys=["MR", "CT"], spatial_axis=[2], prob=0.25),
                RandRotate90d(keys=["MR", "CT"], prob=0.25, max_k=3),
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
        images = {"MR": MR_image, "CT": CT_image}
        image_transformed = self.transform_data(images)

        # labels = []
        # for i in range(8):
        #     zeros = torch.zeros_like(image_transformed[0]["Segs"])
        #     zeros[image_transformed[0]["Segs"] == i] = 1
        #     labels.append(zeros)
        # modified_label = torch.stack(labels, dim=1)
        # image_transformed[0]["Segs"] = torch.squeeze(modified_label, 0)
        return image_transformed

        # For prediction
        # image = {"image": image_path}
        # return self.prediction_transform(image)

    def get_sample(self, index):
        MR_image = self.MR_paths[index]
        CT_image = self.CT_paths[index]
        images = {"MR": MR_image, "CT": CT_image}
        return self.transform(images)
        #return self.prediction_transform(MR)
