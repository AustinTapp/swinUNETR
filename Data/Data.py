import os
import glob
from torch.utils.data import Dataset
from monai.data import DataLoader
from monai.transforms import (
    LoadImaged,
    OrientationD,
    Compose,
    CropForegroundd,
    CopyItemsd,
    ResizeD,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityD,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
)

class NiftiData(Dataset):
    def __init__(self):
        # for image check
        # self.path = os.path.join(os.getcwd()+'\\Images')

        # for standard training
        self.path = "C:\\Users\\pmilab\\AllieWork\\SSL4N\\Data\\SSL4N_fine_tune"
        self.image_path = glob.glob(self.path + '\\*')

        self.transform = Compose(

            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                OrientationD(keys=["image"], axcodes='RAI'),
                Spacingd(keys=["image"], pixdim=(
                    1.0, 1.0, 1.0), mode=("bilinear")),
                # segmentation change to nearest
                ScaleIntensityD(keys=["image"], minv=0.0, maxv=1.0),
                CropForegroundd(keys=["image"], source_key="image"),
                SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
                RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=1),
                CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
                OneOf(transforms=[
                    RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                                       max_spatial_size=32),
                    RandCoarseDropoutd(keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                                       max_spatial_size=64),
                ]
                ),
                RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
                OneOf(transforms=[
                    RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True,
                                       max_spatial_size=32),
                    RandCoarseDropoutd(keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False,
                                       max_spatial_size=64),
                ]
                ),
                RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),

            ]
        )


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        single_path = self.image_path[index]
        image = {"image": single_path}
        return self.transform(image)

    def get_sample(self):
        return self.image_path

