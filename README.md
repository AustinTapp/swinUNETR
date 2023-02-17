# swinUNETR
Supervised Learning for Simultaneous Neonate Skull Segmentation and MR to CT Modality Transfer.

This repository is primarily developed for the use of skull bone segmentations (7 labels), suture segmentation (1 label) and conversion from an input MR to a CT modality.
Note that the CT modality is Geodisically manipulated using suture seed points to apply a unqiue contrast loss between the CT reconstruction and the CT ground truth, which has been geodiscally manipulated. Thus, the image intensties output from the CT are not typical for true CT, but rather a CT that has had its intensity values adjusted based on the sutures. This is to emphasize the suture areas and ensure (later on) that segmentations for the skull do not overtake those of the suture, regardless of if the suture is closed or not.

Geodisic transform: https://github.com/AustinTapp/FastGeodis

Adapted implimentation of https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain

Built primarily with MONAI (https://github.com/Project-MONAI/MONAI) modules and pytorch lightning (https://www.pytorchlightning.ai) backend.

