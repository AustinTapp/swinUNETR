# swinUNETR
Self/Semi-Supervised Learning for Neonate Brain Segmentation

This repository is primarily developed for the use of self and semi supervised learning for the eventual segmentation of gray and white matter from neonate MRI volumes.

Reimplementation of https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain

Built primarily with MONAI (https://github.com/Project-MONAI/MONAI) modules and pytorch lightning (https://www.pytorchlightning.ai) backend.

To install requirements using the .toml file, open a prompt (Anaconda) and navigat [cd] to the directory containing this repo. Run: conda install -c conda-forge poetry
Once poetry is installed, run: 'python -m poetry install'

Based off:
https://github.com/Project-MONAI/tutorials/blob/main/self_supervised_pretraining/ssl_script_train.py
