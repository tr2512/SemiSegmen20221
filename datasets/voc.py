import os
import torch
import numpy as np
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, lbl_dir, img_list=None, transformation=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        if img_list:
            self.img_list = img_list
        else:
            self.img_list = os.listdir(img_dir)
        self.transform = transformation
        self.color_map = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128], [128, 128, 0],
                [128, 128, 128], [0, 0, 64], [0, 0, 192], 
                [0, 128, 64], [0, 128, 192], [128, 0, 64], 
                [128, 0, 192], [128, 128, 64],
                [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0], ]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, img_name)), dtype=np.float32)
        lbl = np.array(Image.open(os.path.join(self.lbl_dir, img_name[:-3] + "png")), dtype=np.int_)
        if self.transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def decode_segmap(self, lbl):
        pass
        
        