import torch
import torch.nn as nn 

class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, lbl_dir, img_list=None, transformation=None):