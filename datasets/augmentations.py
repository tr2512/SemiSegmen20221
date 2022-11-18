import random 
import torch 
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

class RandomCrop(object):

    def __init__(self, size, ignore_index=255):
        self.size = size 
        self.ignore_index = ignore_index 

    def __call__(self, img, lbl):
        h, w = img.size()[1:]
        h_pos = 0
        w_pos = 0
        if self.size[0] < h:
            h_pos = random.randint(0, h - self.size[0] + 1)
        if self.size[1] < w:
            w_pos = random.randint(0, w - self.size[1] + 1)
        img_crop = img[:, h_pos:h_pos + self.size[0], w_pos:w_pos + self.size[1]]
        lbl_crop = lbl[h_pos:h_pos + self.size[0], w_pos:w_pos + self.size[1]]
        img_crop = self._padding(img_crop, 0)
        lbl_crop = self._padding(lbl_crop, self.ignore_index)
        return img_crop, lbl_crop

    def _padding(self, img, pad_value):
        h = (self.size[0] - img.size()[-2]) // 2
        w = (self.size[1] - img.size()[-1]) // 2
        if len(img.size()) == 2:
            padded = torch.ones(self.size, dtype=torch.int64) * pad_value
            padded[h:h + img.size()[0], w:w + img.size()[1]] = img
        else:
            padded = torch.ones((img.size()[0], self.size[0], self.size[1])) * pad_value
            padded[:, h:h + img.size()[1], w:w + img.size()[2]] = img
        return padded
 

class RandomFlip(object):

    def __init__(self, p=0.5):
        self.p = p 

    def __call__(self, img, lbl):
        if random.random() > self.p:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)
        if random.random() > self.p:
            img = TF.vflip(img)
            lbl = TF.vflip(lbl)
        return img, lbl

class ToTensor(object):

    def __init__(self):
        pass 
    
    def __call__(self, img, lbl):
        return torch.from_numpy(img / 255).permute(2, 0, 1), torch.from_numpy(lbl).long()


class RandomScale(object):

    def __init__(self, scales=[0.5, 0.75, 1, 1.5, 2]):
        self.scales = scales 

    def __call__(self, img, lbl):
        scale = random.sample(self.scales, 1)[0]
        h, w = int(img.size()[1] * scale), int(img.size()[2] * scale)
        img = TF.resize(img, (h, w), interpolation=InterpolationMode.BILINEAR)
        lbl = torch.squeeze(TF.resize(torch.unsqueeze(lbl, 0), (h, w), interpolation=InterpolationMode.NEAREST))
        return img, lbl


class Normalization(object):

    def __init__(self):
        pass

    def __call__(self, img, lbl):
        return (img - 0.5) / 0.5, lbl