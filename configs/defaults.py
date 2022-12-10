import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = "logs/train_fully"
_C.MODEL = CN()
_C.MODEL.NAME = "deeplabv3+_resnet50"
_C.MODEL.NUM_CLASSES = 21
_C.MODEL.ATROUS = [6, 12, 18]
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.WEIGHTS = ""
_C.MODEL.FREEZE_BN = False

_C.INPUT = CN()
_C.INPUT.MULTI_SCALES = [0.5, 0.75, 1, 1.5, 2]
_C.INPUT.CROP_SIZE = (512, 512)
_C.INPUT.FLIP_PROB = 0.5
_C.INPUT.DSBN = 2
_C.INPUT.SDA = 3

_C.DATASETS = CN()
_C.DATASETS.TRAIN_IMGDIR = "pascal_voc/train_aug/image"
_C.DATASETS.TRAIN_LBLDIR = "pascal_voc/train_aug/label"
_C.DATASETS.VAL_IMGDIR = "pascal_voc/val/image"
_C.DATASETS.VAL_LBLDIR = "pascal_voc/val/label"
_C.DATASETS.LABEL_LIST = "pascal_voc/subset_train_aug/train_aug_labeled_1-8.txt"
_C.DATASETS.UNLABELLED_LIST = "pascal_voc/subset_train_aug/train_aug_unlabeled_1-8.txt"
_C.DATASETS.TRAIN_LIST = ""

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.STOP_ITER = 40000

_C.SOLVER.LR_METHOD = "poly"
_C.SOLVER.LR = 0.005
_C.SOLVER.POWER = 0.9

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4

_C.SOLVER.BATCH_SIZE = 8 
_C.SOLVER.STRONG_AUG = 2
