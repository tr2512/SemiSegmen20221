import torch
import torch.nn as nn
import argparse
import random
import os
import time
from datetime import datetime

from configs import cfg
from datasets import *
from datasets.strong_aug import SDA
from models import DeeplabV3plus
from utils import setup_logger, IoU, OverallAcc
from models.batch_norm import DSBN

def combine_cfg(config_dir=None):
    cfg_base = cfg.clone()
    if config_dir:
        cfg_base.merge_from_file(config_dir)
    return cfg_base 

def read_file(directory):
    l = []
    with open(directory, "r") as f:
        for line in f.readlines():
            l.append(line[:-1] + ".jpg")
    return l

def train(cfg, logger, pretrain , output_dir):
    best_iou = 0
    logger.info("Begin the training process")

    device = torch.device(cfg.MODEL.DEVICE)

    convert = DSBN(cfg.MODEL.NUM_CLASSES)
    
    saved = torch.load(pretrain)
    teacher_model = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)
    teacher_model.load_state_dict(saved['model_state_dict'])
    teacher_model.to(device)

    model = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)
    model.to(device)

    max_iter = cfg.SOLVER.MAX_ITER
    stop_iter = cfg.SOLVER.STOP_ITER

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
    lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer.zero_grad()

    iteration = 0

    #Load datasets
    lbl_img_list = read_file(cfg.DATASETS.LABEL_LIST)
    ulbl_img_list = read_file(cfg.DATASETS.UNLABELLED_LIST)

    lbl_train_data = VOCDataset(cfg.DATASETS.TRAIN_IMGDIR, cfg.DATASETS.TRAIN_LBLDIR,
                            img_list=lbl_img_list,
                            transformation=Compose([
                            ToTensor(), 
                            Normalization(), 
                            RandomScale(cfg.INPUT.MULTI_SCALES), 
                            RandomCrop(cfg.INPUT.CROP_SIZE), 
                            RandomFlip(cfg.INPUT.FLIP_PROB)]))
    ulbl_train_data = VOCDataset(cfg.DATASETS.TRAIN_IMGDIR, cfg.DATASETS.TRAIN_LBLDIR,
                            img_list=ulbl_img_list,
                            transformation=Compose( list(SDA(cfg.INPUT.SDA)) + [ 
                            ToTensor(), 
                            Normalization(), 
                            RandomScale(cfg.INPUT.MULTI_SCALES), 
                            RandomCrop(cfg.INPUT.CROP_SIZE), 
                            RandomFlip(cfg.INPUT.FLIP_PROB)]))
    val_data = VOCDataset(cfg.DATASETS.VAL_IMGDIR, cfg.DATASETS.VAL_LBLDIR, transformation=
                         Compose([ToTensor(), Normalization(), RandomCrop(cfg.INPUT.CROP_SIZE)]))
    logger.info("Number of labeled train images: " + str(len(lbl_train_data)))
    logger.info("Number of unlabeled train images: " + str(len(ulbl_train_data)))
    logger.info("Number of validation images: " + str(len(val_data)))
    
    supervised_loader = torch.utils.data.DataLoader(
        lbl_train_data,
        batch_size=cfg.SOLVER.BATCH_SIZE//2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    unsupervised_loader = torch.utils.data.DataLoader(
        ulbl_train_data,
        batch_size=cfg.SOLVER.BATCH_SIZE//2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    logger.info("Start training")
    model.train()
    end = time.time()

    while iteration < stop_iter:
        for (images, labels), (u_imgs, _) in zip(supervised_loader, unsupervised_loader):
            model.train()
            data_time = time.time() - end
            end = time.time()

            optimizer.param_groups[0]['lr'] = cfg.SOLVER.LR * (1 - iteration/max_iter)**cfg.SOLVER.POWER

            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            u_imgs = u_imgs.to(device)

            preds = model(images)
            
            loss = criterion(preds, labels)
            loss.backward()

            optimizer.step()

            iteration += 1

            if iteration % 20 == 0:
                logger.info("Iter [%d/%d] Loss: %f Time/iter: %f" % (iteration, 
                cfg.SOLVER.STOP_ITER, loss, data_time))
            if iteration % 1000 == 0:
                logger.info("Validation mode")
                model.eval()
                intersections = torch.zeros(21).to(device)
                unions = torch.zeros(21).to(device)
                rights = 0
                totals = 0
                for imgs, lbls in val_loader:

                    imgs = imgs.to(device)
                    lbls = lbls.to(device)

                    with torch.no_grad():
                        preds = model(imgs)
                        preds = preds.argmax(dim=1)
                        intersection, union = IoU(preds, lbls, 21)
                        intersections += intersection
                        unions += union
                        right, total = OverallAcc(preds, lbls, 21)
                        rights += right
                        totals += total

                ious = intersections / unions
                mean_iou = torch.mean(ious).item()
                acc = rights / totals
                results = "\n" + "Overall acc: " + str(acc) + " Mean IoU: " + str(mean_iou) + "Learning rate: " + str(optimizer.param_groups[0]['lr']) + "\n"
                for i, iou in enumerate(ious):
                    results += "Class " + str(i) + " IoU: " + str(iou.item()) + "\n"
                results = results[:-2]
                logger.info(results)
                torch.save({"model_state_dict": model.state_dict(), 
                            "iteration": iteration,
                            }, os.path.join(output_dir, "current_model.pkl"))
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    torch.save({"model_state_dict": model.state_dict(), 
                            "iteration": iteration,
                            }, os.path.join(output_dir, "best_model.pkl"))
                logger.info("Best iou so far: " + str(best_iou))
            if iteration == stop_iter:
                break
    return model 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch training")
    parser.add_argument("--pretrain", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    
    logger = setup_logger("Fully supervised", args.pretrain , str(datetime.now()) + ".log")
    model = train(cfg, logger, args.pretrain, args.output_dir)