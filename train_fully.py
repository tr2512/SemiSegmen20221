import torch
import torch.nn as nn

import random
import os
import time
from datetime import datetime

from configs import cfg
from datasets import *
from models import DeeplabV3plus
from utils import setup_logger, IoU, OverallAcc

def combine_cfg(config_dir=None):
    cfg_base = cfg.clone()
    if config_dir:
        cfg_base.merge_from_file(config_dir)
    return cfg_base 


def train(cfg, logger):
    best_iou = 0
    logger.info("Begin the training process")

    device = torch.device(cfg.MODEL.DEVICE)

    model = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)
    model.to(device)

    max_iter = cfg.SOLVER.MAX_ITER
    stop_iter = cfg.SOLVER.STOP_ITER

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
    lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer.zero_grad()

    output_dir = cfg.OUTPUT_DIR
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    iteration = 0

    train_data = VOCDataset(cfg.DATASETS.TRAIN_IMGDIR, cfg.DATASETS.TRAIN_LBLDIR,
                            transformation=Compose([
                            ToTensor(), 
                            Normalization(), 
                            RandomScale(cfg.INPUT.MULTI_SCALES), 
                            RandomCrop(cfg.INPUT.CROP_SIZE), 
                            RandomFlip(cfg.INPUT.FLIP_PROB)]))
    val_data = VOCDataset(cfg.DATASETS.VAL_IMGDIR, cfg.DATASETS.VAL_LBLDIR, transformation=
                         Compose([ToTensor(), Normalization(), RandomCrop(cfg.INPUT.CROP_SIZE)]))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.SOLVER.BATCH_SIZE,
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
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            data_time = time.time() - end
            end = time.time()

            optimizer.param_groups[0]['lr'] = cfg.SOLVER.LR * (1 - iteration/max_iter)**cfg.SOLVER.POWER

            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

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
            if iteration == stop_iter:
                break
    return model 

if __name__ == "__main__":
    cfg = combine_cfg()
    logger = setup_logger("Fully supervised", cfg.OUTPUT_DIR, str(datetime.now()) + ".log")
    model = train(cfg, logger)
