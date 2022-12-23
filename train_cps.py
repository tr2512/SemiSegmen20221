import torch
import torch.nn as nn
import argparse
import random
import os
import time
from datetime import datetime

random.seed(0)
torch.manual_seed(0)

from configs import cfg
from datasets import *
from models import DeeplabV3plus
from utils import setup_logger, IoU, OverallAcc

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

class Network(nn.Module):

    def __init__(self, cfg):
        super(Network, self).__init__()
        self.network1 = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)
        self.network2 = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)

    def forward(self, x, step=1):
        if not self.training:
            return self.network1(x)
        if step == 1:
            return self.network1(x)
        elif step == 2:
            return self.network2(x)
        else:
            print("Invalid step")

def train(cfg, logger, pretrain = None ,checkpoint = None, output_dir= None):
    best_iou = 0
    logger.info("Begin the training process")

    device = torch.device(cfg.MODEL.DEVICE)

    model = Network(cfg)
    model.to(device)
    
    iteration = 0

    max_iter = cfg.SOLVER.MAX_ITER
    stop_iter = cfg.SOLVER.STOP_ITER

    optimizer_l = torch.optim.SGD([p for p in model.network1.parameters() if p.requires_grad], lr=0.0009375, 
                                    momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_r = torch.optim.SGD([p for p in model.network2.parameters() if p.requires_grad], lr=0.0009375, 
                                    momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


    #Load datasets
    lbl_img_list = read_file(cfg.DATASETS.LABEL_LIST)
    ulbl_img_list = read_file(cfg.DATASETS.UNLABELLED_LIST)

    lbl_list = []
    ulbl_list = []
    while len(lbl_list) < max_iter*4:
        lbl_list += lbl_img_list
        random.shuffle(lbl_img_list)
    lbl_list = lbl_list[:max_iter*4]
    
    while len(ulbl_list) < max_iter*4:
        ulbl_list += ulbl_img_list
        random.shuffle(ulbl_img_list)
    ulbl_list = ulbl_list[:max_iter*4]
    
    lbl_train_data = VOCDataset(cfg.DATASETS.TRAIN_IMGDIR, cfg.DATASETS.TRAIN_LBLDIR,
                            img_list=lbl_list,
                            transformation=Compose([
                            ToTensor(), 
                            Normalization(), 
                            RandomScale(cfg.INPUT.MULTI_SCALES), 
                            RandomCrop(cfg.INPUT.CROP_SIZE), 
                            RandomFlip(cfg.INPUT.FLIP_PROB)]))

    ulbl_train_data = VOCDataset(cfg.DATASETS.TRAIN_IMGDIR, cfg.DATASETS.TRAIN_LBLDIR,
                            img_list=ulbl_list,
                            transformation=Compose([ 
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
        batch_size=3,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    unsupervised_loader = torch.utils.data.DataLoader(
        ulbl_train_data,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    cps_weight = 1.5

    logger.info("Start training")
    model.train()
    end = time.time()

    while iteration < stop_iter:
        for (imgs, lbls), (u_imgs, _) in zip(supervised_loader, unsupervised_loader):

            imgs, lbls, u_imgs = imgs.to(device), lbls.to(device), u_imgs.to(device)
            model.train()
            data_time = time.time() - end
            end = time.time()
            
            optimizer_l.param_groups[0]['lr'] = 0.0009375 * (1 - iteration/max_iter)**0.9
            optimizer_r.param_groups[0]['lr'] = 0.0009375 * (1 - iteration/max_iter)**0.9     

            optimizer_l.zero_grad()
            optimizer_r.zero_grad()

            pred_sup_l = model(imgs, step=1)
            pred_sup_r = model(imgs, step=2)
            pred_usup_l = model(u_imgs, step=1)
            pred_usup_r = model(u_imgs, step=2)

            pred_l = torch.cat([pred_sup_l, pred_usup_l], dim=0)
            pred_r = torch.cat([pred_sup_r, pred_usup_r], dim=0)

            max_l = pred_l.argmax(dim=1).long()
            max_r = pred_r.argmax(dim=1).long()
            
            cps_loss_l = criterion(pred_l, max_r)
            cps_loss_r = criterion(pred_r, max_l)
            cps_loss = cps_loss_l + cps_loss_r
            cps_loss *= cps_weight

            loss_sup_l = criterion(pred_sup_l, lbls)
            loss_sup_r = criterion(pred_sup_r, lbls)

            loss = loss_sup_l + loss_sup_r + cps_loss
            loss.backward()
            
            optimizer_l.step()
            optimizer_r.step()

            iteration += 1

            if iteration % 20 == 0:
                logger.info("Iter [%d/%d] Left CE Loss: %f Right CE Loss: %f Left CPS_loss: %f Right CPS_loss: %f Time/iter: %f" % (iteration, 
                stop_iter, loss_sup_l, loss_sup_r, cps_loss_l, cps_loss_r, data_time))
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
                results = "\n" + "Overall acc: " + str(acc) + " Mean IoU: " + str(mean_iou) + "Learning rate: " + str(optimizer_l.param_groups[0]['lr']) + "\n"
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    
    os.mkdir("logs/cps")
    logger = setup_logger("CPS", "logs/cps" , str(datetime.now()) + ".log")
    model = train(cfg, logger,args.pretrain, args.checkpoint , output_dir= "logs/cps")
