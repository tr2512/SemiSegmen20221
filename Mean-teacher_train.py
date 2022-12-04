from SemiSegmen20221.models import DeeplabV3plus
from SemiSegmen20221.datasets import *
from SemiSegmen20221.configs import cfg
from SemiSegmen20221.utils import setup_logger, IoU, OverallAcc

import torch
import torch.nn as nn
import argparse
import random
import os
import time
from datetime import datetime


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

def consistency_loss(student, teacher):
    return nn.MSELoss(student, teacher)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def eval_func(model, device, logger, optimizer, val_loader, iteration, output_dir):
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
    results = "\n" + f"{model} | Overall acc: " + str(acc) + " Mean IoU: " + str(mean_iou) + "Learning rate: " + str(optimizer.param_groups[0]['lr']) + "\n"
    for i, iou in enumerate(ious):
        results += f"{model} Class " + str(i) + " IoU: " + str(iou.item()) + "\n"
    results = results[:-2]
    logger.info(results)
    torch.save({f"{model}_state_dict": model.state_dict(), 
                "iteration": iteration,
                }, os.path.join(output_dir, f"current_{model}.pkl"))
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save({f"{model}_state_dict": model.state_dict(), 
                "iteration": iteration,
                }, os.path.join(output_dir, f"best_{model}.pkl"))
    logger.info(f"Best {model} iou so far: " + str(best_iou))


def train(cfg, logger):
    best_iou = 0
    logger.info("Begin the training process")
    device = torch.device(cfg.MODEL.DEVICE)

    #Create models
    student = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)
    student.to(device)

    teacher = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)
    for param in teacher.parameters():
        param.detach_()
    teacher.to(device)

    #Create opitimizer
    optimizer = torch.optim.SGD([p for p in student.parameters() if p.requires_grad], 
    lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer.zero_grad()

    output_dir = cfg.OUTPUT_DIR
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    max_iter = cfg.SOLVER.MAX_ITER
    stop_iter = cfg.SOLVER.STOP_ITER
    iteration = 0

    #Load datasets
    img_list = None
    if cfg.DATASETS.TRAIN_LIST:
        img_list = read_file(cfg.DATASETS.TRAIN_LIST)

    train_data = VOCDataset(cfg.DATASETS.TRAIN_IMGDIR, cfg.DATASETS.TRAIN_LBLDIR,
                            img_list=img_list,
                            transformation=Compose([
                            ToTensor(), 
                            Normalization(), 
                            RandomScale(cfg.INPUT.MULTI_SCALES), 
                            RandomCrop(cfg.INPUT.CROP_SIZE), 
                            RandomFlip(cfg.INPUT.FLIP_PROB)]))
    val_data = VOCDataset(cfg.DATASETS.VAL_IMGDIR, cfg.DATASETS.VAL_LBLDIR, transformation=
                         Compose([ToTensor(), Normalization(), RandomCrop(cfg.INPUT.CROP_SIZE)]))
    
    logger.info("Number of train images: " + str(len(train_data)))
    logger.info("Number of validation images: " + str(len(val_data)))
    
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
    alpha = 0.5

    #Start training
    logger.info("Start training")
    logger.info(f"Smoothing coefficient: {alpha}")
    student.train()
    end = time.time()

    while iteration < stop_iter:
        for i, (images, labels) in enumerate(train_loader):
            student.train()
            data_time = time.time() - end
            end = time.time()

            optimizer.param_groups[0]['lr'] = cfg.SOLVER.LR * (1 - iteration/max_iter)**cfg.SOLVER.POWER

            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            student_preds = student(images)
            teacher_preds = teacher(images)

            student_class_loss = criterion(student_preds, labels)
            teacher_class_loss = criterion(teacher_preds, labels)

            cons_loss = consistency_loss(student_class_loss, teacher_class_loss)
            logger.info("Iter [%d/%d] Consistency_loss: %f" % (interation,
                        cfg.SOLVER.STOP_ITER, cons_loss))

            student_class_loss.backward()
            optimizer.step()

            iteration += 1

            update_ema_variables(student, teacher, alpha, iteration)

            if iteration % 20 == 0:
                logger.info("Iter [%d/%d] Student_loss: %f Time/iter: %f" % (iteration, 
                            cfg.SOLVER.STOP_ITER, student_class_loss, data_time))
                logger.info("Iter [%d/%d] Teacher_loss: %f Time/iter: %f" % (iteration, 
                            cfg.SOLVER.STOP_ITER, teacher_class_loss, data_time))
            #Evaluation
            if iteration % 1000 == 0:
                eval_func(model=student, device=device, logger=logger, optimizer=optimizer,
                          val_loader=val_loader, iteration=iteration, output_dir=output_dir)
                eval_func(model=teacher, device=device, logger=logger, optimizer=optimizer,
                          val_loader=val_loader, iteration=iteration, output_dir=output_dir)
        
            if iteration == stop_iter:
                break

    return teacher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch training")
    parser.add_argument("--config", default="")
    args = parser.parse_args()
    cfg = combine_cfg(args.config)
    print(cfg.OUTPUT_DIR)
    logger = setup_logger("Mean-teacher semi-supervised", cfg.OUTPUT_DIR, str(datetime.now()) + ".log")
    teacher_model = train(cfg, logger)