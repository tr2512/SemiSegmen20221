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

def read_file(directory):
    l = []
    with open(directory, "r") as f:
        for line in f.readlines():
            l.append(line[:-1] + ".jpg")
    return l

def consistency_loss(student, teacher):
    return nn.MSELoss()(student, teacher)

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
    global student_best_iou, teacher_best_iou
    student_best_iou, teacher_best_iou = 0, 0

    output_dir = '/content/drive/MyDrive/Intro_DL/mean-teacher_train/check_points'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    device = torch.device(cfg.MODEL.DEVICE)

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
                            transformation=Compose([
                            ToTensor(), 
                            Normalization(), 
                            RandomScale(cfg.INPUT.MULTI_SCALES), 
                            RandomCrop(cfg.INPUT.CROP_SIZE), 
                            RandomFlip(cfg.INPUT.FLIP_PROB)]))
    val_data = VOCDataset(cfg.DATASETS.VAL_IMGDIR, cfg.DATASETS.VAL_LBLDIR, transformation=
                         Compose([ToTensor(), Normalization(), RandomCrop(cfg.INPUT.CROP_SIZE)]))
    
    lbl_train_loader = torch.utils.data.DataLoader(
        lbl_train_data,
        batch_size=cfg.SOLVER.BATCH_SIZE//2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    ulbl_train_loader = torch.utils.data.DataLoader(
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

    #Create models
    student = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)
    teacher = DeeplabV3plus(cfg.MODEL.ATROUS, cfg.MODEL.NUM_CLASSES)

    #Check for checkpoints
    if os.path.exists(os.path.join(output_dir, "current_teacher.pkl")):
        checkpoint = torch.load(os.path.join(output_dir, "current_teacher.pkl"))
        teacher.load_state_dict(checkpoint['teacher_state_dict'])
        teacher_best_iou = checkpoint['best_iou']

        checkpoint = torch.load(os.path.join(output_dir, "current_student.pkl"))
        student.load_state_dict(checkpoint['student_state_dict'])
        student_best_iou = checkpoint['best_iou']

        iteration = checkpoint['iteration']
        logger.info("Continue training from last checkpoint")
    else:
        logger.info("Begin the training process")
        iteration = 0

        logger.info("Number of train images: " + str(len(lbl_train_data) + len(ulbl_train_data)))
        logger.info("Number of validation images: " + str(len(val_data)))

    #Load models to device
    for param in teacher.parameters():
        param.detach_()
    student.to(device)
    teacher.to(device)

    #Create opitimizer
    optimizer = torch.optim.SGD([p for p in student.parameters() if p.requires_grad], 
    lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer.zero_grad()

    max_iter = cfg.SOLVER.MAX_ITER
    stop_iter = cfg.SOLVER.STOP_ITER

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    alpha = 0.999

    #Start training
    logger.info("Start training")
    logger.info(f"Smoothing coefficient: {alpha}")
    student.train()
    end = time.time()

    while iteration < stop_iter:
        for (images, labels), (u_imgs, _) in zip(lbl_train_loader, ulbl_train_loader):
            student.train()
            data_time = time.time() - end
            end = time.time()

            optimizer.param_groups[0]['lr'] = cfg.SOLVER.LR * (1 - iteration/max_iter)**cfg.SOLVER.POWER

            optimizer.zero_grad()
            images = torch.cat((images, u_imgs), 0)
            images = images.to(device)
            labels = labels.to(device)

            student_preds = student(images)
            teacher_preds = teacher(images)
            
            cons_loss = consistency_loss(student_preds, teacher_preds)
            
            student_preds, _ = torch.split(student_preds, [4, 4])
            teacher_preds, _ = torch.split(teacher_preds, [4, 4])

            student_class_loss = criterion(student_preds, labels)
            teacher_class_loss = criterion(teacher_preds, labels)

            loss = cons_loss + student_class_loss

            loss.backward()
            optimizer.step()

            iteration += 1

            update_ema_variables(student, teacher, alpha, iteration)

            if iteration % 20 == 0:
                logger.info("Iter [%d/%d] Student_loss: %f Time/iter: %f" % (iteration, 
                            cfg.SOLVER.STOP_ITER, student_class_loss, data_time))
                logger.info("Iter [%d/%d] Teacher_loss: %f Time/iter: %f" % (iteration, 
                            cfg.SOLVER.STOP_ITER, teacher_class_loss, data_time))
                logger.info("Iter [%d/%d] Consistency_loss: %f" % (iteration,
                            cfg.SOLVER.STOP_ITER, cons_loss))
            #Evaluation
            if iteration % 400 == 0:
                logger.info("Validation mode")
                eval_func(model=student, device=device, logger=logger, optimizer=optimizer,
                          val_loader=val_loader, iteration=iteration, output_dir=output_dir)
                eval_func(model=teacher, device=device, logger=logger, optimizer=optimizer,
                          val_loader=val_loader, iteration=iteration, output_dir=output_dir, teacher=True)
        
            if iteration == stop_iter:
                break

    return teacher


if __name__ == "__main__":
    logger = setup_logger("Mean-teacher semi-supervised", '/content/drive/MyDrive/Intro_DL/mean-teacher_train/logs', str(datetime.now()) + ".log")
    teacher_model = train(cfg, logger)
