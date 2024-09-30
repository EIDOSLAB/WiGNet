

import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils


import torch.nn as nn
import sys
import numpy as np
from opt_transfer import get_args_parser
import errno

import wandb
import random

import torch.backends.cudnn as cudnn
import warnings

from model.transfer_models import get_model
from dataloaders.celeba_hq import get_celeba

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import utils 

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
        
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        
        if image.shape[0] <= 1:
            continue

        start_time = time.time()
        image, target = image.to(device), target.to(device)

        model.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))



    return metric_logger


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0

    with torch.inference_mode():
        for i,(image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device)
            target = target.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    
        
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    
    torch.set_num_threads(n_threads)

    
    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.loss.global_avg


def main(args):


    if args.seed is not None:
        set_seed(args.seed)

    model, params, trainable_parameters, n_classes = get_model(
                                                        model_type = args.model_type, 
                                                        use_shift = args.use_shift, 
                                                        adapt_knn= args.adapt_knn, 
                                                        checkpoint = args.checkpoint, 
                                                        freezed = args.freezed, 
                                                        dataset = args.dataset, 
                                                        crop_size=args.crop_size)
    

    if args.num_gpu > 1:
        print('Using DataParallel')
        # model = torch.nn.DataParallel(model)
        model = nn.DataParallel(model)

        print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    print('\n\n--------------------------------------')

    print(f"Parameters: {params}")
    print(f"Trainable Parameters: {trainable_parameters}")
    
    print('--------------------------------------\n\n')

    if not args.test_only:
        wandb.log({
            f'params/parameters':params,
            f'params/trainable_params':trainable_parameters
        })


    if args.save_dir and not args.test_only:
        print(f'# Results will be saved in {args.save_dir}')
        try:
            os.makedirs(args.save_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    print(args)

    assert torch.cuda.is_available(), '# --- Cuda Not Available!!'
    device = 'cuda'
    model.to(device)


    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    if args.crop_size == None:
        crop_size = 224
        if '256' in args.model_type:
            crop_size = 256

        if hasattr(model, 'default_cfg'):
            default_cfg = model.default_cfg
            input_sizes = list(default_cfg['input_size'])
            assert input_sizes[1] == input_sizes[2] and len(input_sizes) == 3, f'input sizes for model {args.model_type} is {input_sizes}'
            crop_size = int(input_sizes[1])
    else:
        crop_size = args.crop_size
    
    

    print(f'\n\n!!!Dataloaders with crop size: {crop_size}\n\n')

    args.distributed = False
    if args.dataset == 'CelebA':
        drop_last=False
        if 'pvig' in args.model_type:
            drop_last=True

        train_loader, valid_loader, test_loader, train_sampler, num_classes, _ = get_celeba(args, get_train_sampler=True, crop_size=crop_size, drop_last=drop_last)
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not yet implemented ')
    
    if valid_loader is not None:
        valid_loader = None if len(valid_loader) == 0 else valid_loader
    

    print(f'Train Dataloader: {len(train_loader)}')
    if valid_loader is not None:
        print(f'Valid Dataloader: {len(valid_loader)}')
    print(f'Test Dataloader: {len(test_loader)}')
    print(f'Founded classes: {num_classes}')

    assert num_classes == n_classes, f'Founded {num_classes} classes in the dataset, while you specified {n_classes}'


    if args.loss == 'cross_entropy': 
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == 'nll':
        criterion = nn.NLLLoss()

    
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "constant":
        main_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs - args.lr_warmup_epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # torch.backends.cudnn.deterministic = True
        test_acc, test_acc_5, test_loss = evaluate(model, criterion, test_loader, device=device)
        print(f'Test acc: {test_acc}')

        return test_acc, test_acc_5, test_loss
    


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(f'\n\n--------\nStarting epoch: {epoch}')
        metric_logger = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args, None, scaler)
        print(f'train acc: {metric_logger.acc1.global_avg}')
        
        wandb.log({
            f'train/train_loss':metric_logger.loss.global_avg,
            f'lr':metric_logger.lr.global_avg,
            f'train/train_acc@1':metric_logger.acc1.global_avg,
            f'train/train_acc@5':metric_logger.acc5.global_avg
        }, step = epoch)
        lr_scheduler.step()
        

        # validate after every epoch
        val_acc = 0.0
        if valid_loader is not None:
            val_acc, val_acc_5, val_loss = evaluate(model, criterion, valid_loader, device=device)

            print(f'valid acc: {val_acc}')
            wandb.log({
                f'val/val_loss':val_loss,
                f'val/val_acc@1':val_acc,
                f'val/val_acc@5':val_acc_5
            }, step = epoch)
        
        
        # evaluate after every epoch
        test_acc, test_acc_5, test_loss = evaluate(model, criterion, test_loader, device=device)
        

        print(f'test acc: {test_acc}')
        wandb.log({
            f'test/test_loss':test_loss,
            f'test/test_acc@1':test_acc,
            f'test/test_acc@5':test_acc_5
        }, step = epoch)

        if args.save_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
                "train_acc": metric_logger.acc1.global_avg,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "params":params
            }
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            
            utils.save_on_master(checkpoint, os.path.join(args.save_dir, "checkpoint.pth"))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\n\n--------------\nEnd training')
    print(f"Training time {total_time_str}")
    print(f'Train acc@1: {metric_logger.acc1.global_avg}')
    print(f'Test acc@1: {test_acc}')

    
    wandb.log({
        f'end/train_acc@1': metric_logger.acc1.global_avg,
        f'end/val_acc@1': val_acc,
        f'end/test_acc@1': test_acc
    })






if __name__ == "__main__":

    # torch.autograd.set_detect_anomaly(True)    

    args = get_args_parser().parse_args()

    print('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    print(args.save_dir)

    if not args.test_only:
        wandb.login()
        wandb_name = f'{args.dataset}/{args.model_type}/train_seed{args.seed}'
        
        wandb.init(
            # Set the project where this run will be logged
            project='WACV_WiGNet_transfer_learning_high_res', 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            config=args)

        args.save_dir += wandb_name    
    else:
        args.save_dir = None


    main(args=args)

    if not args.test_only:
        wandb.run.finish()


