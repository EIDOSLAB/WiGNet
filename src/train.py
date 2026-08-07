import warnings

# warnings.filterwarnings("ignore")

import argparse
import time
import os
import sys
import logging
import random
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import torch.backends.cudnn as cudnn
import numpy as np

from timm.data import (
    ImageDataset as Dataset,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)  # , create_loader
from timm.models import create_model, resume_checkpoint  # , convert_splitbn_model
from timm.utils import *
import random
from timm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    JsdCrossEntropy,
)
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from types import SimpleNamespace

from timm.utils import ApexScaler, NativeScaler
from data import create_loader

from opt import parse_args
import wandb

_logger = logging.getLogger("train")

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False
_logger.info(f"\nhas_apex: {has_apex}")

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass
_logger.info(f"has_native_amp: {has_native_amp}\n")

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def load_pretrained_weights(model, args):
    assert (
        "wignext" in args.model
    ), "Pre-trained checkpoint is only available for wignet models"
    _logger.info(f"\nLoading pretrained model from {args.pretrain_path}")

    state_dict = torch.load(args.pretrain_path)["state_dict"]
    model_dict = model.state_dict()

    if args.strict_model_loading:
        model.load_state_dict(state_dict, strict=True)
    else:
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                filtered_dict[k] = v
            else:
                try:
                    _logger.info(
                        f"Skipping module due to shape missmatch: {k} -> pre-trtained shape: {v.shape} - model shape: {model_dict[k].shape}"
                    )
                except KeyError:
                    _logger.info(
                        f"Skipping module due to shape missmatch: {k} -> pre-trtained shape: {v.shape} - model has no such key"
                    )
        model_dict.update(filtered_dict)
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

        # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # _logger.info("\nMissing keys:")
        # for k in missing_keys:
        #     _logger.info(f"  - {k}")
        # _logger.info("\nUnexpected keys:")
        # for k in unexpected_keys:
        #     _logger.info(f"  - {k}")

    _logger.info(f"\n***** Successfully loaded pretrained model. *****\n")
    return model


def get_model(args):
    if "wignext" in args.model:

        if "wignn_ln" in args.model:
            img_size = (args.img_size, args.img_size)
        else:
            img_size = args.img_size

        return create_model(
            args.model,
            img_size=img_size,
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path,
            knn=args.knn,
            window_size=args.window_size,
            use_shifts=args.use_shift,
            adapt_knn=args.adapt_knn,
        )
    return create_model(
        args.model,
        pretrained=args.pretrained,  # K: args.pretrained is for non-wignet models only.
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint,
    )


def setup_wandb(args):
    try:
        dataset_name = args.dataset_name
    except:
        dataset_name = None
        # TODO: Get dataset name from args
    if "wignext" in args.model:
        from_pretrained = not (args.pretrain_path == "" or args.pretrain_path is None)
        run_name = f"m{args.model}_pt{from_pretrained}_d{dataset_name}_shift{args.use_shift}_k{args.knn}_adapt{args.adapt_knn}_ws{args.window_size}"
        if args.job_name is not None:  # TODO: Include job name more cleanly
            run_name = f"{args.job_name}_{run_name}"
    else:
        from_pretrained = (
            args.pretrained
        )  # K: args.pretrained is for non-wignet models only.
        raise NotImplementedError("wandb logging not implemented for non-wignet models")
        # TODO: Wandb logging for non-wignet models
    wandb.init(
        project=args.wandb_project_name,
        name=run_name,
        config=args,
    )


def set_all_seeds():
    # TODO: Get seed from args
    try:
        random.seed(42)
    except:
        # K:
        print(
            "\nrandom.seed(42) failed. Most probably because random is from timm not python:"
        )
        print("random.__file__:", random.__file__)
        print(
            'Most probably this happened bacause "from timm.utils import *" is below "import random"'
        )
        print(
            "Setting seed for timm.utils.random using random_seed() function instead.\n"
        )
        random_seed(
            42
        )  # See https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/random.py
    torch.manual_seed(42)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = str(42)
    torch.cuda.manual_seed_all(42)


from typing import Dict, Any


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # K: from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_helpers.py#L28
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training

    def _remove_prefix(text, prefix):
        # FIXME replace with 3.9 stdlib fn when min at 3.9
        if text.startswith(prefix):
            return text[len(prefix) :]
        return text

    cleaned_state_dict = {}
    to_remove = (
        "module.",  # DDP wrapper
        "_orig_mod.",  # torchcompile dynamo wrapper
    )
    for k, v in state_dict.items():
        for r in to_remove:
            k = _remove_prefix(k, r)
        cleaned_state_dict[k] = v
    return cleaned_state_dict


def resume_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,  # args.resume is passed (which is a .pth.tar)
    optimizer: torch.optim.Optimizer = None,
    loss_scaler: Any = None,
    log_info: bool = True,
    args: argparse.Namespace = None,
):
    # K: from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_helpers.py#L28  (and then modified a bit)
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if log_info:
                _logger.info("Restoring model state from checkpoint...")
            state_dict = clean_state_dict(checkpoint["state_dict"])
            # model.load_state_dict(state_dict)
            args.pretrain_path = checkpoint_path
            model = load_pretrained_weights(model, args)

            if optimizer is not None and "optimizer" in checkpoint:
                if not args.evaluate:
                    if log_info:
                        _logger.info("Restoring optimizer state from checkpoint...")
                    optimizer.load_state_dict(checkpoint["optimizer"])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info("Restoring AMP loss scaler state from checkpoint...")
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if "epoch" in checkpoint:
                resume_epoch = checkpoint["epoch"]
                if "version" in checkpoint and checkpoint["version"] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

                if log_info:
                    _logger.info(
                        "Loaded checkpoint '{}' (epoch {})".format(
                            checkpoint_path, checkpoint["epoch"]
                        )
                    )
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def main():
    setup_default_logging()
    args, args_text = parse_args()

    if args.evaluate:
        set_all_seeds()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                "Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1."
            )
            args.num_gpu = 1

    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.init_method,
            rank=args.rank,
            world_size=args.world_size,
        )
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

        assert not args.no_wandb, "There must be wandb logging in distributed mode."

    assert args.rank >= 0

    if args.local_rank == 0 and not args.evaluate:
        if args.no_wandb:
            _logger.info("The --no-wandb flag is present: No wandb logging")
        else:
            setup_wandb(args)

    if args.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        _logger.info("Training with a single process on %d GPUs." % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    ################## Get Model ##################
    model = get_model(args)

    ################## Load pre-trained checkpoint ##################
    """ K:
    args.pretrain_path means continuing training from a pre-trained checkpoint.
    args.resume is also the checkpoint path but only used for evaluation (TODO: change this, it is confusing).
    args.pretrained is for non-wignet models only and is a boolean (TODO: change this, it is confusing).
    """
    # if args.pretrain_path is not None and args.resume:
    # raise ValueError("Cannot specify both --pretrain_path and --resume")
    # if args.resume and not args.evaluate:
    # raise ValueError("--resume is only used for evaluation")
    if args.evaluate and not args.resume:
        raise ValueError("--evaluate needs args.resume to be provided")

    # if args.pretrain_path is not None:
    #     model = load_pretrained_weights(model, args)

    ################### flops #################
    # print(model)
    # if hasattr(model, "default_cfg"):
    #     default_cfg = model.default_cfg
    #     input_size = [1] + list(default_cfg["input_size"])
    # else:
    #     input_size = [1, 3, 224, 224]
    # print(f'\nUsing input size: {input_size}\n')
    # _logger.info(f"\nUsing input size: {input_size}\n")
    # input = torch.randn(input_size)  # .cuda()

    # from torchprofile import profile_macs
    # model.eval()
    # macs = profile_macs(model, input)
    # model.train()
    # print("model flops:", macs, "input_size:", input_size)
    ##########################################

    if args.local_rank == 0:
        _logger.info(
            "Model %s created, param count: %d"
            % (args.model, sum([m.numel() for m in model.parameters()]))
        )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
    )

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    """ if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2)) """

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    if args.num_gpu > 1:
        if use_amp == "apex":
            _logger.warning(
                "Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP."
            )
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        # try:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0,
            args=args,
        )
        #     print("try OK")
        # except Exception as e:
        #     print("try failed")
        #     if not args.strict_model_loading:
        #         args.pretrain_path = args.resume
        #         model = load_pretrained_weights(model, args)
        #     else:
        #         raise e

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        _logger.info(f"\nargs.model_ema: {args.model_ema} --> Creating EMA model")
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume=args.resume,
        )
        _logger.info("EMA Model created.\n")

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex and use_amp != "native":
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                        "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
                    )
            except Exception as e:
                _logger.error(
                    "Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1"
                )
        if has_apex and use_amp != "native":
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model,
                device_ids=[args.local_rank],
                find_unused_parameters=True,
            )  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    ################# Scheduler - original
    if resume_epoch is None:
        lr_scheduler, num_epochs = create_scheduler(args, optimizer)
        start_epoch = 0
        if args.start_epoch is not None:
            # a specified start_epoch will always override the resume epoch
            start_epoch = args.start_epoch
        elif resume_epoch is not None:
            start_epoch = resume_epoch
        if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.step(start_epoch)
        lr_sch_offset = 0
    else:
        ################# Scheduler - new for fine-tuning
        _logger.info("\nUsing fine-tuning scheduler...")
        new_ft_args = SimpleNamespace(**vars(args))
        new_ft_args.epochs = args.epochs - 300  # 30
        lr_scheduler, num_epochs = create_scheduler(new_ft_args, optimizer)
        start_epoch = 0
        if args.start_epoch is not None:
            # a specified start_epoch will always override the resume epoch
            start_epoch = args.start_epoch
        elif resume_epoch is not None:
            start_epoch = resume_epoch  # 304
        if lr_scheduler is not None and start_epoch > 0:
            # lr_scheduler.step(start_epoch)
            lr_scheduler.step(0)
            lr_sch_offset = start_epoch
            num_epochs = num_epochs + start_epoch

    if args.local_rank == 0:
        _logger.info("Scheduled epochs: {}".format(num_epochs))

    train_dir = os.path.join(args.data, "train")
    if not os.path.exists(train_dir):
        _logger.error("Training folder does not exist at: {}".format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        if args.prefetcher:
            assert (
                not num_aug_splits
            )  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # print(f"\n***** Input size dataloader: {data_config['input_size']} *****\n")
    _logger.info(f"\n***** Input size dataloader: {data_config['input_size']} *****\n")
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]

    loader_train = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        repeated_aug=args.repeated_aug,
    )

    inet200 = "200" in args.data
    if inet200:
        _logger.info("\nUsing Tiny-ImageNet-200 dataset")
        eval_dir = os.path.join(args.data, "test")
        assert args.num_classes == 200, "For Tiny-ImageNet-200, num_classes must be 200"
    else:
        assert args.num_classes == 1000, "For ImageNet-1K, num_classes must be 1000"
        eval_dir = os.path.join(args.data, "val")
        if not os.path.isdir(eval_dir):
            eval_dir = os.path.join(args.data, "validation")
            if not os.path.exists(eval_dir):
                if args.local_rank == 0:
                    raise ValueError("Validation folder does not exist.")

    dataset_eval = Dataset(eval_dir)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config["crop_pct"],
        pin_memory=args.pin_mem,
    )

    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(
            num_splits=num_aug_splits, smoothing=args.smoothing
        ).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.evaluate:
        _logger.info("\n***** Evaluating model *****")
        if model_ema is not None:
            _logger.info("---EMA model is being used...")
            eval_metrics_test = validate(
                model_ema.ema,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
                log_suffix=" (EMA) ",
            )
        else:
            _logger.info("---The main model (not EMA) is being used...")
            eval_metrics_test = validate(
                model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast
            )
        print(eval_metrics_test)
        return

    eval_metric = args.eval_metric
    best_metric_no_ema = None
    best_epoch_no_ema = None
    best_metric_ema = None
    best_epoch_ema = None
    saver_no_ema = None
    output_dir_no_ema = ""
    saver_ema = None
    output_dir_ema = ""
    if args.local_rank == 0:
        output_base = args.output if args.output else "./output"
        exp_name = "-".join(
            [
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                args.model,
                str(data_config["input_size"][-1]),
            ]
        )
        output_dir_no_ema = get_outdir(output_base, "train", f"{exp_name}_NO_EMA")
        decreasing = True if eval_metric == "loss" else False
        saver_no_ema = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=None,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir_no_ema,
            recovery_dir=output_dir_no_ema,
            decreasing=decreasing,
        )
        with open(os.path.join(output_dir_no_ema, "args.yaml"), "w") as f:
            f.write(args_text)

        if args.model_ema:
            output_dir_ema = get_outdir(output_base, "train", f"{exp_name}_EMA")
            saver_ema = CheckpointSaver(
                model=model,
                optimizer=optimizer,
                args=args,
                model_ema=model_ema,
                amp_scaler=loss_scaler,
                checkpoint_dir=output_dir_ema,
                recovery_dir=output_dir_ema,
                decreasing=decreasing,
            )
            with open(os.path.join(output_dir_ema, "args.yaml"), "w") as f:
                f.write(args_text)

    print("\n\n\n", start_epoch, num_epochs, "\n\n\n")

    for epoch in range(start_epoch, num_epochs):
        if args.distributed:
            loader_train.sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            epoch,
            model,
            loader_train,
            optimizer,
            train_loss_fn,
            args,
            lr_scheduler=lr_scheduler,
            saver=saver_ema if args.model_ema else saver_no_ema,
            output_dir=output_dir_ema if args.model_ema else output_dir_no_ema,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
        )

        if args.local_rank == 0 and not args.no_wandb:
            wandb.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        if args.distributed and args.dist_bn in ("broadcast", "reduce"):
            if args.local_rank == 0:
                _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == "reduce")

        eval_metrics_no_ema = validate(
            model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast
        )

        if args.local_rank == 0 and not args.no_wandb:
            wandb.log(
                {
                    "val/loss": eval_metrics_no_ema["loss"],
                    "val/acc@1": eval_metrics_no_ema["top1"],
                    "val/acc@5": eval_metrics_no_ema["top5"],
                },
                step=epoch,
            )

        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
            eval_metrics_ema = validate(
                model_ema.ema,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
                log_suffix=" (EMA)",
            )
            # eval_metrics = ema_eval_metrics

            if args.local_rank == 0 and not args.no_wandb:
                wandb.log(
                    {
                        "val_ema/loss": eval_metrics_ema["loss"],
                        "val_ema/acc@1": eval_metrics_ema["top1"],
                        "val_ema/acc@5": eval_metrics_ema["top5"],
                    },
                    step=epoch,
                )

        if lr_scheduler is not None:
            # step LR for next epoch
            if args.model_ema:
                lr_scheduler.step(
                    epoch + 1 - lr_sch_offset, eval_metrics_ema[eval_metric]
                )
            else:
                lr_scheduler.step(
                    epoch + 1 - lr_sch_offset, eval_metrics_no_ema[eval_metric]
                )

        update_summary(
            epoch,
            train_metrics,
            eval_metrics_no_ema,
            os.path.join(output_dir_no_ema, "summary.csv"),
            write_header=best_metric_no_ema is None,
        )

        if args.model_ema:
            update_summary(
                epoch,
                train_metrics,
                eval_metrics_ema,
                os.path.join(output_dir_ema, "summary.csv"),
                write_header=best_metric_ema is None,
            )

        if saver_no_ema is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics_no_ema[eval_metric]
            best_metric_no_ema, best_epoch_no_ema = saver_no_ema.save_checkpoint(
                epoch, metric=save_metric
            )

        if saver_ema is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics_ema[eval_metric]
            best_metric_ema, best_epoch_ema = saver_ema.save_checkpoint(
                epoch, metric=save_metric
            )

    if best_metric_no_ema is not None:
        _logger.info(
            "*** Best metric (NO EMA): {0} (epoch {1})".format(
                best_metric_no_ema, best_epoch_no_ema
            )
        )

        if args.local_rank == 0 and not args.no_wandb:
            wandb.log(
                {
                    f"best_no_ema/{eval_metric}": best_metric_no_ema,
                    "best_no_ema/epoch": best_epoch_no_ema,
                },
                step=epoch,
            )

    if best_metric_ema is not None:
        _logger.info(
            "*** Best metric (EMA): {0} (epoch {1})".format(
                best_metric_ema, best_epoch_ema
            )
        )

        if args.local_rank == 0 and not args.no_wandb:
            wandb.log(
                {
                    f"best_ema/{eval_metric}": best_metric_ema,
                    "best_ema/epoch": best_epoch_ema,
                },
                step=epoch,
            )

    if not args.no_wandb:
        wandb.finish()


def train_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    lr_scheduler=None,
    saver=None,
    output_dir="",
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, target) in enumerate(loader):

        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                parameters=model.parameters(),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100.0 * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                    )
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if (
            saver is not None
            and args.recovery_interval
            and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
        ):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=""):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):

            # if batch_idx > 20: # TODO remove it
            #     break
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)

            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss = loss_fn(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (
                last_batch or batch_idx % args.log_interval == 0
            ):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m,
                    )
                )

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )
    return metrics


if __name__ == "__main__":
    main()
