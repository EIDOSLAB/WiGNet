import torch
from timm.models import create_model, resume_checkpoint
from timm.models.helpers import clean_state_dict
import torch.nn as nn
from torch.nn import Sequential as Seq
from gcn_lib import act_layer
import numpy as np

from torchprofile import profile_macs

from collections import OrderedDict
import sys

from typing import Union, List

# def remove_pos(state_dict):
#     cleaned_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         if "pos_embed" not in k and "attn_mask" not in k and "adj_mask" not in k:
#             cleaned_state_dict[k] = v
#     return cleaned_state_dict


def remove_relative_pos(state_dict):
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "relative_pos" not in k and "pos_embed" not in k:
            cleaned_state_dict[k] = v
    return cleaned_state_dict


def create_wignn_model(
    crop_size,
    model_type,
    pretrained_creation,
    use_shifts,
    adapt_knn,
    window_size,
    knn,
):
    if crop_size is not None:

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        return create_model(
            model_type,
            pretrained=pretrained_creation,
            use_shifts=use_shifts,
            adapt_knn=adapt_knn,
            img_size=crop_size,
            window_size=window_size,
            knn=knn,
        )
    return create_model(
        model_type,
        pretrained=pretrained_creation,
        use_shifts=use_shifts,
        adapt_knn=adapt_knn,
        window_size=window_size,
        knn=knn,
    )


def create_pvig_model(crop_size, model_type, pretrained_creation):
    if crop_size is not None:
        return create_model(
            model_type, pretrained=pretrained_creation, img_size=crop_size
        )
    return create_model(model_type, pretrained=pretrained_creation)


def create_greedyvig_model(model_type, pretrained_creation):
    return create_model(
        model_type,
        num_classes=1000,
        distillation=False,
        pretrained=pretrained_creation,
    )


def create_mobilevig_model(model_type):
    return create_model(model_type)


def create_swin_model(args, model_type, pretrained_creation=True):
    # if "tiny" in model_type:
    #     model_name = "swinv2_tiny_window8_256"
    # if "small" in model_type:
    #     model_name = "swinv2_small_window8_256"
    # if 'large' in model_type:
    #     model_name = 'swinv2_large_window12_192'
    # else:
    #     raise NotImplementedError(f"Model {model_type} is not implemented yet.\n")

    model_name = model_type

    model = create_model(
        model_name,
        pretrained=pretrained_creation,
        num_classes=1000,  # 1000 is just for now
    )

    assert isinstance(args.crop_size, int)
    assert isinstance(args.window_size, int)
    cs, ws = args.crop_size, args.window_size
    try:
        model.set_input_size(img_size=(cs, cs), window_size=(ws, ws))
    except AttributeError as e:
        print(e)
        print(
            "To solve this, you can upgrade timm to newer versions. This might effect other models."
        )
    return model


def create_model_general(model_type, pretrained_creation=True):
    return create_model(model_type, pretrained=pretrained_creation)


def load_checkpoint_partially(model_type, model, state_dict, args):

    assert (
        "wignn" in model_type or "pvig" in model_type
    ), f"{model_type} model should not use this function."

    assert args.pos_embed_mode in [
        "interpolate-bili",
        "interpolate-bicu",
        "trainable",
    ], f"pos_embed_mode should be either interpolate-bili, interpolate-bicu, or trainable, but got {args.pos_embed_mode}."

    print("##### Loading state dict failed, filtering keys...\n")
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in state_dict.items():
        if k not in model_dict:
            print(f"Module: {k} (from checkpoint) -- Skipped: not in model_dict")
        elif v.size() != model_dict[k].size():
            print(
                f"Module: {k} (from checkpoint) -- Skipped: size mismatch: {list(v.size())} (checkpoint) != {list(model_dict[k].size())} (model)"
            )
        else:
            filtered_dict[k] = v

        if k == "pos_embed":
            if args.freezed:
                if "interpolate" in args.pos_embed_mode:
                    model.pos_embed = nn.Parameter(v)  # To discard the new pos_embed
                    model.pos_embed_interpolation_mode = (
                        "bilinear" if "bili" in args.pos_embed_mode else "bicubic"
                    )
                    filtered_dict[k] = (
                        v  # So later the model will be updated with this (checkpoint) pos_embed
                    )
                    print(
                        "Model will use the interpolation of the freezed backbone pos_embed."
                    )
                else:  # pos_embed_mode == "trainable"
                    pass
                    """This scenario will be handled later: when we're freezing the backbone, 
                        we will let the new pos_embed to be trained on the new dataset."""
            else:
                print(
                    f"backbone is not freezed, so pos_embed will be trained from scratch.\n"
                )  # model will have its new pos_embed (shape matching the new dataset) trained on the new dataset from scratch.

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    return model


def load_wignn_checkpoint(
    model_type,
    model,
    checkpoint,
    args=None,
):
    assert (
        checkpoint is not None
    ), f"Cannot start from pretrained wignn model without checkpoint."

    checkpoint = torch.load(checkpoint, map_location="cpu")
    assert (
        isinstance(checkpoint, dict) and "state_dict" in checkpoint
    ), "Checkpoint is not a dict or does not contain 'state_dict' key"

    # state_dict = clean_state_dict(checkpoint["state_dict"])
    state_dict = checkpoint["state_dict"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model = load_checkpoint_partially(model_type, model, state_dict, args)
    print(f"##### Pretrained weights loaded for {model_type} model.\n")
    return model


def load_pvig_checkpoint(model_type, model, checkpoint, args=None):
    assert (
        checkpoint is not None
    ), f"Cannot start from pretrained {model_type} model without checkpoints"

    state_dict = torch.load(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model = load_checkpoint_partially(model_type, model, state_dict, args)
    print(f"##### Pretrained weights loaded for {model_type} model.\n")
    return model


def load_greedyvig_or_mobilevig_checkpoint(model_type, model, checkpoint):
    assert (
        checkpoint is not None
    ), f"Cannot start from pretrained {model_type} model without checkpoints"

    checkpoint_model = torch.load(checkpoint, map_location="cpu")["state_dict"]
    for k in ["dist_head.weight", "dist_head.bias"]:
        del checkpoint_model[k]

    model.load_state_dict(checkpoint_model, strict=True)
    print(f"##### Pretrained weights for {model_type} loaded.\n")
    return model


def enable_requires_grad(modules: Union[nn.Module, List[nn.Module]]) -> None:
    if isinstance(modules, nn.Module):
        modules = [modules]
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


def get_model(
    model_type,
    use_shift=False,
    adapt_knn=False,
    checkpoint=None,
    pretrained=True,
    freezed=True,
    dataset="CelebA",
    crop_size=None,
    window_size=8,
    knn=9,
    args=None,
):

    if dataset == "CelebA":
        n_classes = 307
    else:
        raise NotImplementedError(f"Dataset: {dataset} not yet implemented")

    pretrained_creation = False
    if checkpoint == "":
        checkpoint = None

    supported_models_key_words = args.other_allowed_models_keywords

    if "wignn" in model_type:
        model = create_wignn_model(
            crop_size,
            model_type,
            pretrained_creation,
            use_shifts=use_shift,
            adapt_knn=adapt_knn,
            window_size=window_size,
            knn=knn,
        )
    elif "pvig" in model_type:
        model = create_pvig_model(crop_size, model_type, pretrained_creation)
    elif "GreedyViG" in model_type:
        model = create_greedyvig_model(model_type, pretrained_creation)
    elif "mobilevig" in model_type:
        model = create_mobilevig_model(model_type)
    elif "swin" in model_type:
        model = create_swin_model(args, model_type, pretrained_creation=True)
    else:
        if any([kw in model_type for kw in supported_models_key_words]):
            model = create_model_general(model_type, pretrained_creation=True)
        else:
            raise NotImplementedError(f"Model: {model_type} not implemented.")

    if not pretrained and checkpoint is not None:
        raise RuntimeError(f"checkpoint should not be provided if pretrained is False.")

    if pretrained and checkpoint is not None:
        if not args.profile_model:
            print(
                f"\n##### Loading pretrained weights from checkpoint {checkpoint} for {model_type} model.\n"
            )
            if "wignn" in model_type:
                model = load_wignn_checkpoint(model_type, model, checkpoint, args)
            elif "pvig" in model_type:
                model = load_pvig_checkpoint(model_type, model, checkpoint, args)
            elif "GreedyViG" in model_type or "mobilevig" in model_type:
                model = load_greedyvig_or_mobilevig_checkpoint(
                    model_type, model, checkpoint
                )
            else:
                raise NotImplementedError(
                    f"Loading in-house pre-trained weights hasn't been implemented for {model_type} model yet."
                )
        else:
            print("args.profile_model is True, so checkpoint will not be loaded.")

    if not pretrained:
        if "resnet" in model_type or "swin" in model_type or "mambaout" in model_type:
            raise NotImplementedError(
                f"Model: {model_type} is loaded with pre-trained weights. Loading from scratch is not implemented yet."
            )
        else:
            print(f"\n##### No pretrained weights loaded for {model_type} model.\n")

    # freeze the model
    for param in model.parameters():
        assert hasattr(param, "requires_grad")
        if freezed:
            param.requires_grad = False
        else:
            param.requires_grad = True

    if "pvig" in model_type or "wignn" in model_type:
        if freezed and args.pos_embed_mode == "trainable":
            for name, param in model.named_parameters():
                if name == "pos_embed":
                    param.requires_grad = True
                    print(
                        f"\n##### Freezed:{freezed}, args.pos_embed_mode:{args.pos_embed_mode}\
                        --> requires grad is set to True for {name}.\n"
                    )

    class Squeeze(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return x.squeeze(self.dim)

    class Unsqueeze(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return x.unsqueeze(self.dim)

    if "pvig" in model_type or "wignn" in model_type:
        if "wignn_ln_" in model_type:
            model.prediction = Seq(
                nn.Linear(model.prediction[0].in_features, 1024, bias=True),
                nn.LayerNorm(1024),
                nn.Dropout(float(args.wignn_vig_pred_dropout)),
                nn.Linear(1024, n_classes, bias=True),
            )
        else:
            model.prediction = Seq(
                nn.Conv2d(model.prediction[0].in_channels, 1024, 1, bias=True),
                nn.BatchNorm2d(1024),
                act_layer("gelu"),
                nn.Dropout(float(args.wignn_vig_pred_dropout)),
                nn.Conv2d(1024, n_classes, 1, bias=True),
            )
        enable_requires_grad(model.prediction)
        print(f"\nUsing {args.wignn_vig_pred_dropout} dropout in prediction head.\n")

    elif "GreedyViG" in model_type:
        model.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(model.prediction[1].in_channels, 768, kernel_size=1, bias=True),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Dropout(0.0),
        )
        model.head = nn.Conv2d(768, n_classes, kernel_size=1, bias=True)
        model.dist_head = nn.Conv2d(768, n_classes, 1, bias=True)
        enable_requires_grad([model.head, model.dist_head, model.prediction])

    elif "mobilevig" in model_type:
        model.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 512, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Dropout(0.0),
        )
        model.head = nn.Conv2d(512, n_classes, 1, bias=True)
        model.dist_head = nn.Conv2d(512, n_classes, 1, bias=True)
        enable_requires_grad([model.head, model.dist_head, model.prediction])

    elif "resnet50" in model_type:
        model.fc = Seq(
            Unsqueeze(2),
            Unsqueeze(3),
            nn.Conv2d(2048, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer("gelu"),
            nn.Dropout(0.25),
            nn.Conv2d(1024, n_classes, 1, bias=True),
            Squeeze(-1),
            Squeeze(-1),
        )
        enable_requires_grad(model.fc)

    elif "resnet18" in model_type:
        model.fc = Seq(
            Unsqueeze(2),
            Unsqueeze(3),
            nn.Conv2d(512, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer("gelu"),
            nn.Dropout(0.25),
            nn.Conv2d(1024, n_classes, 1, bias=True),
            Squeeze(-1),
            Squeeze(-1),
        )
        enable_requires_grad(model.fc)

    elif "swinv2_tiny" in model_type or "swinv2_small" in model_type:
        model.head.fc = Seq(
            Unsqueeze(2),
            Unsqueeze(3),
            nn.Conv2d(768, 768, 1, bias=True),
            nn.BatchNorm2d(768),
            act_layer("gelu"),
            nn.Dropout(0.25),
            nn.Conv2d(768, n_classes, 1, bias=True),
            Squeeze(-1),
            Squeeze(-1),
        )
        enable_requires_grad(model.head)

    elif "mambaout" in model_type:
        model.head.fc = Seq(
            Unsqueeze(2),
            Unsqueeze(3),
            nn.Conv2d(2304, 256, 1, bias=True),
            nn.BatchNorm2d(256),
            act_layer("gelu"),
            nn.Dropout(0.25),
            nn.Conv2d(256, n_classes, 1, bias=True),
            Squeeze(-1),
            Squeeze(-1),
        )
        enable_requires_grad(model.head.fc)

    elif "mobilenetv3" in model_type:
        model.conv_head = nn.Conv2d(960, 512, kernel_size=1, stride=1)
        model.classifier = nn.Linear(512, 307)
        enable_requires_grad([model.classifier, model.conv_head])

    elif "efficientnet" in model_type or "ghostnet" in model_type:
        model.classifier = Seq(
            Unsqueeze(2),
            Unsqueeze(3),
            nn.Conv2d(1280, 512, 1, bias=True),
            nn.BatchNorm2d(512),
            act_layer("gelu"),
            nn.Dropout(0.25),
            nn.Conv2d(512, n_classes, 1, bias=True),
            Squeeze(-1),
            Squeeze(-1),
        )
        enable_requires_grad(model.classifier)

    elif "mobilevit" in model_type:
        model.head.fc = Seq(
            Unsqueeze(2),
            Unsqueeze(3),
            nn.Conv2d(768, 768, 1, bias=True),
            nn.BatchNorm2d(768),
            act_layer("gelu"),
            nn.Dropout(0.25),
            nn.Conv2d(768, n_classes, 1, bias=True),
            Squeeze(-1),
            Squeeze(-1),
        )
        enable_requires_grad(model.head.fc)

    else:
        raise NotImplementedError(f"Model {model_type} is not implemented yet.\n")

    params = sum(p.numel() for p in model.parameters())
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_parameters = sum([np.prod(p.size()) for p in trainable_parameters])

    return model, params, trainable_parameters, n_classes


if __name__ == "__main__":
    checkpoint = "path/to/checkpoint"
    model_type = "wignn_ti_256_gelu"

    dataset = "CelebA"
    model, params, trainable_parameters, n_classes = get_model(
        model_type=model_type,
        use_shift=True,
        adapt_knn=True,
        checkpoint=checkpoint,
        freezed=True,
        dataset=dataset,
        crop_size=512,
    )
    model.eval()
    model.cuda()
    x = torch.rand((1, 3, 512, 512)).cuda()
    # print(model)
    print(f"Parameters: {params}")
    print(f"Trainable Parameters: {trainable_parameters}")

    # out = model(x)
    # print(out.shape)
    macs = profile_macs(model, x)
    print(f"\n\n!!!!! macs : {macs*10**-9}\n\n")
