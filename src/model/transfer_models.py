import torch
from timm.models import create_model, resume_checkpoint
from timm.models.helpers import clean_state_dict
import torch.nn as nn
from torch.nn import Sequential as Seq
from gcn_lib import act_layer
import numpy as np
from model import pyramid_vig
from model import wignn
from model import wignn_256
from model import greedyvig
from model import mobilevig
import torchvision

from torchprofile import profile_macs

from collections import OrderedDict
import sys


def remove_pos(state_dict):
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'pos_embed' not in k and 'attn_mask' not in k and 'adj_mask' not in k:
            cleaned_state_dict[k] = v
    return cleaned_state_dict

def remove_relative_pos(state_dict):
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'relative_pos' not in k and 'pos_embed' not in k:
            cleaned_state_dict[k] = v
    return cleaned_state_dict

def get_model(model_type, use_shift = False, adapt_knn = False, checkpoint = None, pretrained = True, freezed = True, dataset = 'PET', crop_size = None):

    if dataset == 'CelebA':
        n_classes = 307
    else:
        raise NotImplementedError(f'Dataset: {dataset} not yet implemented')
    
    
    pretrained_creation = False
    
    if('wignn' in model_type):
        if crop_size is not None:
            model = create_model(
                model_type,
                pretrained=pretrained_creation,
                use_shifts = use_shift,
                adapt_knn = adapt_knn,
                img_size = crop_size 
            )
        else:
            model = create_model(
                model_type,
                pretrained=pretrained_creation,
                use_shifts = use_shift,
                adapt_knn = adapt_knn 
            )
    elif('pvig' in model_type):
        if crop_size is not None:
            model = create_model(
                model_type,
                pretrained=pretrained_creation,
                img_size = crop_size 
            )
        else:
            model = create_model(
                model_type,
                pretrained=pretrained_creation 
            )
    elif('GreedyViG' in model_type):
        model = create_model(
                model_type,
                num_classes=1000,
                distillation=False,
                pretrained=pretrained_creation
            )
    elif('mobilevig' in model_type):
        model = create_model(
                model_type,
            )

    else:
        raise NotImplementedError(f'Model: {model_type} not yet implemented')
        


        


    # load checkpoint for our models and ViG
    if pretrained and checkpoint != '':
        if 'wignn' in model_type:
            assert checkpoint is not None, f'Cannot start from pretrained {model_type} model without checkpoints'

            checkpoint = torch.load(checkpoint, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                print('Restoring model state from checkpoint...')
                state_dict = clean_state_dict(checkpoint['state_dict'])

                if crop_size is not None:
                    state_dict = remove_pos(state_dict)

                model.load_state_dict(state_dict, strict = False)

            print(f'Pretrain weights for  {model_type} loaded.')

        elif 'pvig' in model_type:
            assert checkpoint is not None, f'Cannot start from pretrained {model_type} model without checkpoints'

            state_dict = torch.load(checkpoint)
            if crop_size is not None:
                state_dict = remove_relative_pos(state_dict)
            model.load_state_dict(state_dict, strict=False)
            print('Pretrain weights for vig loaded.')

        elif 'GreedyViG' in model_type:
            assert checkpoint is not None, f'Cannot start from pretrained {model_type} model without checkpoints'

            checkpoint = torch.load(checkpoint, map_location='cpu')
            checkpoint_model = checkpoint['state_dict']

            state_dict = model.state_dict()
            for k in ['dist_head.weight', 'dist_head.bias']:
                # if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

            model.load_state_dict(checkpoint_model, strict=True) 
            print('Pretrain weights for GreedyViG loaded.')

        elif 'mobilevig' in model_type:
            assert checkpoint is not None, f'Cannot start from pretrained {model_type} model without checkpoints'

            checkpoint = torch.load(checkpoint, map_location='cpu')
            checkpoint_model = checkpoint['state_dict']

            state_dict = model.state_dict()
            for k in ['dist_head.weight', 'dist_head.bias']:
                # if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

            model.load_state_dict(checkpoint_model, strict=True) 
            print('Pretrain weights for MobileViG loaded.')




    
    # freeze the model
    for param in model.parameters():
        if freezed:
            param.requires_grad = False
        else:
            param.requires_grad = True


    if 'pvig' in model_type or 'wignn' in model_type:
        model.prediction = Seq(nn.Conv2d(model.prediction[0].in_channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer('gelu'),
                              nn.Dropout(0.0),
                              nn.Conv2d(1024, n_classes, 1, bias=True))
        model.prediction.requires_grad = True
    elif 'GreedyViG' in model_type:
        model.prediction = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(model.prediction[1].in_channels, 768, kernel_size=1, bias=True),
                                        nn.BatchNorm2d(768),
                                        nn.GELU(),
                                        nn.Dropout(0.0))
        
        model.head = nn.Conv2d(768, n_classes, kernel_size=1, bias=True)
        model.dist_head = nn.Conv2d(768, n_classes, 1, bias=True)
        model.prediction.requires_grad = True
        model.head.requires_grad = True
        model.dist_head.requires_grad = True

    elif 'mobilevig' in model_type:
        model.prediction = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(256, 512, 1, bias=True),
                                        nn.BatchNorm2d(512),
                                        nn.GELU(),
                                        nn.Dropout(0.))
        
        model.head = nn.Conv2d(512, n_classes, 1, bias=True)
        model.dist_head = nn.Conv2d(512, n_classes, 1, bias=True)
        model.prediction.requires_grad = True
        model.head.requires_grad = True
        model.dist_head.requires_grad = True

    else:
        raise NotImplementedError(f'Model {model_type} not yet implemented\n{model}')


    params = sum(p.numel() for p in model.parameters())
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_parameters = sum([np.prod(p.size()) for p in trainable_parameters])

    

    return model, params, trainable_parameters, n_classes





if __name__ == '__main__':
    checkpoint = 'path/to/checkpoint'
    model_type = 'wignn_ti_256_gelu'
    
    
    dataset = 'CelebA'
    model, params, trainable_parameters, n_classes = get_model(model_type = model_type, 
                                                               use_shift=True, 
                                                               adapt_knn=True, 
                                                               checkpoint = checkpoint, 
                                                               freezed=True, 
                                                               dataset = dataset,
                                                               crop_size = 512)
    model.eval()
    model.cuda()
    x = torch.rand((1,3,512,512)).cuda()
    # print(model)
    print(f"Parameters: {params}")
    print(f"Trainable Parameters: {trainable_parameters}")

    # out = model(x)
    # print(out.shape)
    macs = profile_macs(model, x) 
    print(f'\n\n!!!!! macs : {macs*10**-9}\n\n')