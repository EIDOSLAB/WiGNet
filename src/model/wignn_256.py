from timm.models.registry import register_model
from model.wignn import DeepGCN
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from torchprofile import profile_macs

import torch

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'wignn_256_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'wignn_b_256_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class OptInit:
    def __init__(
            self, 
            num_classes=1000, 
            drop_path_rate=0.0, 
            knn = 9, 
            use_shifts = True, 
            use_reduce_ratios = False, 
            img_size = 256,
            adapt_knn = True,

            channels = None,
            blocks = None,
            **kwargs):
            
        self.k = knn # neighbor num (default:9)
        self.conv = 'mr' # graph conv layer {edge, mr}
        self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = 'batch' # batch or instance normalization {batch, instance}
        self.bias = True # bias of conv layer True or False
        self.dropout = 0.0 # dropout rate
        self.use_dilation = True # use dilated knn or not
        self.epsilon = 0.2 # stochastic epsilon for gcn
        self.use_stochastic = False # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.blocks = blocks # number of basic blocks in the backbone
        self.channels = channels  # number of channels of deep features
        self.n_classes = num_classes # Dimension of out_channels
        self.emb_dims = 1024 # Dimension of embeddings
        self.windows_size = 8 
        self.use_shifts = use_shifts
        self.img_size = img_size
        self.use_reduce_ratios = use_reduce_ratios
        self.adapt_knn = adapt_knn

@register_model
def wignn_ti_256_gelu(pretrained=False, **kwargs):

    opt = OptInit(**kwargs, channels = [48, 96, 240, 384], blocks= [2,2,6,2])
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['wignn_256_gelu']
    return model


@register_model
def wignn_s_256_gelu(pretrained=False, **kwargs):

    opt = OptInit(**kwargs, channels = [80, 160, 400, 640], blocks= [2,2,6,2])
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['wignn_256_gelu']
    return model


@register_model
def wignn_m_256_gelu(pretrained=False, **kwargs):

    opt = OptInit(**kwargs, channels = [96, 192, 384, 768], blocks= [2,2,16,2])

    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['wignn_256_gelu']
    return model


@register_model
def wignn_b_256_gelu(pretrained=False, **kwargs):

    opt = OptInit(**kwargs, channels = [128, 256, 512, 1024], blocks= [2,2,18,2])

    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['wignn_b_256_gelu']
    return model



if __name__ == '__main__':

    pass
    

