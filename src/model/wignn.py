
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from gcn_lib import act_layer, WindowGrapher
from timm.models import create_model
import time

from torchprofile import profile_macs

import sys

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'wignn_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'wignn_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        self.k = opt.k                       # knn
        self.act = opt.act                   # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = opt.norm                 # batch or instance normalization {batch, instance}
        self.bias = opt.bias                 # bias of conv layer True or False
        self.epsilon = opt.epsilon           # stochastic epsilon for gcn
        self.stochastic = opt.use_stochastic # stochastic for gcn, True or False
        self.conv = opt.conv                 # graph conv layer {edge, mr}
        self.emb_dims = opt.emb_dims         # # Dimension of embeddings
        self.drop_path = opt.drop_path
        
        self.blocks = opt.blocks             # [2,2,6,2] # number of basic blocks in the backbone
        self.channels = opt.channels         # [80, 160, 400, 640] # number of channels of deep features

        self.img_size = opt.img_size
        self.use_shifts = opt.use_shifts


        self.n_blocks = sum(self.blocks)
        self.window_size = [opt.windows_size for _ in range(len(self.blocks))]
        
        if opt.use_reduce_ratios:
            self.reduce_ratios = [2, 2, 1, 1]
        else:
            self.reduce_ratios = [1, 1, 1, 1]
        
        adapt_knn = opt.adapt_knn
        print(f'Created Model wignn ({self.img_size}) Window: {self.window_size}')
        print(f'Use shifting windows: {self.use_shifts} adapt knn: {adapt_knn}')

        print(f'Knn: {self.k}')
        print(f'Reduce ratios: {self.reduce_ratios}')


        print(f'Channel: {self.channels}')
        print(f'Blocks: {self.blocks}')


        
        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_blocks)]  # stochastic depth decay rule 
        self.num_knn = [int(x.item()) for x in torch.linspace(self.k, self.k, self.n_blocks)]  # number of knn's k
        # max_dilation = 49 // max(num_knn)
        
        self.stem = Stem(out_dim=self.channels[0], act=self.act)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels[0], self.img_size//4, self.img_size//4))

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(Downsample(self.channels[i-1], self.channels[i]))

            for j in range(self.blocks[i]):
                shift_size = 0
                if j % 2 != 0 and self.use_shifts:
                    shift_size = self.window_size[i] // 2

                self.backbone += [
                    Seq(
                        WindowGrapher(
                            in_channels = self.channels[i],
                            kernel_size = self.num_knn[idx],
                            windows_size = self.window_size[i],
                            dilation = 1,
                            conv = self.conv,
                            act = self.act,
                            norm = self.norm,
                            bias = self.bias,
                            stochastic = self.stochastic,
                            epsilon = self.epsilon,
                            drop_path = self.dpr[idx],
                            relative_pos = True,
                            shift_size = shift_size,
                            r = self.reduce_ratios[i],
                            input_resolution = (
                                (self.img_size//4) // (2 ** i),
                                (self.img_size//4) // (2 ** i)),
                            adapt_knn=adapt_knn
                        ),
                        FFN(self.channels[i], self.channels[i] * 4, act=self.act, drop_path=self.dpr[idx])
                    )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.Conv2d(self.channels[-1], self.emb_dims, 1, bias=True),
                              nn.BatchNorm2d(self.emb_dims),
                              act_layer(self.act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(self.emb_dims, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)
    











class OptInit:
    def __init__(self, 
                 num_classes=1000,
                 drop_path_rate=0.0, 
                 knn = 9, 
                 use_shifts = True, 
                 use_reduce_ratios = False, 
                 img_size = 224,
                 adapt_knn = False,

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
        self.channels = channels # number of channels of deep features
        self.n_classes = num_classes # Dimension of out_channels
        self.emb_dims = 1024 # Dimension of embeddings
        self.windows_size = 7

        self.use_shifts = use_shifts
        self.img_size = img_size
        self.use_reduce_ratios = False
        self.adapt_knn = adapt_knn

@register_model
def wignn_ti_224_gelu(pretrained=False, **kwargs):
    

    opt = OptInit(**kwargs, channels = [48, 96, 240, 384], blocks= [2,2,6,2])

    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['wignn_224_gelu']
    return model


@register_model
def wignn_s_224_gelu(pretrained=False, **kwargs):

    opt = OptInit(**kwargs, channels = [80, 160, 400, 640], blocks= [2,2,6,2])

    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['wignn_224_gelu']
    return model


@register_model
def wignn_m_224_gelu(pretrained=False, **kwargs):

    opt = OptInit(**kwargs, channels = [96, 192, 384, 768], blocks= [2,2,16,2])

    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['wignn_224_gelu']
    return model


@register_model
def wignn_b_224_gelu(pretrained=False, **kwargs):

    opt = OptInit(**kwargs, channels = [128, 256, 512, 1024], blocks= [2,2,18,2])

    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['wignn_b_224_gelu']
    return model



if __name__ == '__main__':


    # for img_size in [224,224*2,224*3,224*4,224*5]:

    #     model = create_model(
    #             'wignn_ti_224_gelu',
    #             knn = 9,
    #             use_shifts = True,
    #             img_size = img_size,
    #             adapt_knn = True
    #         )

    #     model = model.cuda()
    #     model.eval()

    #     x = torch.rand((1,3,img_size,img_size)).to(device = 'cuda')

    #     # print(model(x).shape)

    #     macs = profile_macs(model, x) 
    #     print(f'\n\n!!!!! WiGNet macs ({img_size}): {macs}\n\n')

    for img_size in [224,224*2,224*3,224*4,224*5]:
        

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],profile_memory=True, record_shapes=True) as prof:    

            x = torch.rand((1,3,img_size,img_size)).to(device = 'cuda')
            
            model = create_model(
                'wignn_ti_224_gelu',
                knn = 9,
                use_shifts = True,
                img_size = img_size,
                adapt_knn = True
            )

            model = model.cuda()
            model.eval()

            _ = model(x)

        f = open("memory_WiGNet_model.txt", "a")

        f.write(prof.key_averages().table())
        f.write('\n\n')
        f.close()

    print('\n\n---------------\nResults saved in memory_WiGNet_model.txt')
            