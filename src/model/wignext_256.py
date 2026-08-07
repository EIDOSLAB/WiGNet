from timm.models.registry import register_model

try:
    from model.wignext import DeepGCN
except ImportError:
    from wignext import DeepGCN

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from torchprofile import profile_macs

import torch


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "wignn_256_gelu": _cfg(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "wignn_b_256_gelu": _cfg(
        crop_pct=0.95,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}

default_cfgs_224 = {
    "wignn_224_gelu": _cfg(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "wignn_b_224_gelu": _cfg(
        crop_pct=0.95,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


class OptInit:
    """
    All arguments will be overridden by kwargs!
    """

    def __init__(
        self,
        WG_type="WG",
        num_classes=1000,
        drop_path_rate=0.0,
        knn=9,
        conv="mr",
        use_shifts=True,
        use_reduce_ratios=False,
        use_dilation=False,
        img_size=256,
        adapt_knn=True,
        channels=None,
        blocks=None,
        window_size=8,
        use_distance_masking=False,
        glocal=False,
        masking_pad_nodes=False,
        cnn_assist=False,
        use_cpe=False,
        ffn_expand_ratios=[4, 4, 4, 4],
        fc1=True,
        fc1_act=False,
        fc2_g=1,
        fc2_act=False,
        basic_conv_groups=4,
        extra_fc=False,
        **kwargs,
    ):
        self.WG_type = WG_type  # type of window grapher {WG, FCHalf_WG}
        self.k = knn  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr}
        self.act = "gelu"  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = "batch"  # batch or instance normalization {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = use_dilation  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.blocks = blocks  # number of basic blocks in the backbone
        self.channels = channels  # number of channels of deep features
        self.n_classes = num_classes  # Dimension of out_channels
        self.emb_dims = 1024  # Dimension of embeddings
        self.window_size = window_size
        self.use_shifts = use_shifts
        self.img_size = img_size
        self.use_reduce_ratios = use_reduce_ratios
        self.adapt_knn = adapt_knn
        self.use_distance_masking = use_distance_masking
        self.glocal = glocal
        self.masking_pad_nodes = masking_pad_nodes
        self.cnn_assist = cnn_assist
        self.use_cpe = use_cpe
        self.ffn_expand_ratios = ffn_expand_ratios
        self.fc1 = fc1
        self.fc1_act = fc1_act
        self.fc2_g = fc2_g
        self.fc2_act = fc2_act
        self.basic_conv_groups = basic_conv_groups
        self.extra_fc = extra_fc

        self._override(kwargs)

    def _override(self, kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                print(
                    f"\n  - Overriding {key} from the kwargs in OptInit with value {value}"
                )
                setattr(self, key, value)


@register_model
def wignext_ti_256_gelu(pretrained=False, **kwargs):
    opt = OptInit(
        **kwargs,
        WG_type="Conv_WG",
        conv="mr",
        channels=[36, 72, 160, 288],
        blocks=[2, 2, 6, 2],
        glocal=True,
    )
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["wignn_256_gelu"]
    return model


@register_model
def wignext_s_256_gelu(pretrained=False, **kwargs):
    opt = OptInit(
        **kwargs,
        WG_type="Conv_WG",
        conv="mr",
        channels=[48, 96, 240, 384],
        blocks=[2, 2, 6, 2],
        glocal=True,
    )
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["wignn_256_gelu"]
    return model


@register_model
def wignext_m_256_gelu(pretrained=False, **kwargs):
    opt = OptInit(
        **kwargs,
        WG_type="Conv_WG",
        conv="mr",
        channels=[60, 120, 260, 480],
        blocks=[2, 2, 16, 2],
        glocal=True,
    )
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["wignn_256_gelu"]
    return model
