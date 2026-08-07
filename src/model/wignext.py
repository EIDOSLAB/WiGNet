import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

try:
    from gcn_lib import act_layer, WindowGrapher
except ImportError:
    import sys

    sys.path.append("..")
    from gcn_lib import act_layer, WindowGrapher

from timm.models import create_model
import time
from torchprofile import profile_macs
import sys


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
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


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act="relu",
        drop_path=0.0,
    ):
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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class Stem(nn.Module):
    """Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """Convolution-based downsample"""

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
    """
    Args in the opt:
        k: int, knn
        act: str, activation layer {relu, prelu, leakyrelu, gelu, hswish}
        norm: str, batch or instance normalization {batch, instance}
        bias: bool, bias of conv layer True or False
        epsilon: float, stochastic epsilon for gcn
        use_stochastic: bool, stochastic for gcn, True or False
        conv: str, graph conv layer {edge, mr}
        emb_dims: int, Dimension of embeddings
        drop_path: float, stochastic depth decay rule
        use_distance_masking: bool, Use distance masking (similar to GreedyViG) in the graph convolution
        blocks: list, [2,2,6,2] # number of basic blocks in the backbone
        channels: list, [80, 160, 400, 640] # number of channels of deep features
        img_size: int or list, Image size (height, width) or single int for square images
        window_size: int, Window size for the graph convolution
        use_shifts: bool, Use shifting windows in the graph convolution
        adapt_knn: bool, Adaptively adjust the number of k-nearest neighbors based on the input size
        use_reduce_ratios: bool, Use reduce ratios for the number of channels in the graph convolution
    """

    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        self._set_model_config_from_opt(opt)
        self._print_model_config()

        self.stem = Stem(out_dim=self.channels[0], act=self.act)
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.channels[0],
                math.ceil(self.img_size[0] / 4),
                math.ceil(self.img_size[1] / 4),
            )
        )
        self.pos_embed_interpolation_mode = None

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(Downsample(self.channels[i - 1], self.channels[i]))
            for j in range(self.blocks[i]):
                shift_size = 0
                if j % 2 != 0 and self.use_shifts:
                    shift_size = self.window_size[i] // 2

                fc2_g = self.fc2_g[i][j] if isinstance(self.fc2_g, list) else self.fc2_g

                wg = WindowGrapher(
                    in_channels=self.channels[i],
                    kernel_size=self.num_knn[idx],
                    window_size=self.window_size[i],
                    dilation=1,
                    conv=self.conv,
                    act=self.act,
                    norm=self.norm,
                    bias=self.bias,
                    stochastic=self.stochastic,
                    epsilon=self.epsilon,
                    drop_path=self.dpr[idx],
                    relative_pos=True,
                    shift_size=shift_size,
                    r=self.reduce_ratios[i],
                    input_resolution=(
                        math.ceil((self.img_size[0] / 4) / (2**i)),
                        math.ceil((self.img_size[1] / 4) / (2**i)),
                    ),
                    adapt_knn=self.adapt_knn,
                    graph_type=self.graphs_types[i][j],
                    use_distance_masking=self.use_distance_masking,
                    use_cpe=self.use_cpe,
                    WG_type=self.WG_type,
                    fc1=self.fc1,
                    fc1_act=self.fc1_act,
                    fc2_g=fc2_g,
                    fc2_act=self.fc2_act,
                    basic_conv_groups=self.basic_conv_groups,
                    extra_fc=self.extra_fc,
                )

                ffn = FFN(
                    self.channels[i],
                    self.channels[i] * self.ffn_expand_ratios[i],
                    act=self.act,
                    drop_path=self.dpr[idx],
                )

                self.backbone += [Seq(wg, ffn)]
                idx += 1

        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(
            nn.Conv2d(self.channels[-1], self.emb_dims, 1, bias=True),
            nn.BatchNorm2d(self.emb_dims),
            act_layer(self.act),
            nn.Dropout(opt.dropout),
            nn.Conv2d(self.emb_dims, self.n_classes, 1, bias=True),
        )
        self.model_init()

    def _set_model_config_from_opt(self, opt):
        self.n_classes = opt.n_classes
        self.WG_type = opt.WG_type
        self.k = opt.k
        self.act = opt.act
        self.norm = opt.norm
        self.bias = opt.bias
        self.epsilon = opt.epsilon
        self.stochastic = opt.use_stochastic
        self.conv = opt.conv
        self.emb_dims = opt.emb_dims
        self.drop_path = opt.drop_path
        self.use_distance_masking = opt.use_distance_masking
        self.blocks = opt.blocks
        self.channels = opt.channels
        self.img_size = (
            [opt.img_size, opt.img_size]
            if isinstance(opt.img_size, int)
            else opt.img_size
        )
        self.use_shifts = opt.use_shifts
        self.n_blocks = sum(self.blocks)
        self.window_size = [opt.window_size for _ in range(len(self.blocks))]
        self.reduce_ratios = [2, 2, 1, 1] if opt.use_reduce_ratios else [1, 1, 1, 1]
        self.adapt_knn = opt.adapt_knn
        self.glocal = opt.glocal
        self.masking_pad_nodes = opt.masking_pad_nodes
        self.cnn_assist = opt.cnn_assist
        self.use_cpe = opt.use_cpe
        self.WG_type = opt.WG_type
        self.ffn_expand_ratios = opt.ffn_expand_ratios
        self.fc1 = opt.fc1
        self.fc1_act = opt.fc1_act
        self.fc2_g = opt.fc2_g
        self.fc2_act = opt.fc2_act
        self.basic_conv_groups = opt.basic_conv_groups
        self.extra_fc = opt.extra_fc

        if self.glocal:
            self.graphs_types = [
                ["local" if i % 2 == 0 else "global" for i in range(block)]
                for block in self.blocks
            ]
        else:
            self.graphs_types = [
                ["local" for _ in range(block)] for block in self.blocks
            ]

        self.dpr = [
            x.item() for x in torch.linspace(0, self.drop_path, self.n_blocks)
        ]  # stochastic depth decay rule
        self.num_knn = [
            int(x.item()) for x in torch.linspace(self.k, self.k, self.n_blocks)
        ]  # number of knn's k
        # max_dilation = 49 // max(num_knn)

    def _print_model_config(self):
        print("\n\nWiGNet model configuration:\n")
        print(f"  - WG type: {self.WG_type}")
        print(f"  - Number of classes: {self.n_classes}")
        print(f"  - Window size: {self.window_size}\n")
        print(f"  - CNN-ASSIST: {self.cnn_assist}")
        print(f"  - Graph types: {self.graphs_types}")
        print(f"  - Image size: ({self.img_size})")
        print(f"  - Use shifting windows: {self.use_shifts}")
        print(f"  - Adapt knn: {self.adapt_knn}")
        print(f"  - Knn: {self.k}")
        print(f"  - Reduce ratios: {self.reduce_ratios}")
        print(f"  - Channel: {self.channels}")
        print(f"  - Blocks: {self.blocks}")
        print(f"  - Conv: {self.conv}")
        print(f"  - FFN expand ratios: {self.ffn_expand_ratios}")

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def _interpolate_pos(self, H, W):
        if self.pos_embed.shape[2] == H and self.pos_embed.shape[3] == W:
            return self.pos_embed

        print("\n  - Interpolate outer pos...")
        assert self.pos_embed_interpolation_mode in ["bilinear", "bicubic"]
        return F.interpolate(
            self.pos_embed, size=(H, W), mode=self.pos_embed_interpolation_mode
        )

    def forward(self, inputs):
        x = self.stem(inputs)
        B, C, H, W = x.shape
        x = x + self._interpolate_pos(H, W)

        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)  # [bs, 384, 1, 1]
        return self.prediction(x).squeeze(-1).squeeze(-1)


class OptInit:
    """All arguments will be overridden by kwargs"""

    def __init__(
        self,
        num_classes=1000,
        WG_type="WG",
        drop_path_rate=0.0,
        knn=9,
        use_shifts=True,
        use_reduce_ratios=False,
        img_size=224,
        adapt_knn=False,
        channels=None,
        blocks=None,
        emb_dims=1024,
        window_size=7,
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

        self.WG_type = WG_type
        self.k = knn  # neighbor num (default:9)
        self.conv = "mr"  # graph conv layer {edge, mr}
        self.act = "gelu"  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = "batch"  # batch or instance normalization {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.blocks = blocks  # number of basic blocks in the backbone
        self.channels = channels  # number of channels of deep features
        self.n_classes = num_classes  # Dimension of out_channels
        self.emb_dims = emb_dims  # Dimension of embeddings
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
                    f"  - Overriding {key} from the kwargs in OptInit with value {value}"
                )
                setattr(self, key, value)


@register_model
def wignext_ti_224_gelu(pretrained=False, **kwargs):  # wignext-ti
    opt = OptInit(
        **kwargs,
        channels=[36, 72, 160, 288],
        blocks=[2, 2, 6, 2],
        WG_type="Conv_WG",
        glocal=True,
        use_shifts=False,
        adapt_knn=False,
    )
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["wignn_224_gelu"]
    return model
