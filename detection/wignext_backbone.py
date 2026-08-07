import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model
from timm.models.helpers import clean_state_dict


from ..layers import act_layer, WindowGrapher

# from gcn_lib import act_layer, WindowGrapher
import math
from collections import OrderedDict

try:
    from mmdet.registry import MODELS

    # from mmdet.models.builder import BACKBONES as det_BACKBONES
    # from mmdet.utils import get_root_logger
    # from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False
    exit(1)


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
    def __init__(self, opt):
        super(DeepGCN, self).__init__()

        self._set_model_config_from_opt(opt)
        self._print_model_config()

        self.stem = Stem(out_dim=self.channels[0], act=self.act)
        self.pos_embed = self._create_pos_embed()

        if self.cnn_assist:
            raise NotImplementedError()

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:
                self.backbone.append(Downsample(self.channels[i - 1], self.channels[i]))
            for j in range(self.blocks[i]):
                shift_size = 0
                if j % 2 != 0 and self.use_shifts and self.window_size[i][j] > 0:
                    shift_size = self.window_size[i][j] // 2

                fc2_g = self.fc2_g[i][j] if isinstance(self.fc2_g, list) else self.fc2_g
                self.backbone += [
                    Seq(
                        WindowGrapher(
                            in_channels=self.channels[i],
                            kernel_size=self.num_knn[idx],
                            window_size=self.window_size[i][j],
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
                            masking_pad_nodes=self.masking_pad_nodes,
                            use_cpe=self.use_cpe,
                            WG_type=self.WG_type,
                            fc2_g=fc2_g,
                        ),
                        FFN(
                            self.channels[i],
                            self.channels[i] * self.ffn_expand_ratios[i],
                            act=self.act,
                            drop_path=self.dpr[idx],
                        ),
                    )
                ]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def _print_model_config(self):
        print("\n\nWiGNet model configuration:\n")
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
        print(f"  - ffn_expand_ratios: {self.ffn_expand_ratios}")

    def _set_model_config_from_opt(self, opt):
        print(opt)
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
        self.blocks = opt.blocks
        self.channels = opt.channels
        self.out_indices = opt.out_indices
        self.pretrained = opt.pretrained
        self.glocal = opt.glocal
        self.use_shifts = opt.use_shifts
        self.n_blocks = sum(self.blocks)
        self.img_size = (
            [opt.img_size, opt.img_size]
            if isinstance(opt.img_size, int)
            else opt.img_size
        )
        assert len(self.img_size) == 2
        self.reduce_ratios = [2, 2, 1, 1] if opt.use_reduce_ratios else [1, 1, 1, 1]
        self.adapt_knn = opt.adapt_knn
        self.use_distance_masking = opt.use_distance_masking
        self.masking_pad_nodes = opt.masking_pad_nodes
        self.cnn_assist = opt.cnn_assist
        self.use_cpe = opt.use_cpe
        self.ffn_expand_ratios = opt.ffn_expand_ratios
        self.fc2_g = opt.fc2_g

        self._set_window_size(opt)

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

    def _set_window_size(self, opt):
        self.window_size = [
            [opt.window_size for _ in range(block)] for block in self.blocks
        ]
        if not opt.apply_last_windows:
            for i in range(len(self.window_size[-1])):
                self.window_size[-1][i] = -1
            if opt.nlw2:
                self.window_size[-2][-1] = -1
                self.window_size[-2][-2] = -1
            if opt.nlw4:
                self.window_size[-2][-3] = -1
                self.window_size[-2][-4] = -1

    def _create_pos_embed(self):
        if self.pretrained is not None:
            print("\nCreating pos_embed from checkpoint...")
            checkpoint = torch.load(self.pretrained, map_location="cpu")
            assert isinstance(checkpoint, dict) and "state_dict" in checkpoint
            state_dict = clean_state_dict(checkpoint["state_dict"])
            checkpoint_pos_embed = state_dict["pos_embed"]
            new_pos_embed = nn.Parameter(
                F.interpolate(
                    checkpoint_pos_embed, size=(200, 336), mode="bicubic"
                )  # TODO: infer size from self.img_size
            )
            return new_pos_embed

        print("\n  - self.pretrained is None, creating pos_embed from scratch...")
        return nn.Parameter(
            torch.zeros(
                1,
                self.channels[0],
                math.ceil(self.img_size[0] / 4),
                math.ceil(self.img_size[1] / 4),
            )
        )

    def init_weights(self):
        def remove_pos_and_mask(state_dict):
            _state_dict = OrderedDict()
            for k, v in state_dict.items():
                if (
                    "pos_embed" not in k
                    and "relative_pos"
                    not in k  # K: added after the recent refactor (because I changed n_nodes to self.window_size * self.window_size)
                    and "attn_mask" not in k
                    and "adj_mask" not in k
                ):
                    _state_dict[k] = v
            return _state_dict

        checkpoint = torch.load(self.pretrained, map_location="cpu")
        assert isinstance(checkpoint, dict) and "state_dict" in checkpoint
        print("\nRestoring model state from checkpoint...")
        state_dict = clean_state_dict(checkpoint["state_dict"])
        state_dict = remove_pos_and_mask(state_dict)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)

        print("\nMissing keys:")
        for k in missing_keys:
            print(f"  - {k}")

        print("\nUnexpected keys:")
        for k in unexpected_keys:
            print(f"  - {k}")
        print()

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _interpolate_pos(self, H, W):
        if self.pos_embed.shape[2] == H and self.pos_embed.shape[3] == W:
            return self.pos_embed

        print("\n  - Interpolate outer pos...")
        return F.interpolate(self.pos_embed, size=(H, W), mode="bicubic")

    def forward(self, inputs):
        x = self.stem(inputs)
        x = x + self._interpolate_pos(x.shape[2], x.shape[3])
        outs = []
        B, C, H, W = x.shape

        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

            if i in self.out_indices:
                outs.append(x)

        return outs


class OptInit:
    def __init__(
        self,
        num_classes=1000,
        drop_path_rate=0.0,
        knn=9,
        use_shifts=True,
        use_reduce_ratios=False,
        img_size=256,
        adapt_knn=False,
        window_size=8,
        apply_last_windows=True,
        nlw2=False,
        nlw4=False,
        conv="mr",
        channels=None,
        blocks=None,
        pretrained=None,
        out_indices=None,
        glocal=False,
        use_distance_masking=False,
        masking_pad_nodes=False,
        cnn_assist=False,
        use_cpe=False,
        WG_type="WG",
        ffn_expand_ratios=[4, 4, 4, 4],
        fc2_g=1,
    ):
        self.k = knn  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr}
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
        self.n_classes = num_classes
        self.emb_dims = 1024  # Dimension of embeddings
        self.window_size = window_size
        self.apply_last_windows = apply_last_windows
        self.nlw2 = nlw2  # nlw for the block befor the last one
        self.nlw4 = nlw4
        self.use_shifts = use_shifts
        self.img_size = img_size
        self.use_reduce_ratios = use_reduce_ratios
        self.adapt_knn = adapt_knn
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.glocal = glocal
        self.use_distance_masking = use_distance_masking
        self.masking_pad_nodes = masking_pad_nodes
        self.cnn_assist = cnn_assist
        self.use_cpe = use_cpe
        self.WG_type = WG_type
        self.ffn_expand_ratios = ffn_expand_ratios
        self.fc2_g = fc2_g

        if self.nlw2:
            assert not self.apply_last_windows


@MODELS.register_module()
def wignext_ti(pretrained=False, **kwargs):
    opt = OptInit(
        WG_type="Conv_WG",
        conv="mr",
        img_size=[800, 1344],
        use_reduce_ratios=False,
        use_shifts=False,
        adapt_knn=False,
        channels=[36, 72, 160, 288],
        blocks=[2, 2, 6, 2],
        out_indices=[1, 4, 11, 14],
        window_size=16,
        apply_last_windows=False,
        glocal=True,
        pretrained="path/to/pretrained/model.pth",
    )
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["wignn_256_gelu"]
    return model


@MODELS.register_module()
def wignext_s(pretrained=False, **kwargs):
    opt = OptInit(
        WG_type="Conv_WG",
        conv="mr",
        img_size=[800, 1344],
        use_reduce_ratios=False,
        use_shifts=False,
        adapt_knn=False,
        channels=[48, 96, 240, 384],
        blocks=[2, 2, 6, 2],
        out_indices=[1, 4, 11, 14],
        window_size=16,
        apply_last_windows=False,
        glocal=True,
        pretrained="path/to/pretrained/model.pth",
    )
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["wignn_256_gelu"]
    return model


@MODELS.register_module()
def wignext_m(pretrained=False, **kwargs):
    opt = OptInit(
        WG_type="Conv_WG",
        conv="mr",
        img_size=[800, 1344],
        use_reduce_ratios=False,
        use_shifts=False,
        adapt_knn=False,
        channels=[60, 120, 260, 480],
        blocks=[2, 2, 16, 2],
        out_indices=[1, 4, 21, 24],
        window_size=16,
        apply_last_windows=False,
        glocal=True,
        pretrained="path/to/pretrained/model.pth",
    )
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs["wignn_256_gelu"]
    return model


if __name__ == "__main__":
    print("main")
