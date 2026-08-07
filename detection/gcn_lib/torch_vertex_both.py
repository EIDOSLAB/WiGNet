# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
import sys
from .torch_local import (
    window_partition,
    window_reverse,
    PatchEmbed,
    window_partition_channel_last,
)
import time
from einops import rearrange, repeat
import math

def pr(x: torch.Tensor, offset: str = ""):
    print(f"{offset} {list(x.shape)}")
    
    
# class MRConv2d(nn.Module):
#     """
#     Draft for the Rough-GAT layer
#     """

#     def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
#         super(MRConv2d, self).__init__()
#         self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)
        
#         ############## For the Q-GAT implementation
#         self.C = in_channels
#         self.H = 16
#         self.W = 16
#         self.N = self.H * self.W
#         self.num_heads = 1
#         self.head_dim = self.C // self.num_heads
#         self.scale = self.head_dim ** -0.5
        
#         self.q_proj = nn.Linear(self.C, self.C, bias=True)
        
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * self.H - 1) * (2 * self.W - 1), self.num_heads)
#         )
        
#         self.register_buffer("relative_position_index", self._get_rel_pos_index())
#         nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
#     def _get_rel_pos_index(self):
#         """Get pair-wise relative position index for each token inside the window."""
#         coords_h = torch.arange(self.H)
#         coords_w = torch.arange(self.W)
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, W
#         coords_flatten = torch.flatten(coords, 1)  # 2, H*W
        
#         # Compute relative coordinates
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2
        
#         # Shift to start from 0
#         relative_coords[:, :, 0] += self.H - 1  # shift h-coordinates
#         relative_coords[:, :, 1] += self.W - 1  # shift w-coordinates
        
#         # Convert to single index
#         relative_coords[:, :, 0] *= 2 * self.W - 1
#         relative_position_index = relative_coords.sum(-1)  # H*W, H*W
        
#         return relative_position_index

#     # def forward(self, x, edge_index, y=None):
#     ############### The Rough-MR-Attn implementation
#     #     # x: [B, C, N, 1]
#     #     x = F.normalize(x, p=2.0, dim=1)
#     #     x = x.squeeze(-1)  # [B, C, N]
#     #     attn = x.transpose(1, 2) @ x  # [B, N, N]
#     #     attn = F.softmax(attn, dim=-1)  # [B, N, N]
        
#     #     x_k = x.unsqueeze(2)  # [B, C, 1, N]
#     #     attn = attn.unsqueeze(1)  # [B, 1, N, N]
#     #     weighted_nodes = attn * x_k  # [B, C, N, N]

#     #     x_expanded = x.unsqueeze(3)  # [B, C, N, 1]
#     #     diff = weighted_nodes - x_expanded  # [B, C, N, N]
#     #     x_j, _ = torch.max(diff, -1, keepdim=True) # [B, C, N, 1]
        
#     #     x = x.unsqueeze(-1) # [B, C, N, 1]
#     #     b, c, n, _ = x.shape
#     #     x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _) # [B, 2C, N, 1]
#     #     out = self.nn(x) # [B, C, N, 1]
#     #     return out
    
#     # def forward(self, x, edge_index=None, y=None):
#     ########## The Rough-GAT implementation
#     #     rel_pos = edge_index
        
#     #     B, C, N, _ = x.shape  # x: [B, C, N, 1]
#     #     x = F.normalize(x, p=2.0, dim=1) # ||x||_2 = 1
#     #     rel_pos = -rel_pos / 2.0  # [-1, 1]
#     #     x = x.squeeze(-1)  # [B, C, N]
#     #     attn = ((x.transpose(1, 2) @ x) + rel_pos) / math.sqrt(C)  # [B, N, N]
#     #     attn = F.softmax(attn, dim=-1)  # [B, N, N]

#     #     x_j = attn @ x.transpose(1, 2) # [B, N, C]
#     #     x_j = x_j.transpose(1, 2).unsqueeze(-1)  # [B, C, N, 1]

#     #     x = x.unsqueeze(-1) # [B, C, N, 1]
#     #     x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(B, 2 * C, N, _) # [B, 2C, N, 1]
#     #     return self.nn(x)  # [B, C, N, 1]
    
#     def forward(self, x, edge_index=None, y=None):
#         rel_pos = edge_index
#         rel_pos = -rel_pos / 2.0
        
#         # x: [B, C, N, 1]
#         B, C, N, _ = x.shape
        
#         x = x.squeeze(-1).transpose(1, 2)  # [B, N, C]
        
#         if self.num_heads > 1:
#             pass
#         else:
#             Q = self.q_proj(x)  # [B, N, C]
#             Q = F.normalize(Q, p=2.0, dim=-1)  # Normalize Q
#             K = F.normalize(x, p=2.0, dim=-1)  # [B, N, C]
#             V = x  # [B, N, C] - keep V unnormalized
            
#             attn = ((Q @ K.transpose(-2, -1)) + rel_pos) / math.sqrt(C)  # [B, N, N]
#             attn = attn.unsqueeze(1)  # [B, 1, N, N] for compatibility
        
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.view(-1)
#         ].view(N, N, -1)  # [N, N, num_heads]
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        
#         attn = attn + relative_position_bias.unsqueeze(0)  # [B, num_heads, N, N]
#         attn = F.softmax(attn, dim=-1)
        
#         if self.num_heads > 1:
#             pass
#         else:
#             attn = attn.squeeze(1)  # [B, N, N]
#             x_attended = attn @ V  # [B, N, C]
#             x_attended = x_attended.transpose(1, 2)  # [B, C, N]
        
#         x = x.transpose(1, 2)  # [B, C, N] - transpose back
#         x_attended = x_attended.unsqueeze(-1)  # [B, C, N, 1]
#         x = x.unsqueeze(-1)  # [B, C, N, 1]
        
#         x_combined = torch.cat([x.unsqueeze(2), x_attended.unsqueeze(2)], dim=2)
#         x_combined = x_combined.reshape(B, 2 * C, N, 1) # [B, 2C, N, 1]
#         return self.nn(x_combined)  # [B, C, N, 1]


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        # x: [B, C, N, 1]
        # edge_index: [2, B, N, k]
        
        # print("\n\n\n--- MRConv2d forward ---")
        # print(edge_index.shape)
        # print("\n\n")
        
        
        x_i = batched_index_select(x, edge_index[1]) # [B, C, N, k]
        if y is not None:
            x_j = batched_index_select(y, edge_index[0]) # [B, C, N, k]
        else:
            x_j = batched_index_select(x, edge_index[0]) # [B, C, N, k]
    
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True) # [B, C, N, 1]
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _) # [B, 2C, N, 1]
        out = self.nn(x) # [B, C, N, 1]
        return out
    
    
class LightMRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(LightMRConv2d, self).__init__()

    def forward(self, x, edge_index, y=None):
        # x: [B, C, N, 1]
        # edge_index: [2, B, N, k]
        x_i = batched_index_select(x, edge_index[1]) # [B, C, N, k]
        if y is not None:
            x_j = batched_index_select(y, edge_index[0]) # [B, C, N, k]
        else:
            x_j = batched_index_select(x, edge_index[0]) # [B, C, N, k]
    
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True) # [B, C, N, 1]
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _) # [B, 2C, N, 1]
        return x



class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(
            self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True
        )
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(
        self, in_channels, out_channels, conv="edge", act="relu", norm=None, bias=True
    ):
        super(GraphConv2d, self).__init__()
        if conv == "edge":
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "mr":
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "light_mr":
            self.gconv = LightMRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "sage":
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == "gin":
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError("conv:{} is not supported".format(conv))

    def forward(self, x, edge_index=None, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        dilation=1,
        conv="edge",
        act="relu",
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
        r=1,
        use_distance_masking=False,
    ):
        super(DyGraphConv2d, self).__init__(
            in_channels, out_channels, conv, act, norm, bias
        )
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(
            kernel_size, dilation, stochastic, epsilon
        )
        self.use_distance_masking = use_distance_masking

    def _mask_edge_index(self, edge_index, adj_mask):
        """
        adj_mask could also be the dist_mask. It works the same way.
        
        edge_index: [2, B*M, N, k]
        adj_mask: [B*M, N, k] (=1 -> keep / =0 -> masked)
        
        e.g.:
        adj_mask: [[1,1,0,0],
                   [1,1,1,0]]
        
        edge_index_j: [[14,43,20,21],
                       [18,12,32,24]]
        
        edge_index_i: [[0,0,0,0],
                       [1,1,1,1],
                       [2,2,2,2]]
        
        edge_index_j in output: [[14,43,0,0],
                                [18,12,32,1]]
        """
        
        if adj_mask is None:
            return edge_index
        
        edge_index_j = edge_index[0] # [B, N, k] (targets)
        edge_index_i = edge_index[1] # [B, N, k] (sources)
        adj_mask_inv = torch.ones_like(adj_mask) - adj_mask
        edge_index_j = (edge_index_j * adj_mask) + (edge_index_i * adj_mask_inv) # Replace masked edges with the corresponding edge_index_i (self-loop)
        return torch.stack((edge_index_j, edge_index_i), dim=0).long()
    
    def _per_window_mean_dist_estimation(self, x):
        """
        Estimates the mean and std of the distance between nodes in each window using the method used in GreedyViG.
        Args:
            x: [B*M, C, ws, ws] (M=num_windows_per_image)
        Returns:
            mean: [B*M, 1, 1] - Mean distance in each window
            std:  [B*M, 1, 1] - Standard deviation of distances in each window
        """
        B, C, H, W = x.shape
        x = F.normalize(x, p=2.0, dim=1)
        x_rolled = torch.cat([x[:, :, -H // 2:, :], x[:, :, :-H // 2, :]], dim=2)
        x_rolled = torch.cat(
            [x_rolled[:, :, :, -W // 2:], x_rolled[:, :, :, :-W // 2]], dim=3
        )

        norm = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True) # [B*M, 1, ws, ws]
        mean = torch.mean(norm, dim=[2, 3], keepdim=True) # [B*M, 1, 1, 1]
        std = torch.std(norm, dim=[2, 3], keepdim=True) # [B*M, 1, 1, 1]
        return [mean.squeeze(-1), std.squeeze(-1)] # [B*M, 1, 1]
    
    def _create_dist_mask(self, distances, dist_stats):
        mean, std = dist_stats
        dist_mask = torch.where(distances < mean - std, 1, 0)
        return dist_mask

    def forward(self, x, relative_pos=None, adj_mask=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
            
        x = x.reshape(B, C, -1, 1).contiguous() # [B, C, N, 1]
        
        #####################
        #####################

        # edge_index, distances = self.dilated_knn_graph(x, y, relative_pos) # [2, B*M, N, k], [B*M, N, k] or None
        ############### NOTE: 
        # I had the line above which also expects the "distances", however, it is not being used. So I changed it to the line below.
        # ofc, I also removed the distances from the return of dilated_knn_graph function. By this, both Wignet and ViG should be fine.
        
        edge_index = self.dilated_knn_graph(x, y, relative_pos) # [2, B*M, N, k], [B*M, N, k] or None
        edge_index = self._mask_edge_index(edge_index, adj_mask)
        
        # if self.use_distance_masking:
        #     assert distances is not None 
        #     dist_stats = self._per_window_mean_dist_estimation(x)
        #     dist_mask = self._create_dist_mask(distances, dist_stats)
        #     edge_index = self._mask_edge_index(edge_index, dist_mask)
        
        # edge_index = relative_pos
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)

        return x.reshape(x.shape[0], -1, H, W).contiguous(), edge_index
    

class ConditionalPositionEncoding(nn.Module):
    """
    Implementation of conditional positional encoding. For more details refer to paper: 
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )

    def forward(self, x):
        x = self.pe(x) + x
        return x


class WindowGrapher(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=9,
        window_size=8,
        dilation=1,
        conv="edge",
        act="relu",
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
        drop_path=0.0,
        relative_pos=False,
        shift_size=0,
        r=1,
        input_resolution=(256 // 4, 256 // 4),
        adapt_knn=True,
        graph_type="local",
        use_distance_masking=False,
        masking_pad_nodes=False,
        use_cpe=False,
        WG_type="WG", # The WindowGrapher type (WG, FCHalf_WG,)
        fc2_g=1,
    ):
        super(WindowGrapher, self).__init__()
        
        self.WG_type = WG_type
        
        self.n_nodes = window_size * window_size # K: Previously: self.pretrained_window_size * self.pretrained_window_size
        self.use_distance_masking = use_distance_masking
        self.masking_pad_nodes = masking_pad_nodes
        self.use_cpe = use_cpe
        assert graph_type in ["local", "global"]
        self.graph_type = graph_type
        shift_size = 0 if self.graph_type == "global" else shift_size
        self._check_and_adjust_window_partitioning_params(
            window_size, input_resolution, kernel_size, shift_size, r
        )
        self._print_module_config()
        
        if self.use_cpe:
            print("  - Using Conditional Position Encoding (CPE)!")
            self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)

        fc1_in_channels = in_channels
        fc1_out_channels = in_channels // 2 if self.WG_type == "FCHalf_WG" else in_channels
        self.fc1 = nn.Sequential(
            nn.Conv2d(fc1_in_channels, fc1_out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(fc1_out_channels),
        )

        self.graph_conv = DyGraphConv2d(
            fc1_out_channels,
            fc1_out_channels * 2,
            kernel_size,
            dilation,
            conv,
            act,
            norm,
            bias,
            stochastic,
            epsilon,
            r=r,
            use_distance_masking=self.use_distance_masking,
        )
        
        if self.WG_type in ["WG", "FCHalf_WG"]:
            fc2_k, fc2_p = 1, 0
        elif self.WG_type in ["Conv_WG", "Conv_FCHalf_WG"]:
            fc2_k, fc2_p = 3, 1
        else:
            raise NotImplementedError(f"Unsupported WG type: {self.WG_type}")

        print(f"  - fc2 kernel: {fc2_k}, groups: {fc2_g}")
        self.fc2 = nn.Sequential(
            nn.Conv2d(fc1_out_channels * 2, fc1_in_channels, fc2_k, stride=1, padding=fc2_p, groups=fc2_g),
            nn.BatchNorm2d(fc1_in_channels),
        )
        
        # if "light" in conv.lower():
        #     self.fc2.append(act_layer(act))
        #     print(f"  - fc2 has act layer")
        print(f"  - fc2 does not using act layer")
      
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.relative_pos = self._create_relative_pos(in_channels, self.n_nodes, r) if relative_pos else None
        pdd_mask = self._set_input_resolution_and_padding_mask(input_resolution) # self.Hp, self.Wp

        attn_mask = None
        adj_mask = None
        if self.shift_size > 0:
            attn_mask = self._create_attn_mask(self.Hp, self.Wp)
            if adapt_knn:
                adj_mask = self._create_adj_mask(attn_mask)

        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("adj_mask", adj_mask)
        self.register_buffer("pdd_mask", pdd_mask)
        
    def _print_module_config(self):
        print(f"\nWindowGrapher ({self.WG_type}) Config:")
        print(f"  - window_size: {self.window_size}")
        print(f"  - graph_type: {self.graph_type}")
        print(f"  - shift_size: {self.shift_size}")
        print(f"  - input_resolution: {self.input_resolution}")
        print(f"  - kernel_size: {self.kernel_size}")
        print(f"  - r: {self.r}")
        print(f"  - dist_masking: {self.use_distance_masking}")
        
        
    def _check_and_adjust_window_partitioning_params(self, window_size, input_resolution, kernel_size, shift_size, r):
        assert window_size > 0 or window_size == -1, "window_size must be positive or -1 to disable window partitioning"
        if window_size == -1:
            assert shift_size == 0
        else:
            if min(input_resolution) <= window_size:
                shift_size = 0
                window_size = -1 # to disable window partitioning
            else:
                assert 0 <= shift_size < window_size, "shift_size must in 0-window_size"
                if shift_size > 0:
                    assert shift_size == window_size // 2, "shift_size must be half of window_size"

                max_connection_allowed = (window_size // r) ** 2
                if shift_size > 0:
                    assert shift_size % r == 0
                    max_connection_allowed = (shift_size // r) ** 2

                assert (
                    kernel_size <= max_connection_allowed
                ), f"trying k = {kernel_size} while the max can be: {max_connection_allowed}"
        
        self.window_size = window_size
        self.shift_size = shift_size
        self.r = r
        self.kernel_size = kernel_size
        self.input_resolution = input_resolution
        
    def _set_input_resolution_and_padding_mask(self, input_resolution):
        H, W = input_resolution
        if self.window_size > 0:
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
        else:
            Hp = H
            Wp = W

        self.H = H
        self.W = W
        self.Hp = Hp
        self.Wp = Wp
        
        pdd_mask = None
        if Hp != H or Wp != W:
            print(f"  - Input will need padding: H: {H} -> {Hp}, W: {W} -> {Wp}")
            if self.masking_pad_nodes:
                pdd_mask = self._create_padding_mask()
        return pdd_mask
            
            
    def _create_relative_pos(self, in_channels, n_nodes, r):
        print("  - Using relative_pos")
        relative_pos_tensor = (
            torch.from_numpy(
                np.float32(
                    get_2d_relative_pos_embed(in_channels, int(n_nodes**0.5))
                )
            )
            .unsqueeze(0)
            .unsqueeze(1)
        )
        # Adjusting rel pos because of r
        relative_pos_tensor = F.interpolate(
            relative_pos_tensor,
            size=(n_nodes, n_nodes // (r * r)),
            mode="bicubic",
            align_corners=False,
        )
        return nn.Parameter(
            -relative_pos_tensor.squeeze(1), requires_grad=False
        )

    def _create_attn_mask(self, H, W):
        print(f"  - Shifting windows!")
        
        img_mask = torch.zeros((1, 1, H, W))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        mask_windows_unf = window_partition(
            img_mask, self.window_size
        )  # nW, 1, window_size, window_size,
        mask_windows = mask_windows_unf.view(-1, self.window_size * self.window_size) # nW x N

        if self.r > 1:
            mask_windows_y = F.max_pool2d(mask_windows_unf, self.r, self.r)
            mask_windows_y = mask_windows_y.view(
                -1, (self.window_size // self.r) * (self.window_size // self.r)
            )
        else:
            mask_windows_y = mask_windows

        attn_mask = mask_windows_y.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW x N x (N // r)
        HIGH, ZERO = float(1000000.0), float(0.0)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, HIGH).masked_fill(attn_mask == 0, ZERO)
        return attn_mask # nW x N x (N // r)

    def _create_adj_mask(self, attn_mask):
        print("  - Adapting knn!")
        adj_mask = torch.empty(
            (attn_mask.shape[0], attn_mask.shape[1], self.kernel_size)
        )  # nW x N x k
        for w in range(attn_mask.shape[0]):
            for i in range(attn_mask.shape[1]):
                all_connection = torch.sum(attn_mask[w, i] == 0)
                scaled_knn = (self.kernel_size * all_connection) // (
                    self.window_size * (self.window_size // self.r)
                )
                n_connections_allowed = int(max(scaled_knn, 3.0))
                # print(f'Window: {w} node {i} - allowed_connection = {all_connection} (k = {n_connections_allowed})')
                masked = torch.zeros(self.kernel_size - n_connections_allowed)
                un_masked = torch.ones(n_connections_allowed)
                adj_mask[w, i] = torch.cat([un_masked, masked], dim=0)
                
        return adj_mask # nW x N x k
    
    def _merge_pos_attn_pdd(self, B, H, W):
        
        def _adjust_rel_pos_shape(relative_pos, H, W):
            if relative_pos is None or H * W == relative_pos.shape[1]:
                return relative_pos
            else:
                N = H * W
                N_reduced = N // (self.r * self.r)
                return F.interpolate(
                    relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic"
                ).squeeze(0)
            
        if self.window_size > 0:
            relative_pos = _adjust_rel_pos_shape(
                self.relative_pos, self.window_size, self.window_size
            )
        else:
            relative_pos = _adjust_rel_pos_shape(self.relative_pos, H, W)

        if self.attn_mask is None:
            if self.pdd_mask is not None:
                nW = self.pdd_mask.shape[0]
                return  relative_pos.repeat(nW * B, 1, 1) + self.pdd_mask.repeat(B, 1, 1)
            return relative_pos

        assert self.window_size > 0
        assert self.shift_size > 0
        
        nW = self.attn_mask.shape[0]
        pos_att = relative_pos.repeat(nW * B, 1, 1) + self.attn_mask.repeat(B, 1, 1)  # B, N, N
        if self.pdd_mask is not None:
            assert self.pdd_mask.shape[0] == nW
            return pos_att + self.pdd_mask.repeat(B, 1, 1)
        return pos_att

    def _create_padding_mask(self):
        if self.r > 1:
           raise NotImplementedError("Masking pad nodes not implemented for r > 1")

        Hp, Wp = self.Hp, self.Wp
        H, W = self.H, self.W
        
        mask = torch.ones((1, 1, Hp, Wp))
        mask[:, :, H:, :] = 0  # Bottom padding
        mask[:, :, :, W:] = 0  # Right padding
        
        if self.shift_size > 0:
            mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        assert self.window_size > 0
        mask_windows_unf = window_partition(mask, self.window_size, graph_type=self.graph_type)  # nW, 1, ws, ws
        mask_windows = mask_windows_unf.view(-1, self.window_size * self.window_size) # nW x N
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW x N x (N // r)
        HIGH, ZERO = float(1000000.0), float(0.0)
        mask = mask.masked_fill(mask != 0, HIGH).masked_fill(mask == 0, ZERO)
        print("  - Padding mask created with shape: ", list(mask.shape))
        return mask # nW x N x (N // r)
        
    def _pad_x(self, x):
        H, W = x.shape[2], x.shape[3]
        assert H == self.H and W == self.W, \
        f"Input shape doesn't match the expected H and W: {self.H}, {self.W} != {H}, {W}"
        
        if W % self.window_size != 0:
            x = F.pad(x, (0, self.window_size - W % self.window_size))
        if H % self.window_size != 0:
            x = F.pad(x, (0, 0, 0, self.window_size - H % self.window_size))
            
        assert self.Hp == x.shape[2] and self.Wp == x.shape[3], \
        f"The padded input shape doesn't match the expected Hp and Wp: {self.Hp}, {self.Wp} != {x.shape[2]}, {x.shape[3]}"
    
        return x

    def forward(self, x):

        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape

        if self.window_size > 0:
            x = self._pad_x(x)

        if self.shift_size > 0:
            assert self.window_size > 0
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        if self.window_size > 0:
            x = window_partition(x, window_size=self.window_size, graph_type=self.graph_type)
        if self.use_cpe:
            x = self.cpe(x)

        pos_att_pdd = self._merge_pos_attn_pdd(B=B, H=self.Hp, W=self.Wp) # [B*nW, N, N] (N=ws*ws)
        
        adj_mask = None
        if self.adj_mask is not None:
            adj_mask = self.adj_mask.repeat(B, 1, 1)

        x, edge_index = self.graph_conv(x, pos_att_pdd, adj_mask)

        if self.window_size > 0:
            x = window_reverse(x, self.window_size, H=self.Hp, W=self.Wp, graph_type=self.graph_type)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        x = x[:, :, :H, :W]
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
