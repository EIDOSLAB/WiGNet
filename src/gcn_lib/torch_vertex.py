
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
import sys
from .torch_local import window_partition, window_reverse, PatchEmbed, window_partition_channel_last
import time

from einops import rearrange, repeat


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

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
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
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
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
           self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)

        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        self.debug = True

    def _mask_edge_index(self, edge_index, adj_mask):
        # edge_index: [2, B, N, k]
        # adj_mask: [B, N, k] (=1 -> keep / =0 -> masked)

        if adj_mask is None:
            return edge_index
        
        
        # edge_index_j: [B, N, k]
        edge_index_j = edge_index[0]
        edge_index_i = edge_index[1]
        adj_mask_inv = torch.ones_like(adj_mask) - adj_mask
        # adj_mask: [[1,1,0,0],
        #            [1,1,1,0]]
        #
        # edge_index_j: [[14,43,20,21],
        #                [18,12,32,24]]
        #
        # edge_index_i: [[0,0,0,0],
        #                [1,1,1,1]]
        #
        # output: [[14,43,0,0],
        #          [18,12,32,1]]
        edge_index_j = (edge_index_j * adj_mask) + (edge_index_i * adj_mask_inv)
        
        return torch.stack((edge_index_j, edge_index_i), dim=0).long()



    def forward(self, x, relative_pos=None, adj_mask = None):
        # print('Doing gnn')
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            # print(f'y: {y.shape}')
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
                
        edge_index = self.dilated_knn_graph(x, y, relative_pos)

        edge_index = self._mask_edge_index(edge_index, adj_mask)

        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        

        if self.debug:
            return x.reshape(x.shape[0], -1, H, W).contiguous(), edge_index    
        return x.reshape(x.shape[0], -1, H, W).contiguous()
    





class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        
        x, edge_index = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
    

class WindowGrapher(nn.Module):
    """
    Local Grapher module with graph convolution and fc layers
    """
    def __init__(
            self,
            in_channels,
            kernel_size=9,
            windows_size = 7,
            dilation=1,
            conv='edge',
            act='relu',
            norm=None,
            bias=True,
            stochastic=False,
            epsilon=0.0,
            drop_path=0.0,
            relative_pos=False,
            shift_size = 0,
            r = 1,
            input_resolution = (224//4,224//4),
            adapt_knn = False):
        super(WindowGrapher, self).__init__()

        if min(input_resolution) <= windows_size:
            # if window size is larger than input resolution, we don't partition windows
            shift_size = 0
            windows_size = min(input_resolution)
        assert 0 <= shift_size < windows_size, "shift_size must in 0-window_size"
       

        max_connection_allowed = (windows_size // r)**2
        if shift_size > 0:
            assert shift_size % r == 0
            max_connection_allowed = (shift_size // r)**2

        assert kernel_size <= max_connection_allowed, f'trying k = {kernel_size} while the max can be: {max_connection_allowed}'


        self.windows_size = windows_size
        self.shift_size = shift_size
        self.r = r


        n_nodes = self.windows_size * self.windows_size

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.graph_conv = DyGraphConv2d(in_channels, (in_channels * 2), kernel_size, dilation, conv,
                  act, norm, bias, stochastic, epsilon, r = r)
        
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        
        
        
        
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n_nodes**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n_nodes, n_nodes//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)


        attn_mask = None
        adj_mask = None
        if self.shift_size > 0:
            print(f'Shifting windows!')
            H, W = input_resolution
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, 1, H, W))
            h_slices = (slice(0, -self.windows_size),
                        slice(-self.windows_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.windows_size),
                        slice(-self.windows_size, -self.shift_size),
                        slice(-self.shift_size, None))

            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, h, w] = cnt
                    cnt += 1

            # print(img_mask)
            # print('\n\n')

            mask_windows_unf = window_partition(img_mask, self.windows_size)  # nW, 1, windows_size, windows_size,
            
            
            mask_windows = mask_windows_unf.view(-1, self.windows_size * self.windows_size)
        
            if self.r > 1:
                mask_windows_y = F.max_pool2d(mask_windows_unf, self.r, self.r)
                mask_windows_y = mask_windows_y.view(-1, (self.windows_size // self.r) * (self.windows_size // self.r))
            else:
                mask_windows_y = mask_windows
            
            attn_mask = mask_windows_y.unsqueeze(1) - mask_windows.unsqueeze(2) # nW x N x (N // r)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(1000000.0)).masked_fill(attn_mask == 0, float(0.0))

            # Get n_connections_allowed for each node in each windows
            if adapt_knn:
                print('Adapting knn!')
                adj_mask = torch.empty((attn_mask.shape[0], attn_mask.shape[1], kernel_size)) # nW x N x k
                for w in range(attn_mask.shape[0]):
                    for i in range(attn_mask.shape[1]):
                        all_connection = torch.sum(attn_mask[w,i] == 0)
                        scaled_knn = (kernel_size * all_connection) // (self.windows_size * (self.windows_size // r))
                        n_connections_allowed = int(max(scaled_knn, 3.0))
                        # print(f'Window: {w} node {i} - allowed_connection = {all_connection} (k = {n_connections_allowed})')
                        masked = torch.zeros(kernel_size - n_connections_allowed)
                        un_masked = torch.ones(n_connections_allowed)
                        adj_mask[w,i] = torch.cat([un_masked,masked],dim=0)

        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("adj_mask", adj_mask)


    def _merge_pos_attn(self, batch_size):
        if self.attn_mask is None:
            return self.relative_pos

        if self.relative_pos is None:
            print('Should not be here..')
            self.relative_pos = torch.zeros((1, self.attn_mask.shape[1], self.attn_mask.shape[2])).to(self.attn_mask.device)
        
        nW_nGH = self.attn_mask.shape[0]
        #print(f'Attention mask: {self.attn_mask.shape}')
        return self.relative_pos.repeat(nW_nGH*batch_size,1,1) + self.attn_mask.repeat(batch_size, 1, 1) # B, N, N
    
        
        


    def forward(self, x):


        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        

        
        x = window_partition(x, window_size = self.windows_size)
        
        # merge relative_pos w/ attn_mask TODO
        pos_att = self._merge_pos_attn(batch_size=B)
        # pos_att: b*nW, N, N

        adj_mask = None
        if self.adj_mask is not None:
            adj_mask = self.adj_mask.repeat(B,1,1)
        x, edge_index = self.graph_conv(x, pos_att, adj_mask)


        x = window_reverse(x, self.windows_size, H=H, W=W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        x = self.fc2(x)

        x = self.drop_path(x) + _tmp
        
        return x
    
    