# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F
import sys

def pr(x, msg):
    print(f"{msg} {list(x.shape)}")

def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims) = [B, N, C]    
    Returns:
        pairwise distance: (batch_size, num_points, num_points) = [B, N, N]
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1) = [B, C, N, 1]
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        
        x = x.transpose(2, 1).squeeze(-1) # [B, N, C]
        batch_size, n_points, n_dims = x.shape
        
        n_part = 10000
        if n_points > n_part:
            raise Exception(f"This wasn't supposed to happen for Wignet!") # K
            ### memory efficient implementation ###
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach()) # [B*M, N, N]
            if relative_pos is not None:
                dist += relative_pos
            distances, nn_idx = torch.topk(-dist, k=k) # Both [B*M, N, k]
        
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1) 
        # center_idx: [B*M, N, k], each row is the source node's index, k times
        
    edge_index = torch.stack((nn_idx, center_idx), dim=0) # [2, B*M, N, k]
    return edge_index, distances



def dense_weighted_knn_adj(x, k=9, relative_pos=None):
    """Get KNN based on the pairwise distance and return adjacency matrix.
    Args:
        x: [B, C, N, 1] (normalized)
        k: int
    Returns:
        adj_matrix: [B, N, N]
        distances: [B, N, k] - distances to k nearest neighbors
    """
    
    x = x.transpose(2, 1).squeeze(-1) # [B, N, C]
    B, N, C = x.shape
    
    dist = pairwise_distance(x) # [B, N, N]
    if relative_pos is not None:
        dist += relative_pos
    distances, nn_idx = torch.topk(-dist, k=k + 1) # Both [B, N, k + 1]
    distances = -distances / 4
    
    adj_matrix = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)  # [B, N, N]
    adj_matrix.scatter_(2, nn_idx,  1 - distances)
    adj_matrix = adj_matrix - torch.eye(N, device=x.device).repeat(B, 1, 1)
    adj_matrix = adj_matrix.masked_fill(adj_matrix == 0, float('-inf'))
    adj_matrix = F.softmax(adj_matrix, dim=-1)
    return adj_matrix, distances


def random_adj(x, k=16, relative_pos=None):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        adj = torch.randint(low=0, high=k, size=(batch_size, n_points, n_points), device=x.device, dtype=x.dtype)
        return adj, None  # distances are not computed in this case


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        
        # nn_idx = torch.randint(1, 195, (batch_size,n_points,k)).to(dist.device)
        _, nn_idx = torch.topk(-dist, k=k)

        # print('Campling values')
        # nn_idx = torch.clamp(nn_idx, min=0, max=n_points-1)   

        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0) # targets, sources


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        
        assert self.dilation == 1, "Dilation more than 1 is not for now"

    def forward(self, x, y=None, relative_pos=None, conv_type=None):
        if y is not None:
            raise NotImplementedError("Was not supposed to happen for Wignet!")
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            if conv_type == 'gcn' or conv_type == 'light_gcn':
                adj, distances = dense_weighted_knn_adj(x, self.k * self.dilation, relative_pos)
                return adj, distances
            elif conv_type == 'gcn_random_graph':
                adj, distances = random_adj(x, self.k * self.dilation, relative_pos)
                return adj, distances
            else:
                edge_index, distances = dense_knn_matrix(x, self.k * self.dilation, relative_pos) # distances could be None if using memory efficient implementation
                return self._dilated(edge_index), distances
