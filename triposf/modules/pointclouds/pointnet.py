# MIT License

# Copyright (c) 2020 Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, Andreas Geiger.
# Copyright (c) 2025 VAST-AI-Research and contributors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# modified from https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/encoder/pointnet.py

import torch
import torch.nn as nn
import copy
from torch import Tensor

# Fallback for torch_scatter if not available
try:
    from torch_scatter import scatter_mean
except ImportError:
    def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
        """Pure PyTorch port of torch_scatter.scatter_mean (same signature and defaults)."""
        dim = dim % src.dim()
        if index.dim() == 1:
            for _ in range(dim):
                index = index.unsqueeze(0)
        for _ in range(index.dim(), src.dim()):
            index = index.unsqueeze(-1)
        index = index.expand(src.size())
        if out is None:
            size = list(src.size())
            size[dim] = int(index.max()) + 1 if dim_size is None else dim_size
            out = src.new_zeros(size)
        out.scatter_add_(dim, index, src)
        count = torch.zeros_like(out).scatter_add_(dim, index, torch.ones_like(src))
        return out / count.clamp(min=1)

def scale_tensor(
    dat, inp_scale=None, tgt_scale=None
):
    if inp_scale is None:
        inp_scale = (-0.5, 0.5)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    assert tgt_scale[1] > tgt_scale[0] and inp_scale[1] > inp_scale[0]
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat.clamp(tgt_scale[0] + 1e-6, tgt_scale[1] - 1e-6)

# Resnet Blocks for pointnet
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.GELU(approximate="tanh")

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.xavier_uniform_(self.fc_0.weight)
        if self.fc_0.bias is not None:
            nn.init.constant_(self.fc_0.bias, 0)
        if self.shortcut is not None:
            nn.init.xavier_uniform_(self.shortcut.weight)
            if self.shortcut.bias is not None:
                nn.init.constant_(self.shortcut.bias, 0)
        
        nn.init.xavier_uniform_(self.fc_1.weight)
        if self.fc_1.bias is not None:
            nn.init.constant_(self.fc_1.bias, 0)
        
        
    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class LocalPoolPointnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, hidden_dim=128, scatter_type='mean', n_blocks=5):
        super().__init__()
        self.scatter_type = scatter_type
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.fc_pos = nn.Linear(in_channels, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, out_channels)

        if self.scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('Incorrect scatter type')
        self.initialize_weights()

    def initialize_weights(self):
        
        nn.init.xavier_uniform_(self.fc_pos.weight)
        if self.fc_pos.bias is not None:
            nn.init.constant_(self.fc_pos.bias, 0)

        nn.init.xavier_uniform_(self.fc_c.weight)
        if self.fc_c.bias is not None:
            nn.init.constant_(self.fc_c.bias, 0)

    def convert_to_sparse_feats(self, c, sparse_coords):
        '''
        Input:
            sparse_coords: Tensor [Nx, 4], point to sparse indices
            c: Tensor [B, res, C], input feats of each grid
        Output:
            c_out: Tensor [B, Np, C], aggregated grid feats of each point
        '''
        feats_new = torch.zeros((sparse_coords.shape[0], c.shape[-1]), device=c.device, dtype=c.dtype)
        offsets = 0
        
        batch_nums = copy.deepcopy(sparse_coords[..., 0])
        for i in range(len(c)):
            coords_num_i = (batch_nums == i).sum()
            feats_new[offsets: offsets + coords_num_i] = c[i, : coords_num_i]
            offsets += coords_num_i
        return feats_new

    def generate_sparse_grid_features(self, index, c, max_coord_num):
        # scatter grid features from points
        bs, fea_dim = c.size(0), c.size(2)
        res = max_coord_num
        c_out = c.new_zeros(bs, self.out_channels, res)
        c_out = scatter_mean(c.permute(0, 2, 1), index, out=c_out).permute(0, 2, 1) # B x res X C
        return c_out

    def pool_sparse_local(self, index, c, max_coord_num):
        '''
        Input:
            index: Tensor [B, 1, Np], sparse indices of each point
            c: Tensor [B, Np, C], input feats of each point
        Output:
            c_out: Tensor [B, Np, C], aggregated grid feats of each point
        '''
        
        bs, fea_dim = c.size(0), c.size(2)
        res = max_coord_num
        c_out = c.new_zeros(bs, fea_dim, res)
        c_out = self.scatter(c.permute(0, 2, 1), index, out=c_out)

        # gather feature back to points
        c_out = c_out.gather(dim=2, index=index.expand(-1, fea_dim, -1))
        return c_out.permute(0, 2, 1)

    @torch.no_grad()
    def coordinate2sparseindex(self, x, sparse_coords, res):
        '''
        Input:
            x: Tensor [B, Np, 3], points scaled at ([0, 1] * res)
            sparse_coords: Tensor [Nx, 4] ([batch_number, x, y, z])
            res: Int, resolution of the grid index
        Output:
            sparse_index: Tensor [B, 1, Np], sparse indices of each point
        '''
        B = x.shape[0]
        sparse_index = torch.zeros((B, x.shape[1]), device=x.device, dtype=torch.int64)
        
        index = (x[..., 0] * res + x[..., 1]) * res + x[..., 2]
        sparse_indices = copy.deepcopy(sparse_coords)
        sparse_indices[..., 1] = (sparse_indices[..., 1] * res + sparse_indices[..., 2]) * res + sparse_indices[..., 3]
        sparse_indices = sparse_indices[..., :2]
        
        for i in range(B):
            mask_i = sparse_indices[..., 0] == i
            coords_i = sparse_indices[mask_i, 1]
            coords_num_i = len(coords_i)
            sparse_index[i] = torch.searchsorted(coords_i, index[i])
                
        return sparse_index[:, None, :]

    def forward(self, p, sparse_coords, res=64, bbox_size=(-0.5, 0.5)):
        '''
        Input:
            p : Tensor [B, Np(819_200), 3]
            sparse_coords: Tensor [Nx, 4] ([batch_number, x, y, z])

        Output:
            sparse_pc_feats: [Nx, self.out_channels]
        '''
        batch_size, T, D = p.size()
        max_coord_num = 0
        for i in range(batch_size):
            max_coord_num = max(max_coord_num, (sparse_coords[..., 0] == i).sum().item() + 5)
        
        if D == 6:
            p, normals = p[..., :3], p[..., 3:]

        coord = (scale_tensor(p, inp_scale=bbox_size) * res)
        p = 2 * (coord - (coord.floor() + 0.5)) # dist to the centrios, [-1., 1.]
        index = self.coordinate2sparseindex(coord.long(), sparse_coords, res)

        if D == 6:
            p = torch.cat((p, normals), dim=-1)
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_sparse_local(index, net, max_coord_num=max_coord_num)
            
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        feats = self.generate_sparse_grid_features(index, c, max_coord_num=max_coord_num)
        feats = self.convert_to_sparse_feats(feats, sparse_coords)
        
        torch.cuda.empty_cache()
        return feats
