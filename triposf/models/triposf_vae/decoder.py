# MIT License

# Copyright (c) Microsoft Corporation.
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

from typing import *
import torch
import torch.nn as nn
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations import MeshExtractResult
from ...representations.mesh import SparseFeatures2Mesh
from ...modules.sparse.linear import SparseLinear
from ...modules.sparse.nonlinearity import SparseGELU

class SparseOccHead(nn.Module):
    def __init__(self, channels: int, out_channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), out_channels),
        )

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return self.mlp(x)
    
class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
        self.pruning_head = SparseOccHead(self.out_channels, out_channels=1)
            
    def forward(self, x: sp.SparseTensor, pruning=False, training=True) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        if pruning:
            occ_prob = self.pruning_head(h)
            occ_mask = (occ_prob.feats >= 0.).squeeze(-1)
            if training == False:
                h = sp.SparseTensor(feats=h.feats[occ_mask], coords=h.coords[occ_mask])
            return h, occ_prob
        else:
            return h, None


class TripoSFVAEDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        use_sparse_flexicube: bool = True,
        use_sparse_sparse_flexicube: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        assert not (use_sparse_flexicube == False and use_sparse_sparse_flexicube == True)
        self.resolution = resolution
        self.rep_config = representation_config
        self.mesh_extractor = SparseFeatures2Mesh(res=self.resolution*4, use_color=self.rep_config.get('use_color', False), use_sparse_flexicube=use_sparse_flexicube, use_sparse_sparse_flexicube=use_sparse_sparse_flexicube)
        self.out_channels = self.mesh_extractor.feats_channels
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4,
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8,
            )
        ])
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
        for module in self.upsample:
            nn.init.constant_(module.pruning_head.mlp[-1].weight, 0)
            nn.init.constant_(module.pruning_head.mlp[-1].bias, 0)
    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)  
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i].float(), training=self.training)
            ret.append(mesh)
            
        return ret
    
    @torch.no_grad()
    def split_for_meshing(self, x: sp.SparseTensor, chunk_size=4, padding=4, verbose=False):
        
        sub_resolution = self.resolution // chunk_size
        upsample_ratio = 4 # hard-coded here
        assert sub_resolution % padding == 0
        out = []
        if verbose:
            print(f"Input coords range: x[{x.coords[:, 1].min()}, {x.coords[:, 1].max()}], "
                  f"y[{x.coords[:, 2].min()}, {x.coords[:, 2].max()}], "
                  f"z[{x.coords[:, 3].min()}, {x.coords[:, 3].max()}]")
            print(f"Resolution: {self.resolution}, sub_resolution: {sub_resolution}")
        
        for i in range(chunk_size):
            for j in range(chunk_size):
                for k in range(chunk_size):
                    # Calculate padded boundaries
                    start_x = max(0, i * sub_resolution - padding)
                    end_x = min((i + 1) * sub_resolution + padding, self.resolution)
                    start_y = max(0, j * sub_resolution - padding)
                    end_y = min((j + 1) * sub_resolution + padding, self.resolution)
                    start_z = max(0, k * sub_resolution - padding)
                    end_z = min((k + 1) * sub_resolution + padding, self.resolution)
                    
                    # Store original (unpadded) boundaries for later cropping
                    orig_start_x = i * sub_resolution
                    orig_end_x = (i + 1) * sub_resolution
                    orig_start_y = j * sub_resolution
                    orig_end_y = (j + 1) * sub_resolution
                    orig_start_z = k * sub_resolution
                    orig_end_z = (k + 1) * sub_resolution

                    if verbose:
                        print(f"\nChunk ({i},{j},{k}):")
                        print(f"Padded bounds: x[{start_x}, {end_x}], y[{start_y}, {end_y}], z[{start_z}, {end_z}]")
                        print(f"Original bounds: x[{orig_start_x}, {orig_end_x}], y[{orig_start_y}, {orig_end_y}], z[{orig_start_z}, {orig_end_z}]")

                    mask = torch.logical_and(
                        torch.logical_and(
                            torch.logical_and(x.coords[:, 1] >= start_x, x.coords[:, 1] < end_x),
                            torch.logical_and(x.coords[:, 2] >= start_y, x.coords[:, 2] < end_y)
                        ),
                        torch.logical_and(x.coords[:, 3] >= start_z, x.coords[:, 3] < end_z)
                    )

                    if mask.sum() > 0:
                        # Get the coordinates and shift them to local space
                        coords = x.coords[mask].clone()
                        if verbose:
                            print(f"Before local shift - coords range: x[{coords[:, 1].min()}, {coords[:, 1].max()}], "
                                f"y[{coords[:, 2].min()}, {coords[:, 2].max()}], "
                                f"z[{coords[:, 3].min()}, {coords[:, 3].max()}]")
                        
                        # Shift to local coordinates
                        coords[:, 1:] = coords[:, 1:] - torch.tensor([start_x, start_y, start_z], 
                                                                    device=coords.device).view(1, 3)
                        if verbose:
                            print(f"After local shift - coords range: x[{coords[:, 1].min()}, {coords[:, 1].max()}], "
                                f"y[{coords[:, 2].min()}, {coords[:, 2].max()}], "
                                f"z[{coords[:, 3].min()}, {coords[:, 3].max()}]")

                        chunk_tensor = sp.SparseTensor(x.feats[mask], coords)
                        # Store the boundaries and offsets as metadata for later reconstruction
                        chunk_tensor.bounds = {
                            'padded': (start_x * upsample_ratio, end_x * upsample_ratio + (upsample_ratio - 1), start_y * upsample_ratio, end_y * upsample_ratio + (upsample_ratio - 1), start_z * upsample_ratio, end_z * upsample_ratio + (upsample_ratio - 1)),
                            'original': (orig_start_x * upsample_ratio, orig_end_x * upsample_ratio + (upsample_ratio - 1), orig_start_y * upsample_ratio, orig_end_y * upsample_ratio + (upsample_ratio - 1), orig_start_z * upsample_ratio, orig_end_z * upsample_ratio + (upsample_ratio - 1)),
                            'offsets': (start_x * upsample_ratio, start_y * upsample_ratio, start_z * upsample_ratio)  # Store offsets for reconstruction
                        }
                        out.append(chunk_tensor)

                    del mask
                    torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def upsamples(self, chunk: sp.SparseTensor, pruning=False): # Only for inferencing
        dtype = chunk.dtype
        for block in self.upsample:
            chunk, _ = block(chunk, pruning=pruning, training=False)
        chunk = chunk.type(dtype)
        chunk = self.out_layer(chunk)
        return chunk

    def forward(self, x: sp.SparseTensor, pruning=False):
        batch_size = x.shape[0]
        chunk_size = 8 # hard-coded to balance memory usage and reconstruction speed duing inference
        if x.coords.shape[0] < 150000:
            chunk_size = 4

        h = super().forward(x)
        chunks = self.split_for_meshing(h, chunk_size=chunk_size, verbose=False)
        all_coords, all_feats = [], []

        for chunk_idx, chunk in enumerate(chunks):
            try:
                chunk_result = self.upsamples(chunk, pruning=pruning)
            except Exception as e:
                print(f"Failed to process chunk {chunk_idx}: {e}")
                continue
            
            for b in range(batch_size):
                mask = torch.nonzero(chunk_result.coords[:, 0] == b).squeeze(-1)
                if mask.numel() > 0:
                    coords = chunk_result.coords[mask].clone()

                    # Restore global coordinates
                    offsets = torch.tensor(chunk.bounds['offsets'], 
                                            device=coords.device).view(1, 3)
                    coords[:, 1:] = coords[:, 1:] + offsets

                    # Filter points within original bounds
                    bounds = chunk.bounds['original']
                    within_bounds = torch.logical_and(
                        torch.logical_and(
                            torch.logical_and(
                                coords[:, 1] >= bounds[0],
                                coords[:, 1] < bounds[1]
                            ),
                            torch.logical_and(
                                coords[:, 2] >= bounds[2],
                                coords[:, 2] < bounds[3]
                            )
                        ),
                        torch.logical_and(
                            coords[:, 3] >= bounds[4],
                            coords[:, 3] < bounds[5]
                        )
                    )
                    
                    if within_bounds.any():
                        all_coords.append(coords[within_bounds])
                        all_feats.append(chunk_result.feats[mask][within_bounds])
                    
            torch.cuda.empty_cache()

        if len(all_coords) > 0:
            final_coords = torch.cat(all_coords)
            final_feats = torch.cat(all_feats)
            
            return self.to_representation(sp.SparseTensor(final_feats, final_coords))
        else:
            return self.to_representation(sp.SparseTensor(x.feats[:0], x.coords[:0]))
