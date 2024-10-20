# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#           2024 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch T5 model."""

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ['PromptEncoder']


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        """
        Encoder for quadratic Bézier curve prompts for decoder.

        Args:
            embed_dim: The prompts' embedding dimension. Needs to be divisible
            by 8.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, embed_dim // 8)))

    def forward(self, curves: torch.FloatTensor) -> torch.FloatTensor:
        """
        Embeds a quadratic Bézier curve.

        Args:
          curves: point coordinates of shape (B, 4, 2)

        Returns:
          Embeddings for the points with shape (B, E)
        """
        bs = curves.shape[0]

        coords = curves.clone()
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1).view(bs, -1)
