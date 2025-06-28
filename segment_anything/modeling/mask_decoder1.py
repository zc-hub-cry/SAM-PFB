# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#mask_decoder.py
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
import numpy as np
import math
import gcn.path_config as dirs
from torch import Tensor
from typing import Optional, Callable, Union, Tuple, Any
from einops.layers.torch import Rearrange
from mmengine.model import constant_init, kaiming_init


class MemoryEnhancedMultiScaleBoundaryEnhancementModule(nn.Module):
    def __init__(self, in_channels, num_masks, memory_size=7, activation=nn.ReLU):
        super(MemoryEnhancedMultiScaleBoundaryEnhancementModule, self).__init__()
        self.memory_size = memory_size
        self.activation = activation()
        self.num_masks = num_masks

        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            activation(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            activation()
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False),
            activation(),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False),
            activation()
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False),
            activation(),
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False),
            activation()
        )

        self.fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.boundary_map = nn.Conv2d(in_channels, num_masks, kernel_size=1, bias=False)

        self.register_buffer('memory', torch.zeros(memory_size, in_channels, 1, 1))
        self.memory_ptr = 0

    def forward(self, upscaled_embedding, initial_masks):
        batch_size, _, H, W = upscaled_embedding.size()

        boundary1 = self.scale1(upscaled_embedding)
        boundary2 = self.scale2(upscaled_embedding)
        boundary3 = self.scale3(upscaled_embedding)

        fused_boundary = torch.cat([boundary1, boundary2, boundary3], dim=1)
        fused_boundary = self.fusion(fused_boundary)
        boundary_attention = self.sigmoid(fused_boundary)  # [batch, in_channels, H, W]

        boundary_attention = self.boundary_map(boundary_attention)  # [batch, num_masks, H, W]

        masks = initial_masks * boundary_attention  # [batch, num_masks, H, W]

        if self.memory.size(0) > 0:
            expanded_memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1,-1, -1)  # [batch, memory_size, C, 1, 1]
            expanded_memory = expanded_memory.view(batch_size, self.memory_size, -1)  # [batch, memory_size, C]

            memory_feature = expanded_memory.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # [batch, C, 1, 1]
            memory_map = self.boundary_map(memory_feature)  # [batch, num_masks, 1, 1]
            masks = masks + memory_map  # [batch, num_masks, H, W]

        with torch.no_grad():
            current_boundary = fused_boundary.mean(dim=[2, 3], keepdim=True)  # [batch, C, 1, 1]
            num_insert = min(batch_size, self.memory_size - self.memory_ptr)
            if num_insert > 0:
                self.memory[self.memory_ptr:self.memory_ptr + num_insert] = current_boundary[:num_insert]
                self.memory_ptr = (self.memory_ptr + num_insert) % self.memory_size

        return masks


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            memory_size: int = 7 
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.me_ms_bcem = MemoryEnhancedMultiScaleBoundaryEnhancementModule(
            in_channels=transformer_dim // 8,  
            num_masks=self.num_mask_tokens, 
            memory_size=memory_size
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            mode='train',
            gt=None,
            img_size=256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred, attn_out, up_embed = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            mode=mode,
            gt=gt,
            img_size=img_size
        )
        return masks, iou_pred, attn_out, up_embed

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            mode='train',
            gt=None,
            img_size=256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        msk_feat = torch.matmul(mask_tokens_out, src.transpose(1, 2))

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [b, c, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [b, num_masks, h, w]

        masks = self.me_ms_bcem(upscaled_embedding, masks)  

        return masks, iou_pred, msk_feat, upscaled_embedding

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder2(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer2: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            memory_size: int = 7 
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer2 = transformer2

        self.num_multimask_outputs = num_multimask_outputs

        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.skip_connect = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 8),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 16),
            activation(),
        )
        self.output_hypernetworks_mlps2 = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 16, 5)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        self.med_sel = nn.Sequential(
            nn.Linear(self.num_mask_tokens, self.num_mask_tokens),
            nn.ReLU()
        )

        # self.self_attn = Attention(
        #     4096, num_heads=8
        # )
        self.self_attn = MultiHeadDifferentialAttention(
            d_model=1024, num_heads=8, lambda_init=0.8
        )
        # self.self_attn2 = Attention(
        #     4096, num_heads=8
        # )
        self.self_attn2 = MultiHeadDifferentialAttention(
            d_model=1024, num_heads=8, lambda_init=0.8
        )
        self.norm1 = nn.LayerNorm(4096)
        self.norm2 = nn.LayerNorm(4096)
        self.mlp = MLPBlock(4096, 8192)
        self.med_sel = nn.Sequential(
            nn.Linear(self.num_mask_tokens, 1),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)


        self.me_ms_bcem2 = MemoryEnhancedMultiScaleBoundaryEnhancementModule(
            in_channels=transformer_dim // 16, 
            num_masks=self.num_mask_tokens,  
            memory_size=memory_size
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            mask_feat: torch.Tensor,
            gt=None,
            mode='test',
            msk_feat=None,
            up_embed=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred, attn_out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            mask_feat=mask_feat,
            gt=gt,
            mode=mode,
            msk_feat=msk_feat,
            up_embed=up_embed
        )
        return masks, iou_pred, attn_out

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            mask_feat: torch.Tensor,
            gt=None,
            mode='test',
            msk_feat=None,
            up_embed=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 处理 mask_feat
        if len(mask_feat.shape) == 3:
            mask_feat = mask_feat.unsqueeze(0)
        mask_feat = self.softmax(mask_feat).flatten(start_dim=2)
        flag_resize = 0
        if msk_feat.shape[-1] != 4096:
            msk_feat_no_grad = msk_feat.detach()
            msk_feat_no_grad = msk_feat_no_grad.resize_(msk_feat_no_grad.shape[-3], msk_feat_no_grad.shape[-2], 1024)
            msk_feat = msk_feat_no_grad
            flag_resize = 1

        if gt is not None:
            gt_feat = gt.clone()

 
        msk_feat = self.self_attn(q=msk_feat, k=msk_feat, v=msk_feat) 
        msk_feat = self.norm1(msk_feat)
        msk_feat = self.self_attn2(q=msk_feat, k=msk_feat, v=msk_feat)
        msk_feat = msk_feat.clone() + self.mlp(msk_feat)
        msk_feat = self.norm2(msk_feat)
        msk_feat = self.med_sel(msk_feat.transpose(-2, -1))

        if flag_resize == 1:
            msk_feat = msk_feat.reshape(msk_feat.shape[-3], 1024, msk_feat.shape[-1])
            flag_resize = 0
        msk_feat = msk_feat.transpose(-1, -2).view(b, -1, h, w)
        msk_feat = self.softmax(msk_feat)


        src = src.clone() + torch.mul(src, msk_feat)
        hs, src, attn_out = self.transformer2(src, pos_src, tokens, mask_feat)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 0: (1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        upscaled_embedding = self.skip_connect(torch.cat((upscaled_embedding, up_embed), dim=1))
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps2[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [b, num_masks, h, w]

        masks = self.me_ms_bcem2(upscaled_embedding, masks)  

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred, attn_out


class MLP2(nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))



class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """

    def __init__(self, d, eps=1e-5):
        """
        Args:
            d (int): Dimension of the input features.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.
    """

    def __init__(self, d_model):
        """
        Args:
            d_model (int): Dimension of the input features.
        """
        super().__init__()
        # Intermediate projection layers
        # Typically, SwiGLU splits the computation into two parts
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        """
        Forward pass for SwiGLU.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying SwiGLU.
        """
        # Apply the gates
        g = F.silu(self.WG(x))  # Activation part
        z = self.W1(x)  # Linear part
        # Element-wise multiplication and projection
        return self.W2(g * z)


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    Incorporates a causal mask to ensure autoregressive behavior.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for queries, keys, and values
        # Project to 2 * d_head per head for differential attention
        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # Learnable parameters for lambda reparameterization
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.d_head))

        self.lambda_init = lambda_init

        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5  # Epsilon for numerical stability

        # Initialize weights (optional but recommended)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters for improved training stability.
        """
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying differential attention.
        """
        batch, N, d_model = X.shape

        # Project inputs to queries, keys, and values
        Q = self.W_q(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        K = self.W_k(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        V = self.W_v(X)  # Shape: (batch, N, 2 * num_heads * d_head)

        # Reshape and permute for multi-head attention
        # New shape: (batch, num_heads, sequence_length, 2 * d_head)
        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)

        # Compute lambda using reparameterization
        # lambda_val = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        # Compute dot products for each head
        # Shape of lambda_val: (num_heads,)
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()  # (num_heads,)
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()  # (num_heads,)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)

        # Expand lambda_val to match attention dimensions
        # Shape: (batch, num_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # ------------------- Causal Mask Implementation ------------------- #
        # Create a causal mask to prevent attention to future tokens
        # Shape of mask: (1, 1, N, N)
        mask = torch.tril(torch.ones((N, N), device=X.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        # Replace 1s with 0.0 and 0s with -inf
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        # -------------------------------------------------------------------- #

        # Compute attention scores
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)

        # Apply the causal mask
        A1 = A1 + mask  # Mask out future positions
        A2 = A2 + mask  # Mask out future positions

        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)  # (batch, num_heads, N, N)
        attention2 = F.softmax(A2, dim=-1)  # (batch, num_heads, N, N)
        attention = attention1 - lambda_val * attention2  # (batch, num_heads, N, N)

        # Apply attention weights to values
        O = torch.matmul(attention, V)  # (batch, num_heads, N, 2 * d_head)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)  # (batch*num_heads, N, 2*d_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (batch*num_heads, N, 1)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale  # (batch*num_heads, N, 2*d_head)

        # Reshape back to (batch, num_heads, N, 2 * d_head)
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        # Scale the normalized output
        O_normalized = O_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch, N, num_heads * 2 * d_head)
        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)

        # Final linear projection
        out = self.W_o(O_concat)  # (batch, N, d_model)

        return out


class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda in Differential Attention.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x):
        """
        Forward pass for a single transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after processing through the layer.
        """
        # Apply Multi-Head Differential Attention with residual connection
        y = self.attn(self.norm1(x)) + x
        # Apply SwiGLU Feed-Forward Network with residual connection
        z = self.ff(self.norm2(y)) + y
        return z

