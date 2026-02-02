import math
from functools import partial
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
from einops import rearrange, repeat
import time
import numpy as np
import random
from collections import OrderedDict
import timm

from .loss_func import compute_proxy_loss, FourierLoss, ortho_proj_loss_fn_v2
from .cha_mae_vit import ChAMAEViT, trunc_normal_
from .model_utils import MAB

from .aggregators_old import *

# def get_aggregator_old(*args, **kwargs):

#     name = kwargs['aggregator']['name']

#     aggregator = None

#     if name == 'cls_probe':
#         aggregator = CLS_probe(use_mlp=False, **kwargs['aggregator'])
#     elif name == 'cls_token':
#         aggregator = CLS_token(**kwargs['aggregator'])
#     elif name == 'cls_mlp':
#         aggregator = CLS_probe(use_mlp=True, **kwargs['aggregator'])
#     elif name == 'mab':
#         aggregator = MAB_Aggregator(**kwargs['aggregator'])
#     elif name == 'mhca':
#         aggregator = MHCA_Aggregator(**kwargs['aggregator'])
#     elif name == 'mhca_cls':
#         aggregator = MHCA_CLS_Aggregator(**kwargs['aggregator'], use_mean_features=False)
#     elif name == 'mhca_cls_mean':
#         aggregator = MHCA_CLS_Aggregator(**kwargs['aggregator'], use_mean_features=True)
#     elif name == 'mhca_dual':
#         aggregator = MHCA_Dual_Aggregator(**kwargs['aggregator'], use_mean_features=True)
#     elif name == 'abmilp':
#         aggregator = AbMILP_Aggregator(**kwargs['aggregator'])
#     elif name == 'ep':
#         aggregator = EP_Aggregator(**kwargs['aggregator'])
#     elif name == 'mabx3':
#         aggregator = MABx3_Aggregator(**kwargs['aggregator'])
#     elif name == 'mhcax3':
#         aggregator = MHCAx3_Aggregator(**kwargs['aggregator'])
#     elif name == 'proto':
#         aggregator = Proto_Aggregator(**kwargs['aggregator'])
#     elif 'protobin' in name:
#         aggregator = Protobin_Aggregator(**kwargs['aggregator'])
#     elif name == 'simpool':
#         aggregator = SimPool_Aggregator(**kwargs['aggregator'])

#     return aggregator

def get_aggregator(*args, **kwargs):

    name = kwargs['aggregator']['name']

    aggregator = None

    if name == 'cls_probe':
        aggregator = CLS_probe(use_mlp=False, **kwargs['aggregator'])
    elif name == 'cls_token':
        aggregator = CLS_token(**kwargs['aggregator'])
    elif name == 'cls_mlp':
        aggregator = CLS_probe(use_mlp=True, **kwargs['aggregator'])
    elif name == 'mab':
        aggregator = MAB_Aggregator(**kwargs['aggregator'])
    elif name == 'mhca':
        aggregator = MHCA_Aggregator(**kwargs['aggregator'])
    elif name == 'abmilp':
        aggregator = AbMILP_Aggregator(**kwargs['aggregator'])
    elif name == 'ep':
        aggregator = EP_Aggregator(**kwargs['aggregator'])
    elif name == 'simpool':
        aggregator = SimPool_Aggregator(**kwargs['aggregator'])
    elif 'protobin' in name:
        aggregator = Protobin_Aggregator(**kwargs['aggregator'])

    return aggregator




class LinearModel(ChAMAEViT):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        """
        re-initialize all the ViT model params to simple nn.Identity.
        """

        self.cls_token = None
        self.pos_embed = None
        self.patch_embed = None
        self.blocks = None
        self.norm = None

        self.aggregator = get_aggregator(*args, **kwargs)

        self.apply(self._init_weights)

    def reduce_input_to_channel_mask(self, x, mask):

        B, C, N, D = x.shape
        x_flat = rearrange(x, 'B Cin HW Cout -> (B Cin) HW Cout')

        if mask is None:
            return x_flat, x_flat, None, torch.tensor([C for _ in range(B)], dtype=int, device=x.device)
        
        split_chunks = mask.sum(-1)
        mask_flat = mask.view(-1)
        valid_idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
        x_valid = x_flat[valid_idx]  

        return x_valid, x_flat, valid_idx, split_chunks

    def forward(
        self,
        x: Tensor,
        channel_ids_list: list[list[int]] | None = None,
        valid_channel_masks: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        return_all_tokens=False,
    ):

        x_origin = x.clone()  ## for reconstruction loss if any
        b, c, n, d = x.shape

        ### channel dropout, used in ChannelViT
        if self.training and self.training_sample in ["HCS", "HCS_SYMMETRIC"]:
            dropout_res = self.channel_dropout(
                x, channel_sample=self.training_sample, channel_ids_list=channel_ids_list, valid_channel_masks=valid_channel_masks
            )
            x, channel_ids_list, valid_channel_masks = dropout_res["x"], dropout_res["channel_ids_list"], dropout_res["channel_masks"]
            c = x.shape[1]

        if valid_channel_masks is None:
            out = self.aggregator(x)
        else:
            full_patch_masks = self.generate_patch_masks_from_channel_masks(valid_channel_masks)
            out = self.aggregator(x, mask=full_patch_masks, channel_mask=valid_channel_masks)
            # x, x_flat, valid_idx, split_chunks = self.reduce_input_to_channel_mask(x, valid_channel_masks)
            # chunks = torch.split(x, split_chunks.tolist(), dim=0)
            # unique_sizes, inverse_indices = torch.unique(split_chunks, return_inverse=True)
            # group_lists = [[] for _ in range(len(unique_sizes))]
            # for idx, chunk in zip(inverse_indices.tolist(), chunks):
            #     group_lists[idx].append(chunk)
            # group_tensors = [torch.stack(group, dim=0) for group in group_lists]
            # outputs = [self.aggregator(group_tensor) for group_tensor in group_tensors]
            # count_per_group = torch.zeros_like(unique_sizes)
            # final_output = torch.empty(len(split_chunks), x.shape[-1], device=x.device)

            # for i, group_idx in enumerate(inverse_indices):
            #     group_pos = count_per_group[group_idx].item()
            #     final_output[i] = outputs[group_idx][group_pos]
            #     count_per_group[group_idx] += 1

            # out = final_output


        ############## return predictions + losses if any
        if hasattr(self, "classifier_head"):
            ## for classification tasks, we return the logits
            out_logits = self.classifier_head(out)
            res = {"output": out_logits}
        else:
            res = {"output": out}

        ############## compute losses during training
        if self.training:
            ## compute proxy loss
            if self.use_proxy_loss:
                out_proxy = self.proxy_head(out)

                proxy_loss = compute_proxy_loss(
                    proxies=self.output_proxies,
                    img_emb=out_proxy,
                    gt_imgs=y,
                    scale=self.proxy_scale,
                )
                res["proxy_loss"] = proxy_loss
            else:
                res["proxy_loss"] = torch.tensor(0.0)

            ## compute cross entropy loss if any
            if self.use_cross_entropy_loss:
                ce_loss = self.compute_cross_entropy_loss(out_logits / self.cross_entropy_temperature, y)
                res["ce_loss"] = ce_loss
            else:
                res["ce_loss"] = torch.tensor(0.0)

            for k in ["mae_img_loss", "mae_fourier_loss", "mae_loss"]:
                    res[k] = torch.tensor(0.0)

            ## Diverse regularization losses
            res["diverse_channeltoken_loss"] = torch.tensor(0.0)
            res["diverse_patchtoken_loss"] = torch.tensor(0.0)

            ## final loss
            res["loss"] = (
                self.proxy_loss_lambda * res["proxy_loss"]
                + self.mae_lambda * res["mae_loss"]
                + self.diverse_patch_token_weight * res["diverse_patchtoken_loss"]
                + self.diverse_channel_token_weight * res["diverse_channeltoken_loss"]
                + self.cross_entropy_lambda * res["ce_loss"]
            )
        return res


def get_linear_model(model_size: str, patch_size=16, **kwargs):
    if model_size == "tiny":
        model = LinearModel(
            patch_size=patch_size,
            embed_dim=192,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    elif model_size == "small":
        model = LinearModel(
            patch_size=patch_size,
            embed_dim=384,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    elif model_size == "base":
        model = LinearModel(
            patch_size=patch_size,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model name: {model_size}")
    return model
