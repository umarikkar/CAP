import math
import warnings
import torch
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
from torch import nn
from einops import repeat

from typing import Optional, Tuple, Any

class Attentive(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        num_queries: int = 1,
        use_batch_norm: bool = False,
        qkv_bias: bool = False,
        linear_bias: bool = False,
        average_pool: bool = True,
    ):
        super().__init__()
        if dim == 1024:
            num_heads = 16
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool

        self.dim = dim
        self.out_features = out_features

        # self.k_iso = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v_iso = nn.Linear(dim, dim, bias=qkv_bias)

        # self.k_red = nn.Linear(out_features, dim, bias=qkv_bias)
        # self.v_red = nn.Linear(out_features, dim, bias=qkv_bias)

        self.k =nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

        # self.lin_iso = nn.Linear(dim, dim, bias=linear_bias)
        # self.lin_inc = nn.Linear(dim, out_features, bias=linear_bias)

        self.linear = nn.Linear(dim, out_features, bias=linear_bias)


        self.bn = (
            nn.BatchNorm1d(dim, affine=False, eps=1e-6)
            if use_batch_norm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cls_token=None, average_pool=None, attn_mask=None, mask=None, red=False, inc=False) -> torch.Tensor:

        B, N, C = x.shape

        # if red:
        #     C = self.dim
        #     fn_k = self.k_red
        #     fn_v = self.v_red
        # else:
        #     fn_k = self.k_iso
        #     fn_v = self.v_iso

        # this is just to get the None and wrong arg name out of the way. 
        attn_mask = attn_mask or mask

        average_pool=self.average_pool if average_pool is None else average_pool

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)

        if cls_token == None:
            cls_token = self.cls_token.expand(B, -1, -1)
            num_queries=self.num_queries
        else:
            cls_token = cls_token
            num_queries=cls_token.shape[1]

        q = cls_token.reshape(
            B, num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)

        fn_k = self.k
        fn_v = self.v

        k = (
            fn_k(x).reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            fn_v(x).reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x_cls = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x_cls = x_cls.transpose(1, 2).reshape(B, num_queries, C)
        x_cls = x_cls.mean(dim=1) if average_pool else x_cls

        # if inc:
        #     x_cls = self.lin_inc(x_cls)
        # else:
        #     x_cls = self.lin_iso(x_cls)

        x_cls = self.linear(x_cls)

        return x_cls
    
class Prototypical_multi_binarized_simple(nn.Module):
    def __init__(self, dim, num_prototypes, num_classes, topk_k=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes
        self.num_prototypes_total = num_prototypes * num_classes
        self.topk_k = topk_k

        # real‑valued weights: binarised on‑the‑fly
        # 1x1 convolutional kernels 
        self.prototype_vectors = nn.Parameter(torch.randn(
            self.num_prototypes_total, dim, 1, 1) * 0.02)

        self.linear = nn.Linear(self.num_prototypes_total, num_classes)

    @staticmethod
    def _binarise(x): 
        # each weight becomes a single bit (positive 1 or negative 0)
        # simple XOR computation instead of floating-point multiply
        # built-in regulariser: prevents from memorising tiny numerical values in the training dta
        # forward-value tensor (no gradients)
        signed_val = (x >= 0).float() * 2.0 - 1.0 # +1/-1

        # gradient-only tensor (value=0, grad=∂L/∂x)
        grad_pass = x - x.detach()

        # straight-through estimator: makes binarized snaps differentiable
        # is the identity whn computing gradients, keeps using snapped +-1 in forward, otherwise the gradient would be flat
        return signed_val + grad_pass 
        #return (x >= 0).float() * 2.0 - 1.0 + (x - x.detach())

    def forward(self, x):
        if x.dim() == 2: #for cls-input
            x = x[:, :, None, None]
        # binarise prototypes for similarity
        # during forward pass: binarized tensor of ±1
        # during backward gradients flow to the master copy tensor
        protos_bin = self._binarise(self.prototype_vectors)

        # standard cosine similarity (activation)
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(protos_bin, dim=1)
        act = F.conv2d(x_norm, p_norm)              # (B, P, H, W)

        # pooling & classification as before
        B, P, _, _ = act.shape
        act = act.view(B, P, -1)
        k = self.topk_k if self.training else 1

        # top-k pooling: taking the mean of the k strongest hits
        pooled = act.topk(k, dim=-1).values.mean(-1)
        return self.linear(pooled)

class Prototypical_multi(nn.Module):
    def __init__(
        self,
        dim: int,
        num_prototypes: int,
        num_classes: int,
        topk_k: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes
        self.num_prototypes_total = self.num_prototypes_per_class * self.num_classes
        self.topk_k = topk_k
        self.input_vector_length = 64
        self.n_eps_channels = 2
        self.epsilon_val = 1e-4
        
        # Prototype vectors
        self.prototype_vectors = nn.Parameter(
            torch.randn(self.num_prototypes_total, dim, 1, 1) * 0.02
        )
               
        # Classification layer
        self.linear = nn.Linear(self.num_prototypes_total, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def cos_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between input features and prototype vectors."""
        normalizing_factor = self.prototype_vectors.shape[-1] ** 0.5
        
        # Add epsilon channels to input
        epsilon_channel_x = torch.full(
            (x.shape[0], self.n_eps_channels, x.shape[2], x.shape[3]),
            self.epsilon_val,
            device=x.device,
            requires_grad=False,
        )
        x = torch.cat((x, epsilon_channel_x), dim=1)
        
        # Normalize input
        x_length = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.epsilon_val)
        x_normalized = (self.input_vector_length * x / x_length) / normalizing_factor
        
        # Add epsilon channels to prototypes
        epsilon_channel_p = torch.full(
            (self.prototype_vectors.shape[0], self.n_eps_channels, 1, 1),
            self.epsilon_val,
            device=self.prototype_vectors.device,
            requires_grad=False,
        )
        prototypes = torch.cat((self.prototype_vectors, epsilon_channel_p), dim=1)
        
        # Normalize prototypes
        prototype_length = torch.sqrt(
            torch.sum(prototypes**2, dim=1, keepdim=True) + self.epsilon_val
        )
        normalized_prototypes = prototypes / (prototype_length + self.epsilon_val)
        normalized_prototypes /= normalizing_factor
        
        # Compute cosine similarity via convolution
        activations = F.conv2d(x_normalized, normalized_prototypes)
        activations = activations / (self.input_vector_length * 1.01)
        
        return F.relu(activations)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply add-on layers
        #x = self.add_on_layers(x)
        
        # Compute prototype activations
        activations = self.cos_activation(x)
        
        # Global pooling with top-k
        batch_size, num_prototypes = activations.shape[:2]
        activations = activations.view(batch_size, num_prototypes, -1)
        
        # Use top-k pooling during training, max pooling during evaluation
        topk_k = self.topk_k if self.training else 1
        topk_activations, _ = torch.topk(activations, topk_k, dim=-1)
        pooled_activations = torch.mean(topk_activations, dim=-1)
        
        # Classification
        logits = self.linear(pooled_activations)
        return logits


class Prototypical_multi_binarized_simple(nn.Module):
    def __init__(self, dim, num_prototypes, num_classes, topk_k=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes
        self.num_prototypes_total = num_prototypes * num_classes
        self.topk_k = topk_k

        # real‑valued weights: binarised on‑the‑fly
        # 1x1 convolutional kernels 
        self.prototype_vectors = nn.Parameter(torch.randn(
            self.num_prototypes_total, dim, 1, 1) * 0.02)

        self.linear = nn.Linear(self.num_prototypes_total, num_classes)

    @staticmethod
    def _binarise(x): 
        # each weight becomes a single bit (positive 1 or negative 0)
        # simple XOR computation instead of floating-point multiply
        # built-in regulariser: prevents from memorising tiny numerical values in the training dta
        # forward-value tensor (no gradients)
        signed_val = (x >= 0).float() * 2.0 - 1.0 # +1/-1

        # gradient-only tensor (value=0, grad=∂L/∂x)
        grad_pass = x - x.detach()

        # straight-through estimator: makes binarized snaps differentiable
        # is the identity whn computing gradients, keeps using snapped +-1 in forward, otherwise the gradient would be flat
        return signed_val + grad_pass 
        #return (x >= 0).float() * 2.0 - 1.0 + (x - x.detach())

    def forward(self, x, mask=None, **kwargs):
        if x.dim() == 2: #for cls-input
            x = x[:, :, None, None]
        elif x.dim() == 3:
            # we get the input as B, N C, need to rearrange to B C N 1
            x = rearrange(x, 'b n c -> b c n 1') 
        # binarise prototypes for similarity
        # during forward pass: binarized tensor of ±1
        # during backward gradients flow to the master copy tensor
        protos_bin = self._binarise(self.prototype_vectors)

        # standard cosine similarity (activation)
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(protos_bin, dim=1)

        act = F.conv2d(x_norm, p_norm)              # (B, P, H, W)

        # pooling & classification as before
        B, P, _, _ = act.shape
        act = act.view(B, P, -1)
        k = self.topk_k if self.training else 1

        if mask is not None:
            mask_flat = mask.view(B, 1, -1)  
            act = act.masked_fill(~mask_flat, float('-inf')) 

        # top-k pooling: taking the mean of the k strongest hits
        pooled = act.topk(k, dim=-1).values.mean(-1)
        return self.linear(pooled)

    
class EfficientProbing(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 32,
        d_out: int = 1
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.d_out = d_out
        self.num_queries = num_queries
        
        self.v = nn.Linear(dim, dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        
    def forward(self, x: torch.Tensor, cls=None, mask=None, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        C_prime = C // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = (x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        q = q * self.scale
        v = (self.v(x).reshape(B, N, self.num_queries, C // (self.d_out * self.num_queries)).permute(0, 2, 1, 3))

        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            attn.masked_fill_(mask.logical_not(), float("-inf"))

        attn = attn.softmax(dim=-1)

        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        x_cls = x_cls.view(B, C_prime)
        
        return x_cls




class MAB(nn.Module):
    ### https://proceedings.mlr.press/v97/lee19d/lee19d.pdf
    ### adapted from https://github.com/juho-lee/set_transformer/blob/master/modules.py
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln=False, attn_drop=0.0, proj_drop=0.0, scale=None, init_query=True, squeeze_out=True):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.scale = scale if scale is not None else math.sqrt(self.dim_V)
        self.squeeze_out = squeeze_out

        if init_query:
            self.query = nn.Parameter(torch.randn(1, dim_Q))
            trunc_normal_(self.query, std=0.02)

    def forward(self, X: Tensor, Q=None, squeeze_out=True, mask=None):
        """
        Q: shape (t,d)
        X: shape (b,l,d')
        Output: shape (b,t,d')
        """
        squeeze_out = self.squeeze_out if squeeze_out is None else squeeze_out

        Q = self.query if Q is None else Q

        if len(Q.size()) != len(X.size()):
            Q = repeat(Q, "t d -> b t d", b=X.shape[0])

        Q = self.fc_q(Q)

        K, V = self.fc_k(X), self.fc_v(X)

        dim_split = self.dim_V // self.num_heads

        Q_ = rearrange(Q, 'b n (h d) -> b (n h) d', d=dim_split)
        K_ = rearrange(K, 'b n (h d) -> b (n h) d', d=dim_split)
        V_ = rearrange(V, 'b n (h d) -> b (n h) d', d=dim_split)

        A = Q_.bmm(K_.transpose(1, 2)) / self.scale

        if mask is not None:
            mask = repeat(mask, 'b 1 1 n -> b 1 (n h)', h=self.num_heads)
            A.masked_fill_(mask.logical_not(), float("-inf"))

        A = torch.softmax(A, 2)
        A = self.attn_drop(A)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(1), 1), 2)

        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + self.proj_drop(F.relu(self.fc_o(O)))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)

        if squeeze_out:
            O = O.squeeze(1)

        return O
    

class SimPool(nn.Module):
    def __init__(self, dim, out_features, qkv_bias=False, qk_scale=None):
        super().__init__()
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5
        
        self.norm_patches = nn.LayerNorm(dim, eps=1e-6)
        
        # Final classification layer
        self.linear = nn.Linear(dim, out_features)

    def prepare_input(self, x):
        if len(x.shape) == 4:  # CNN format: (B, d, H, W)
            B, d, H, W = x.shape
            x = x.reshape(B, d, H*W).permute(0, 2, 1)  # (B, H*W, d)
        return x

    def forward(self, x, return_attn=False, mask=None):
        # Prepare input - convert to (B, N, D) format
        x = self.prepare_input(x)
        
        # Apply layer normalization
        x_norm = self.norm_patches(x)
        
        # Data-dependent query: global average of input features
        q = x_norm.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # Compute attention (no key/value projections, WV = identity)
        attn_logits = torch.matmul(q, x_norm.transpose(-2, -1)) * self.scale  # (B, 1, N)

        if mask is not None: 
            mask = rearrange(mask, 'b 1 1 n -> b 1 n')
            attn_logits.masked_fill_(mask.logical_not(), float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)  # (B, 1, N)
        
        # Aggregate features (WV = identity, so values are just x_norm)
        aggregated = torch.matmul(attn, x_norm).squeeze(1)  # (B, D)
        
        # Apply classification layer
        out = self.linear(aggregated)
        
        if return_attn:
            return out, attn
        else:
            return out


def maybe_flatten_images(img: torch.Tensor, patch_size: int, channel_agnostic: bool = False) -> torch.Tensor:
    """
    Flattens 2D images into tokens with the same pixel values

    Parameters
    ----------
    img : input image tensor (N, C, H, W);  if 3D, just return (do nothing)

    Returns
    -------
    flattened_img: flattened image tensor (N, L, patch_size**2 * C)
    """
    if len(img.shape) == 3:
        return img

    if (img.shape[2] != img.shape[3]) or (img.shape[2] % patch_size != 0):
        raise ValueError("image H must equal image W and be divisible by patch_size")
    in_chans = img.shape[1]

    h = w = int(img.shape[2] // patch_size)
    x = img.reshape(shape=(img.shape[0], in_chans, h, patch_size, w, patch_size))

    if channel_agnostic:
        # x = torch.permute(x, (0, 1, 2, 4, 3, 5))  # NCHPWQ -> NCHWPQ
        # x = x.reshape(shape=(img.shape[0], in_chans * h * w, int(patch_size**2)))
        x = rearrange(x, "b c h p1 w p2 -> b (c h w) (p1 p2)", p1=patch_size, p2=patch_size)
    else:
        x = torch.permute(x, (0, 2, 4, 3, 5, 1))  # NCHPWQ -> NHWPQC
        x = x.reshape(shape=(img.shape[0], h * w, int(patch_size**2 * in_chans)))
    return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_2d_sincos_pos_embed(dim: int, grid_size: int, cls_token: bool = True) -> torch.Tensor:
    """
    Build 2D sine-cosine positional embeddings as a fixed (non-trainable) buffer.
    Returns [1, (1 + grid_size*grid_size), dim] if cls_token else [1, grid_size*grid_size, dim].
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos."
    # Create grid of positions
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=0)  # [2, Gh, Gw]
    grid = grid.reshape(2, 1, grid_size * grid_size)  # [2, 1, N]

    # Frequencies
    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)
    omega = 1.0 / (10000 ** omega)  # [dim/4]

    pos_h = grid[0].transpose(0, 1)  # [1, N]
    pos_w = grid[1].transpose(0, 1)  # [1, N]

    out = []
    for pos in (pos_h, pos_w):
        pos = pos * omega  # [1, N, dim/4] via broadcasting later
        pos = pos.unsqueeze(-1)  # [1, N, dim/4, 1]
        # sin and cos on last axis
        sin = torch.sin(pos.squeeze(-1))  # [1, N, dim/4]
        cos = torch.cos(pos.squeeze(-1))  # [1, N, dim/4]
        out += [sin, cos]

    # concat along last dim -> [1, N, dim]
    pe = torch.cat(out, dim=-1)
    if cls_token:
        pe = torch.cat([torch.zeros(1, 1, dim, dtype=pe.dtype), pe], dim=1)
    return pe  # [1, N(+1), dim]


class SingleHeadSelfAttention(nn.Module):
    """
    Simple single-head self-attention (batch_first) used only as an optional
    pre-processing step in SA-AbMILP. Keeps dimensionality (D -> D).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


class AbMILP(nn.Module):
    """
    Selective Aggregation via Attention-based Multiple Instance Learning Pooling (AbMILP).

    - Matches the paper’s Selective Aggregation when self_attention_apply_to="none".
    - SA-AbMILP when self_attention_apply_to in {"map","both"} (a light single-head self-attn pre-step).
    - Scores each token with a tiny predictor t: R^D -> R, softmax over tokens, weighted sum -> global vector.
    - No learnable queries and W_V effectively identity (pool original or self-attended tokens).

    Args:
        dim:           token or channel dimension D.
        out_features:  number of output classes.
        self_attention_apply_to: "none" | "map" | "both"
            - "none": scorer sees raw tokens, pool raw tokens.
            - "map": scorer sees self-attended tokens, pool raw tokens.
            - "both": scorer sees self-attended tokens, pool self-attended tokens.
        activation:    "tanh" or "relu" for the scorer MLP.
        depth:         depth of the scorer MLP (>=1). depth=1 = single Linear(D->1).
        cond:          "none" or "pe" to add fixed 2D sin-cos positional embeddings to scorer input.
        content:       "all" (keep first token if present) or "patch" (drop first token).
        num_patches:   required if cond="pe". Should equal H*W if you pass maps, or N_patches if tokens.

    Input:
        x:  either [B, D, H, W] feature maps or [B, N, D] token sequences.
        cls: optional [B, D] or [B, 1, D] cls token to prepend when content="all".

    Returns:
        logits:   [B, out_features]
        attn_map: [B, N_used, 1] selection weights over tokens that were pooled
                  (N_used excludes cls when content="patch").
    """
    def __init__(
        self,
        dim: int,
        out_features: int,
        self_attention_apply_to: str = "none",  # "none" | "map" | "both"
        activation: str = "tanh",               # "tanh" | "relu"
        depth: int = 2,
        cond: str = "none",                     # "none" | "pe"
        content: str = "all",                   # "all" | "patch"
        num_patches: Optional[int] = None,
    ):
        super().__init__()
        assert self_attention_apply_to in {"none", "map", "both"}
        assert activation in {"tanh", "relu"}
        assert content in {"all", "patch"}
        if cond == "pe":
            assert num_patches is not None, "num_patches must be provided when cond='pe'"

        self.dim = dim
        self.out_features = out_features
        self.self_attention_apply_to = self_attention_apply_to
        self.cond = cond
        self.content = content

        # Optional pre self-attention
        self.self_attn = (
            SingleHeadSelfAttention(dim)
            if self.self_attention_apply_to != "none"
            else nn.Identity()
        )

        # Optional fixed positional encoding for the scorer input
        if self.cond == "pe":
            grid = int(round(num_patches ** 0.5))
            assert grid * grid == num_patches, "num_patches must be a perfect square for pe"
            pe = get_2d_sincos_pos_embed(dim, grid, cls_token=(content != "patch"))
            self.register_buffer("pos_embed", pe, persistent=False)  # [1, N, D]
        else:
            self.pos_embed = None

        # Tiny scorer t(z) -> scalar
        layers = []
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
        layers.append(nn.Linear(dim, 1))
        self.attention_predictor = nn.Sequential(*layers)

        # Final linear head on the pooled D-dim vector
        self.head = nn.Linear(dim, out_features, bias=False)

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        # [B, D, H, W] -> [B, N, D] or pass-through if already [B, N, D]
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        elif x.dim() != 3:
            raise ValueError("x must be [B, D, H, W] or [B, N, D]")
        return x

    def _maybe_concat_cls(self, x: torch.Tensor, cls: Optional[torch.Tensor]) -> torch.Tensor:
        if cls is None:
            return x
        if cls.dim() == 2:
            cls = cls.unsqueeze(1)  # [B, 1, D]
        elif cls.dim() != 3 or cls.size(1) != 1:
            raise ValueError("cls must be [B, D] or [B, 1, D]")
        return torch.cat([cls, x], dim=1)  # [B, 1+N, D]

    def forward(self, x: torch.Tensor, cls: Optional[torch.Tensor] = None, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) ensure token layout
        x = self._to_tokens(x)  # [B, N, D]

        # 2) optionally prepend cls token if user provided it and we are keeping all tokens
        if cls is not None and self.content == "all":
            x = self._maybe_concat_cls(x, cls)  # [B, 1+N, D]

        # 3) optionally drop the first token to keep patches only
        if self.content == "patch" and x.size(1) >= 1:
            x = x[:, 1:]  # [B, N_patches, D]

        # 4) optional single-head self-attention pre-step
        x_attn = self.self_attn(x) if self.self_attention_apply_to != "none" else x

        # 5) choose what the scorer sees
        scorer_in = x_attn if self.self_attention_apply_to in {"map", "both"} else x

        # 6) optional positional conditioning (fixed, non-trainable)
        if self.pos_embed is not None:
            if self.pos_embed.size(1) != scorer_in.size(1):
                raise ValueError(
                    f"pos_embed length {self.pos_embed.size(1)} != token length {scorer_in.size(1)}. "
                    f"Check num_patches and content."
                )
            scorer_in = scorer_in + self.pos_embed  # broadcast [1, N, D]

        # 7) predict token scores and softmax over tokens
        attn_map = self.attention_predictor(scorer_in)  # [B, N, 1]
        attn_map = F.softmax(attn_map, dim=1)

        # 8) decide which stream to pool
        pooled_source = x_attn if self.self_attention_apply_to == "both" else x  # [B, N, D]
        z = (pooled_source * attn_map).sum(dim=1)  # [B, D]

        # 9) classifier
        logits = self.head(z)  # [B, C]
        return logits#, attn_map