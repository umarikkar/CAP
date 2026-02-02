import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
from .model_utils import MAB, trunc_normal_, AbMILP, Attentive, EfficientProbing, Prototypical_multi, SimPool, Prototypical_multi_binarized_simple

import torch.nn.functional as F


class ABMIL(nn.Module):
    def __init__(self, dim=768, temp=0.07, projections=True, proj_dropout=0.0):
        super().__init__()
        self.dim = dim
        self.temp = temp

        print('TEMP IS:', self.temp)

        self.attention_w = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

        self.projections = projections
        if self.projections:
            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.act1 = nn.GELU()

        self.drop1 = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

    def forward(self, x, mask=None, temp=None):
        """
        x: [B, C, D]
        mask: [B, C]  (True = valid, False = invalid)
        """
        tempp = self.temp if temp is None else temp
        B, C, D = x.shape

        if self.projections:
            x = self.act1(self.proj1(x))

        x = self.drop1(x)

        # Compute attention scores
        A = self.attention_w(x)  # [B, C, 1]
        A = A.permute(0, 2, 1)   # [B, 1, C]

        # Apply mask before softmax
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, C]
            A = A.masked_fill(~mask, float('-inf'))

        # Softmax only over valid positions
        A = torch.softmax(A / tempp, dim=-1)

        # Multiply and aggregate only over valid channels
        Z = torch.matmul(A, x)  # [B, 1, D]
        Z = Z.squeeze(1)        # [B, D]

        if self.projections:
            Z = self.proj2(Z)

        return Z


class CLS_token(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()


    def forward(self, x, *args, **kwargs):

        """
        extract CLS token.
        """

        x = x[:, 0]

        return x



class CLS_probe(nn.Module):

    def __init__(self, embed_dim=384, chan_setting='joint_joint', use_mlp=False, *args, **kwargs):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim*2), 
                                 nn.GELU(), 
                                 nn.Linear(embed_dim*2, embed_dim)) if use_mlp else nn.Identity()
        self.chan_setting = chan_setting
        

    def forward(self, x):

        """
        extract CLS token.
        """
        b, c, n, d = x.shape

        x = x[:, :, 0].detach()

        x = self.mlp(x)

        if x.size(1)==1:
            x = rearrange(x, 'b n d -> b (n d)')

        return x


class BaseAggregator(nn.Module):
    def __init__(self, embed_dim, chan_setting='joint_joint', num_heads=6, *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.chan_setting = chan_setting

        [self.inp_type, self.agg_type] = chan_setting.split('_')

        # if chan_setting in ['sep_sep', 'joint_sep']:
        #     # we are doing two forward passes, so we need a dimension reducer.
        #     self.dim_reduction = 

        # fn will be assigned in subclasses

        if 'dual' in self.agg_type:
            self.embed_dim = embed_dim // 2
            self.dim_reducer = nn.Sequential(nn.Linear(embed_dim, embed_dim//2))

            self.norm = nn.LayerNorm(embed_dim)

            if self.agg_type in ['dualv2', 'dualv3', 'dualv4']:

                # with MLP of r=1
                r=1
                # self.alphas = nn.Parameter(0.5 * torch.ones(2))
                if self.agg_type == 'dualv2':
                    self.mixer = nn.Sequential(nn.Linear(embed_dim//2, r*embed_dim), nn.GELU(), 
                                            nn.Linear(r*embed_dim, embed_dim))
                    self.alphas = nn.Parameter(0.5 * torch.ones(2))
                elif self.agg_type == 'dualv3':
                    self.embed_dim = embed_dim //2
                    self.mixer = nn.Linear(embed_dim//2, embed_dim)
                    self.alphas = nn.Parameter(0.5 * torch.ones(2))
                elif self.agg_type == 'dualv4':
                    self.mixer = nn.Linear(embed_dim//2, embed_dim)

                self.norm0 = nn.LayerNorm(embed_dim//2)
            elif self.agg_type == 'dualx3':
                self.mixer = nn.Linear(embed_dim // 2, embed_dim)
            elif self.agg_type == 'dualx4':
                r=1
                self.mixer = nn.Sequential(nn.Linear(embed_dim//2, r*embed_dim), nn.GELU(), 
                            nn.Linear(r*embed_dim, embed_dim))

        self.fn = None  
        self.fn2 = None
        self.scale=1

    def forward(self, x, mask=None, channel_mask=None, agg_type=None, inp_type=None, fn=None):

        inp_type = self.inp_type if inp_type is None else inp_type
        agg_type = self.agg_type if agg_type is None else agg_type

        fn = self.fn if fn is None else fn

        if 'dual' in agg_type:

            x_reduced = self.dim_reducer(x)

            # x_reduced = x.clone()

            x_joint = self.forward(x_reduced, mask=mask, channel_mask=channel_mask, agg_type='joint')
            x_sep = self.forward(x_reduced, mask=mask, channel_mask=channel_mask, agg_type='sep')

            if self.agg_type in ['dualv2', 'dualv3']:
                x_joint = self.norm0(x_joint)
                x_sep = self.norm0(x_sep)

                a_soft = (self.alphas*self.scale).softmax(0)

                x = a_soft[0] * x_joint + a_soft[1] * x_sep

                x = self.mixer(x)
                x = self.norm(x)

            elif self.agg_type in ['dualv4']:
                # x_joint = self.norm0(x_joint)
                # x_sep = self.norm0(x_sep)

                x = torch.stack((x_joint, x_sep), dim=1).max(1)[0]
                x = self.mixer(x)
                # x = self.norm(x)

            elif self.agg_type == 'dualx3':

                x = torch.stack((x_joint, x_sep), dim=1)
                x = self.fn(x)
                x = self.norm(self.mixer(x))

            elif self.agg_type == 'dualx4':

                x = torch.stack((x_joint, x_sep), dim=1)
                x = self.fn(x)
                x = self.norm(self.mixer(x))

            else:
                x = torch.cat((x_joint, x_sep), dim=-1)
                x = self.norm(x)

            # print('x')b + 

        elif agg_type == 'chan':

            b,c,n,d = x.shape

            x = rearrange(x, 'b c n d -> (b n) c d')
            
            if channel_mask is not None:
                channel_mask = repeat(channel_mask, 'b c -> b c n 1', n=n)
                channel_mask = rearrange(channel_mask, 'b c n 1 -> (b n) 1 1 c')

            x = self.fn(x, mask=channel_mask)
            
            x = rearrange(x, '(b n) d -> b n d', n=n)

            x = self.fn(x)

            # print('x')

        elif agg_type == 'joint':

                if inp_type == 'joint' and x.shape[1] != 1:
                    x_cls = x[:,:1,:1]
                    x_ptc = rearrange(x[:,:,1:], 'b (m c) n d -> b m (c n) d', c=x.shape[1])
                    x = torch.cat((x_cls, x_ptc), dim=2)

                elif inp_type == 'sep':
                    # collapse the CLS tokens.
                    x_cls = x[:,:,:1]

                    if channel_mask is not None:
                        channel_mask = channel_mask.unsqueeze(-1).unsqueeze(-1)
                        mask_f = channel_mask.float()
                        masked_sum = (x_cls * mask_f).sum(dim=1, keepdim=True) 
                        counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
                        x_cls = masked_sum / counts
                    else:
                        x_cls = x_cls.mean(1, keepdim=True)
                    

                    x_ptc = rearrange(x[:,:,1:], 'b c n d -> b 1 (c n) d')
                    x = torch.cat((x_cls, x_ptc), dim=2)

                x = rearrange(x, 'b c n d -> b (c n) d')
                x = self.fn(x, mask=mask)

        elif agg_type == 'sep':
            if inp_type == 'joint' and (x.shape[2] not in [17, 197]):
                x_ptc = x[:, :, 1:]

                n_tokens = x_ptc.shape[2]
                
                if n_tokens % 196 == 0:
                    # CHAMMI or JUMP
                    c = n_tokens // 196
                elif n_tokens % 16 == 0:
                    # So2Sat
                    c = n_tokens // 16

                x_cls = repeat(x[:, :, :1], 'b 1 1 d -> b c 1 d', c=c)
                x_ptc = rearrange(x_ptc, 'b 1 (c n) d -> b c n d', c=c)
                x = torch.cat([x_cls, x_ptc], dim=2)

            b,c,n,d = x.shape

            x = rearrange(x, 'b c n d -> (b c) n d')

            x = self.fn(x)

            x = rearrange(x, '(b c) d -> b c d', c=c)

            if channel_mask is not None:
                channel_mask = rearrange(channel_mask, 'b c -> b 1 1 c')

            x = self.fn(x, mask=channel_mask)
        else:
            raise ValueError(f"Unknown chan_setting: {self.chan_setting}")
        
        return x


class BaseCLSAggregator(nn.Module):
    def __init__(self, embed_dim, chan_setting='joint_joint', num_heads=6, *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.chan_setting = chan_setting

        [self.inp_type, self.agg_type] = chan_setting.split('_')

        # fn will be assigned in subclasses
        self.fn = None  

    def forward(self, x, **kwargs):

        # x.shape = [B ,C, 1, D]
        x = rearrange(x, 'b c n d -> b (c n) d')
        x = self.fn(x, **kwargs)

        return x


class MAB_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)

        # we override this to accommodate dual aggregator setting.
        embed_dim = self.embed_dim

        self.fn = MAB(dim_Q=embed_dim, dim_K=embed_dim, dim_V=embed_dim, num_heads=num_heads)
        self.fn2 = MAB(dim_Q=embed_dim, dim_K=embed_dim, dim_V=embed_dim, num_heads=num_heads)

class SimPool_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)

        # we override this to accommodate dual aggregator setting.
        embed_dim = self.embed_dim
        self.fn = SimPool(dim=embed_dim, out_features=embed_dim)

class MABx3_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, chan_alpha=1.0, *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)
        self.fn = MAB(dim_Q=embed_dim, dim_K=embed_dim, dim_V=embed_dim, num_heads=num_heads)
        self.alpha = nn.Parameter(chan_alpha * torch.ones([]))

    def forward(self, x):

        b, c, n, d = x.shape

        x = rearrange(x, 'b c n d -> (b n) c d')

        y = self.fn(x, x, squeeze_out=False)

        x = x + self.alpha * y

        x = rearrange(x, '(b n) c d -> b c n d', b=b, n=n)

        x = super(MABx3_Aggregator, self).forward(x)

        return x

class MHCAx3_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, chan_alpha=1.0, msa_dim='chan', *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)
        self.fn = Attentive(dim=embed_dim, out_features=embed_dim, num_heads=num_heads)

        self.msa_dim=msa_dim
        self.alpha = nn.Parameter(chan_alpha * torch.ones([]))

        self.rearrange_strings = {
            'chan':('b c n d -> (b n) c d', '(b n) c d -> b c n d'),
            'all':('b c n d -> b (c n) d', 'b (c n) d -> b c n d'), 
            'spac':('b c n d -> (b c) n d', '(b c) n d -> b c n d')
        }

    def forward(self, x):

        b, c, n, d = x.shape

        x_atn = rearrange(x, self.rearrange_strings[self.msa_dim][0], b=b, c=c, n=n, d=d)

        # if self.training:
        #     n_b, n_q, n_k = x_atn.size(0), x_atn.size(1), x_atn.size(1)
        #     attn_mask = (torch.rand(n_b, 1, n_q, n_q, device=x_atn.device) < 0.5)
        # else:
        #     attn_mask=None

        # ref_chans = 8

        attn_mask=None

        y = self.fn(x_atn, cls_token=x_atn, average_pool=False, attn_mask=attn_mask)
        y = rearrange(y, self.rearrange_strings[self.msa_dim][1], b=b, c=c, n=n, d=d)

        # y = y * c / ref_chans

        x = x + self.alpha * y

        x = super(MHCAx3_Aggregator, self).forward(x)

        return x


class Protox3_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, chan_alpha=1.0, msa_dim='chan', *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)

        self.fn = Prototypical_multi(dim=embed_dim, num_prototypes=197, num_classes=embed_dim)

        self.msa_dim=msa_dim
        self.alpha = nn.Parameter(chan_alpha * torch.ones([]))

        self.rearrange_strings = {
            'chan':('b c n d -> (b n) c d', '(b n) c d -> b c n d'),
            'all':('b c n d -> b (c n) d', 'b (c n) d -> b c n d'), 
            'spac':('b c n d -> (b c) n d', '(b c) n d -> b c n d')
        }

    def forward(self, x):

        b, c, n, d = x.shape

        y = self.fn(x)

        x_atn = rearrange(x, self.rearrange_strings[self.msa_dim][0], b=b, c=c, n=n, d=d)
        y = self.fn(x_atn, cls_token=x_atn, average_pool=False)
        y = rearrange(y, self.rearrange_strings[self.msa_dim][1], b=b, c=c, n=n, d=d)

        x = x + self.alpha * y

        x = super(MHCAx3_Aggregator, self).forward(x)

        return x
    

class AbMILP_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', *args, **kwargs):
        super().__init__(embed_dim, chan_setting, *args, **kwargs)
                # we override this to accommodate dual aggregator setting.
        embed_dim = self.embed_dim
        self.fn = AbMILP(dim=embed_dim, out_features=embed_dim)


class EP_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)
                # we override this to accommodate dual aggregator setting.
        embed_dim = self.embed_dim
        self.fn = EfficientProbing(dim=embed_dim, num_queries=32)

class Proto_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)
                # we override this to accommodate dual aggregator setting.
        embed_dim = self.embed_dim
        self.fn = Prototypical_multi(dim=embed_dim, num_prototypes=4, num_classes=embed_dim)

        

class Protobin_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, num_prototypes=4, *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads,  *args, **kwargs)
                # we override this to accommodate dual aggregator setting.
        embed_dim = self.embed_dim
        self.fn = Prototypical_multi_binarized_simple(dim=embed_dim, num_prototypes=num_prototypes, num_classes=embed_dim)

class MHCA_Aggregator(BaseAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)
                # we override this to accommodate dual aggregator setting.
        embed_dim = self.embed_dim
        self.fn = Attentive(dim=embed_dim, out_features=embed_dim, num_heads=num_heads)


from einops import repeat

class MHCA_CLS_Aggregator(BaseCLSAggregator):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, chan_alpha=1.0, msa_dim='chan', 
                 learnable_alpha=True, use_mean_features=False,
                 *args, **kwargs):
        super().__init__(embed_dim, chan_setting, num_heads, *args, **kwargs)

        self.use_mean_features = use_mean_features

        self.norm = nn.LayerNorm(embed_dim)

        if use_mean_features:
            out_features = embed_dim // 2
        else:
            out_features = embed_dim

        self.fn = Attentive(dim=embed_dim, out_features=out_features, num_heads=num_heads)

        self.msa_dim=msa_dim
        if learnable_alpha:
            self.alpha = nn.Parameter(chan_alpha * torch.ones([]))
        else:
            self.alpha = chan_alpha

        self.rearrange_strings = {
            'chan':('b c n d -> (b n) c d', '(b n) c d -> b c n d'),
            'all':('b c n d -> b (c n) d', 'b (c n) d -> b c n d'), 
            'spac':('b c n d -> (b c) n d', '(b c) n d -> b c n d')
        }

    def forward(self, x, channel_masks=None, *args, **kwargs):

        b, c, n, d = x.shape

        if n>1:
            # extract the CLS token.
            x = x[:,:,:1]
            n=1

        if self.msa_dim != 'none':
            x_atn = rearrange(x, self.rearrange_strings[self.msa_dim][0], b=b, c=c, n=n, d=d)
            y = self.fn(x_atn, cls_token=x_atn, average_pool=False)
            y = rearrange(y, self.rearrange_strings[self.msa_dim][1], b=b, c=c, n=n, d=d)
            x = x + self.alpha * y

        if channel_masks is not None:
            channel_masks = repeat(channel_masks, 'b c -> b n1 n2 c', n1=1, n2=1)


        if self.use_mean_features:
            x1 = super(MHCA_CLS_Aggregator, self).forward(x, attn_mask=channel_masks)

            x_mean = rearrange(x, 'b c 1 d -> b c d').mean(1)
            
            x2 = self.fn.linear(x_mean)

            x = self.norm(torch.cat((x1, x2), dim=-1))

        else:
            
            x = super(MHCA_CLS_Aggregator, self).forward(x, attn_mask=channel_masks)

        return x
    

class MHCA_Dual_Aggregator(nn.Module):
    def __init__(self, embed_dim=768, chan_setting='joint_joint', num_heads=6, chan_alpha=1.0, msa_dim='chan', 
                 learnable_alpha=True, use_mean_features=False,
                 *args, **kwargs):
        super().__init__()

        self.use_mean_features = use_mean_features

        self.norm = nn.LayerNorm(2*embed_dim)
        self.reducer = nn.Linear(2*embed_dim, embed_dim)

        self.fn = Attentive(dim=embed_dim, out_features=embed_dim, num_heads=num_heads)

        self.msa_dim=msa_dim
        if learnable_alpha:
            self.alpha = nn.Parameter(chan_alpha * torch.ones([]))
        else:
            self.alpha = chan_alpha

        self.rearrange_strings = {
            'chan':('b c n d -> (b n) c d', '(b n) c d -> b c n d'),
            'all':('b c n d -> b (c n) d', 'b (c n) d -> b c n d'), 
            'spac':('b c n d -> (b c) n d', '(b c) n d -> b c n d')
        }

    def forward(self, x, channel_masks=None, patch_masks=None, *args, **kwargs):

        x_cls_global = x[:,0]

        # perform within-chan pooling.
        x_rem = x[:,1:]

        # assume 196 tokens for now.
        n = 196
        c = x_rem.size(1) // n 

        x_rem = rearrange(x_rem, 'b (c n) d -> (b c) n d', c=c, n=n)
        
        if patch_masks is not None:
            pm = patch_masks[:,:,:,1:]
            pm = rearrange(pm, 'b 1 1 (c n) -> (b c) 1 1 n', c=c, n=n)
        else:
            pm = None

        pooled_per_chan = self.fn(x_rem, attn_mask=pm)
        pooled_per_chan = rearrange(pooled_per_chan, '(b c) d -> b c d', c=c)

        if channel_masks is not None:
            channel_masks = repeat(channel_masks, 'b c -> b 1 1 c')

        x_cls_local = self.fn(pooled_per_chan, attn_mask=channel_masks)

        x_cls = torch.cat((x_cls_global, x_cls_local), dim=-1)
        x_cls = self.norm(x_cls)
        x_cls = self.reducer(x_cls)

        return x_cls

    

