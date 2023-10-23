from torch import nn
import torch
from einops import rearrange
import torch.nn.functional as F
from models.arch.vq import BlockBasedResidualVectorQuantizer


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# class MS_MSA(nn.Module):
#     def __init__(
#             self,
#             dim,
#             dim_head,
#             heads,
#     ):
#         super().__init__()
#
#         self.num_heads = heads
#         self.dim_head = dim_head
#         self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
#         self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
#         self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
#         self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
#         self.proj = nn.Linear(dim_head * heads, dim, bias=True)
#         self.pos_emb = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
#             GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
#         )
#         self.dim = dim
#             # b x 3 x h x w  -->  b x dim_head x h x w
#
#     def forward(self, x_in, mask=None):
#         """
#         x_in: [b,h,w,c]
#         return out: [b,h,w,c]
#         """
#
#         b, h, w, c = x_in.shape
#         x = x_in.reshape(b,h*w,c)
#         q_inp = self.to_q(x)
#         k_inp = self.to_k(x)
#         v_inp = self.to_v(x)  # b x (hw) x (dim_head*heads)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
#                                 (q_inp, k_inp, v_inp))
#         v = v  # v.shape = b x heads x (hw) x dim_head
#         # q: b,heads,hw,c
#         q = q.transpose(-2, -1)
#         k = k.transpose(-2, -1)
#         v = v.transpose(-2, -1)
#
#         q = F.normalize(q, dim=-1, p=2)
#         k = F.normalize(k, dim=-1, p=2)
#         attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
#         attn = attn * self.rescale
#         attn = attn.softmax(dim=-1)
#         x = attn @ v   # b,heads,d,hw
#         x = x.permute(0, 3, 1, 2).contiguous()    # Transpose
#         x = x.reshape(b, h * w, self.num_heads * self.dim_head)
#         out_c = self.proj(x).view(b, h, w, c)
#         out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
#         out = out_c + out_p
#
#         return out


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)  # b x (hw) x (dim_head*heads)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v  # v.shape = b x heads x (hw) x dim_head
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2).contiguous()    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
        self.dim = dim
        self.mult = mult

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1).contiguous()


class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """

        x = x.permute(0, 2, 3, 1).contiguous()
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2).contiguous()
        return out

