from torch import nn
from models.arch.MSAB import MSAB
from models.arch.vq import BlockBasedResidualVectorQuantizer
import torch
from torch.nn import functional as F

channel_query_dict = {
    64: 256,
    128: 128,
    256: 64,
}


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=6):
        super().__init__()
        ksz = 3
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, ksz, stride=1, padding=1),
                                   MSAB(dim=out_ch, num_blocks=num_blocks, dim_head=out_ch//4, heads=4))
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        x = self.block(x)
        return x


class ConcatFusion(nn.Module):
    def __init__(self, in_dim, num_fea=3):
        super().__init__()
        self.conv = nn.Conv2d(in_dim*num_fea, in_dim, 3, 1, 1)

    def forward(self, z_list):
        z_cat = torch.cat(z_list, dim=1)
        return self.conv(z_cat)


class VQModule(nn.Module):
    def __init__(self, in_ch, e_dim=512, n_e=1024, depth=6):
        super().__init__()
        self.opt = (in_ch, e_dim, n_e, depth)
        self.quantize = BlockBasedResidualVectorQuantizer(n_e=n_e, e_dim=e_dim, depth=depth)

    def forward(self, x):
        z_q, codebook_loss, indices = self.quantize(x)
        return z_q, codebook_loss, indices

    def forward_with_query(self, x, query):
        code_book = self.quantize.embedding.weight
        x_unfold = self.quantize.unfold(x).permute(0, 2, 1).reshape(-1, self.quantize.e_dim)
        z_q, alpha_list = query(x_unfold, code_book)
        z_q_fold = self.quantize.fold(z_q.contiguous(), x.shape)
        return z_q_fold, alpha_list

    def lookup(self, indices, shape):
        z_q = 0
        for index in indices:
            z_q += self.quantize.embedding(index)
        z_q_fold = self.quantize.fold(z_q, shape)
        return z_q_fold


class QueryModule(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.codebook_transform = None
        self.depth = depth

    def forward(self, z, codebook):
        N, d = codebook.shape
        z_t = z
        codebook_t = self.codebook_transform
        z_q, residual = 0, z_t.detach()
        maps = []
        for i in range(self.depth):
            dist_map = self.dist(residual, codebook_t)
            maps.append(dist_map)
            pred = torch.argmin(dist_map, keepdim=False, dim=1)
            pred_one_hot = F.one_hot(pred, N).float()
            delta = torch.einsum("bm,md->bd", pred_one_hot, codebook)
            z_q = z_q + delta
            residual = residual - delta

        return z_q, maps

    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                   torch.sum(y ** 2, dim=1) - 2 * \
                   torch.matmul(x, y.t())


class Network(nn.Module):
    def __init__(self, in_ch=3, n_e=1024, stage=0, depth=6, num_block=3):
        super().__init__()
        assert stage in [0, 1, 2]
        self.stage = stage
        curr_res = 256
        self.conv_in = nn.Conv2d(in_ch, channel_query_dict[curr_res], 3, 1, 1)
        self.encoder_256 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block)

        self.down1 = nn.Conv2d(channel_query_dict[curr_res], channel_query_dict[curr_res // 2], 3, stride=2, padding=1)
        curr_res = curr_res // 2  # 128
        self.encoder_128 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block)

        self.down2 = nn.Conv2d(channel_query_dict[curr_res], channel_query_dict[curr_res // 2], 3, stride=2, padding=1)
        curr_res = curr_res // 2  # 64
        self.encoder_64 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=num_block)

        self.vq_64 = VQModule(channel_query_dict[curr_res], channel_query_dict[curr_res] * 4, n_e, depth=depth)

        self.decoder_64 = BasicBlock(channel_query_dict[curr_res], channel_query_dict[curr_res], num_blocks=3)

        self.up2 = nn.Upsample(scale_factor=2)
        curr_res *= 2
        self.decoder_128 = BasicBlock(channel_query_dict[curr_res // 2], channel_query_dict[curr_res], num_blocks=3)

        self.up3 = nn.Upsample(scale_factor=2)
        curr_res *= 2
        self.decoder_256 = BasicBlock(channel_query_dict[curr_res // 2], channel_query_dict[curr_res], num_blocks=3)

        self.conv_out = nn.Conv2d(channel_query_dict[curr_res], 3, 3, 1, 1)

        if self.stage in [1, 2]:
            self.query = QueryModule(depth)
            if self.stage == 2:
                self.fusion_128 = ConcatFusion(channel_query_dict[128], num_fea=5)
                self.fusion_256 = ConcatFusion(channel_query_dict[256], num_fea=5)
                self.down_fusion_local = nn.Conv2d(channel_query_dict[256], channel_query_dict[128], kernel_size=(3, 3),
                                                   stride=(2, 2), padding=(1, 1))
                self.down_fusion_prior = nn.Conv2d(channel_query_dict[256], channel_query_dict[128], kernel_size=(3, 3),
                                                   stride=(2, 2), padding=(1, 1))
                self.up_fusion_local = nn.Sequential(nn.Upsample(scale_factor=2),
                                                     nn.Conv2d(channel_query_dict[128], channel_query_dict[256], 1, 1,
                                                               0))
                self.up_fusion_prior = nn.Sequential(nn.Upsample(scale_factor=2),
                                                     nn.Conv2d(channel_query_dict[128], channel_query_dict[256], 1, 1,
                                                               0))
                self.up_fusion_2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                                 nn.Conv2d(channel_query_dict[64], channel_query_dict[128], 1, 1, 0))
                self.up_fusion_3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                                 nn.Conv2d(channel_query_dict[128], channel_query_dict[256], 1, 1, 0))
                self.decoder_128_fusion = BasicBlock(channel_query_dict[128], channel_query_dict[128])
                self.decoder_256_fusion = BasicBlock(channel_query_dict[256], channel_query_dict[256])
                self.conv_fusion_out = nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1)

                self.thr_conv = nn.Conv2d(3, 3, 3, 1, 1)
                self.BA_256 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.GELU(),
                                            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
                                            nn.Conv2d(64, 1, 3, 1, 1))
                self.BA_128 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.GELU(),
                                            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
                                            nn.Conv2d(64, 1, 4, 2, 1))

    def set_query_codebook(self):
        self.query.codebook_transform = nn.Parameter(self.vq_64.quantize.embedding.weight.clone(),
                                                        requires_grad=True)
        print("Successfully set query codebook !")

    def forward(self, x):
        if self.stage == 0:
            return self.forward_s1(x)
        elif self.stage == 1:
            return self.forward_s2(x)
        else:
            return self.forward_s3(x)

    def encode(self, x):
        x = self.conv_in(x)
        f1 = self.encoder_256(x)
        f2 = self.encoder_128(self.down1(f1))
        f3 = self.encoder_64(self.down2(f2))
        return f1, f2, f3

    def decode(self, fq):
        f3_d = self.decoder_64(fq)
        f2_d = self.decoder_128(self.up2(f3_d))
        f1_d = self.decoder_256(self.up3(f2_d))
        return f1_d, f2_d, f3_d

    def forward_s1(self, x):
        f1, f2, f3 = self.encode(x)
        fq, codebook_loss, distance_map = self.vq_64(f3)
        f1_d, f2_d, f3_d = self.decode(fq)
        x_rec = self.conv_out(f1_d)
        return x_rec, codebook_loss, distance_map, [f1, f2, f3]

    def forward_s2(self, x):
        f1, f2, f3 = self.encode(x)
        fq, distance_map = self.vq_64.forward_with_query(f3, self.query)
        f1_d, f2_d, f3_d = self.decode(fq)
        x_rec = self.conv_out(f1_d)
        return x_rec, distance_map, [f1, f2, f3]

    def forward_s3(self, x):
        M = F.relu(x - self.thr_conv(x))
        with torch.no_grad():
            f1, f2, f3 = self.encode(x)
            fq, distance_map = self.vq_64.forward_with_query(f3, self.query)
            f1_d, f2_d, f3_d = self.decode(fq)
        f1_cat = self.fusion_128([self.up_fusion_2(f3), f2, f2_d, self.down_fusion_local(f1), self.down_fusion_prior(f1_d)])
        f1_f = self.decoder_128_fusion(f1_cat)
        f1_f = f1_f + f1_f * self.BA_128(M)

        f2_cat = self.fusion_256([self.up_fusion_3(f1_f), f1, f1_d, self.up_fusion_local(f2), self.up_fusion_prior(f2_d)])
        f2_f = self.decoder_256_fusion(f2_cat)
        f2_f = f2_f + f2_f * self.BA_256(M)
        x_rec = self.conv_fusion_out(f2_f)
        return x_rec

    def train_parameters(self):
        if self.stage == 0:
            return self.parameters()
        elif self.stage == 1:
            return list(self.conv_in.parameters()) + list(self.encoder_64.parameters()) + \
                   list(self.encoder_128.parameters()) + list(self.encoder_256.parameters()) + \
                   list(self.down1.parameters()) + list(self.down2.parameters()) + list(self.query.parameters())
        else:
            return list(self.fusion_128.parameters()) + list(self.fusion_256.parameters()) + \
                   list(self.down_fusion_prior.parameters()) + list(self.down_fusion_local.parameters()) + \
                   list(self.up_fusion_prior.parameters()) + list(self.up_fusion_local.parameters()) + \
                   list(self.up_fusion_3.parameters()) + list(self.up_fusion_2.parameters()) + \
                   list(self.decoder_128_fusion.parameters()) + list(self.decoder_256_fusion.parameters()) + \
                   list(self.conv_fusion_out.parameters()) + list(self.thr_conv.parameters()) + \
                   list(self.BA_256.parameters()) + list(self.BA_128.parameters())

