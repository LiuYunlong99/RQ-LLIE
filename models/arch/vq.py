from torch import nn
import torch
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                   torch.sum(y ** 2, dim=1) - 2 * \
                   torch.matmul(x, y.t())

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)  # b x N

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            # codebook_loss = self.dist(z_q, z_q_gt.detach()).mean() \
            # + self.beta * self.dist(z_q_gt.detach(), z)
            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        b, _, h, w = indices.shape

        indices = indices.flatten().to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q

    def get_k_nearest_neighbors(self, z, k=8):
        # reshape z -> (batch, height, width, channel) and flatten
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()  # b h w c
        z_flattened = z.view(-1, self.e_dim)  # bhw c

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)  # b x N
        _, idx = torch.topk(-d, k, dim=-1)
        centers = self.embedding(idx)
        centers = centers.view(b, h*w, k, c)
        return centers


class ResidualVectorQuantizer(VectorQuantizer):
    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False, depth=6):
        super().__init__(n_e, e_dim, beta, LQ_stage)
        self.depth = depth

    def forward(self, z, gt_indices=None, current_iter=None):
        b, c, h, w = z.shape
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        if gt_indices is not None:
            # gt_indices.shape = b x d x n
            z_q_gt = 0
            for i in range(len(gt_indices)):
                z_q_gt = z_q_gt + self.embedding(gt_indices[i])
            z_q_gt = z_q_gt.view(z.shape)

        codebook = self.embedding.weight
        z_q, residual, indices = 0, z_flattened, []
        for i in range(self.depth):
            d = self.dist(residual, codebook)  # b x N
            min_encoding_indices = torch.argmin(d, dim=1)  # b x 1
            delta = self.embedding(min_encoding_indices)
            z_q = z_q + delta
            residual = residual - delta
            indices.append(min_encoding_indices.clone())

        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)

        if self.LQ_stage and gt_indices is not None:
            # d = self.dist(z_flattened, codebook)
            # d_gt = self.dist(z_q_gt, codebook)
            # codebook_loss = F.kl_div(F.log_softmax(-d, dim=-1), F.softmax(-d_gt, dim=-1))
            codebook_loss = self.beta * self.dist(z_q_gt.detach(), z)
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        indices = torch.stack(indices, dim=1).reshape(b, h, w, -1)
        return z_q, codebook_loss, indices

    def get_codebook_entry(self, indices):
        b, d, h, w = indices.shape
        gt_indices = indices.reshape(b, d, h * w)
        z_q = torch.sum(self.embedding(gt_indices), dim=1, keepdim=False)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


class BlockBasedResidualVectorQuantizer(ResidualVectorQuantizer):
    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False, depth=6):
        super().__init__(n_e, e_dim, beta, LQ_stage, depth)
        self.unfold = nn.Unfold(kernel_size=(2, 2))
        self.z_length = e_dim // 4
        self.z_stride = self.z_length // 2
        print(f"RQ Depth = {self.depth} ...")

    def forward(self, z, gt_indices=None, current_iter=None):
        assert self.z_length == z.size(1)
        z_flattened = self.unfold(z).permute(0, 2, 1)
        z_flattened = z_flattened.reshape(-1, self.e_dim)
        codebook = self.embedding.weight
        z_q, residual, indices = 0, z_flattened, []
        for i in range(self.depth):
            d = self.dist(residual, codebook)  # b x N
            min_encoding_indices = torch.argmin(d, dim=1)  # b x 1
            delta = self.embedding(min_encoding_indices)

            """
            pred_one_hot = F.one_hot(min_encoding_indices, self.n_e).float()
            delta = torch.einsum("bm,md->bd", pred_one_hot, codebook)
            """

            z_q = z_q + delta
            residual = residual - delta
            # indices.append(min_encoding_indices.clone())
            indices.append(d)

        z_q = self.fold(z_q, z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)
        codebook_loss = q_latent_loss + e_latent_loss * self.beta

        z_q = z + (z_q - z).detach()

        # indices = torch.stack(indices, dim=1).view(z.size(0), -1, self.depth)  # b x n x d
        return z_q, codebook_loss, indices

    def fold(self, z_t, shape_z):
        b, c, h, w = shape_z
        z_t = z_t.view(b, -1, self.e_dim).permute(0, 2, 1)
        fold = nn.Fold(output_size=(h, w), kernel_size=(2, 2))
        count_t = torch.ones(1, self.z_length, h, w).cuda()
        count_t = self.unfold(count_t)
        count_t = fold(count_t)
        z_q = fold(z_t)
        z_q = z_q / count_t
        return z_q.view(shape_z)

