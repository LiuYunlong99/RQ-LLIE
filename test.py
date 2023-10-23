from models.arch.network import Network
from torch.nn import functional as F
import torch


model = Network(stage=2, depth=8).cuda()
model.set_query_codebook()
model.load_state_dict(torch.load("./pretrained_models/LOLv1.pth"))
x = torch.ones(1, 3, 256, 256).cuda()

with torch.no_grad():
    M = F.relu(x - model.thr_conv(x))

    f1, f2, f3 = model.encode(x)
    fq, distance_map = model.vq_64.forward_with_query(f3, model.query)
    f1_d, f2_d, f3_d = model.decode(fq)

    f1_cat = model.fusion_128([model.up_fusion_2(f3), f2, f2_d, model.down_fusion_local(f1), model.down_fusion_prior(f1_d)])
    f1_f = model.decoder_128_fusion(f1_cat)
    f1_f_wo_ba = f1_f
    f1_f = f1_f + f1_f * model.BA_128(M)

    f2_cat = torch.cat([model.up_fusion_3(f1_f), f1, f1_d, model.up_fusion_local(f2), model.up_fusion_prior(f2_d)])
    f2_f = model.decoder_256_fusion(f2_cat)
    f2_f = f2_f + f2_f * model.BA_256(M)
    x_rec = model.conv_fusion_out(f2_f)

ckpt = torch.load("/home/liuyunlong/project/code/SNR-Aware-Low-Light-Enhance-main/ckpt.pth")

dic = {'feat_256': f1, 'feat_128': f2, 'feat_64': f3, 'feat_q': fq, 'decode_64': f3_d, 'decode_128': f2_d,
       'decode_256': f1_d, 'fusion_128': f1_f, 'fusion_256': f2_f, 'fusion_128_wo_ba': f1_f_wo_ba}

for item in dic:
    print(f"{item}-->{torch.equal(dic[item], ckpt[item])}")
