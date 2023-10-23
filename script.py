import torch

file_name = "/home/liuyunlong/project/code/SNR-Aware-Low-Light-Enhance-main/experiments/LOLv2_Synthetic_S3_D6_w_LA/models/best_G.pth"
state_dict = torch.load(file_name)


state_dict_new = {}

for item in state_dict:
    if item == 'htlyl_64.codebook_transform':
        state_dict_new['query.codebook_transform'] = state_dict[item]
    elif 'SA' in item:
        state_dict_new[item.replace('SA', 'BA')] = state_dict[item]
    else:
        state_dict_new[item] = state_dict[item]

torch.save(state_dict_new,"./pretrained_models/LOLv2Synthetic.pth")
