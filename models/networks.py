import torch
from models.arch.network import Network


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'RQ_LLIE':
        netG = Network(stage=opt['stage'], depth=opt_net['num_code'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

