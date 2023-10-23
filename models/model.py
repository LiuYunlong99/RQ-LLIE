import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss


logger = logging.getLogger('base')


class LLIEModel(BaseModel):
    def __init__(self, opt):
        super(LLIEModel, self).__init__(opt)
        self.stage = opt['stage']
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
                                                find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
        if self.stage == 1:
            opt['stage'] = 0
            self.netHQ = networks.define_G(opt).to(self.device)
            opt['stage'] = 1
            if opt['dist']:
                self.netHQ = DistributedDataParallel(self.netHQ, device_ids=[torch.cuda.current_device()])
            else:
                self.netHQ = DataParallel(self.netHQ)
            ckpt = torch.load(opt['path']['net_hq'])
            try:
                self.netHQ.module.load_state_dict(ckpt, strict=True)
            except RuntimeError:
                print("Failed to load HQ Model ...")
            self.netG.module.load_state_dict(ckpt, strict=False)
            self.netG.module.set_query_codebook()

        elif self.stage == 2:
            ckpt = torch.load(opt['path']['pretrained_model'])
            self.netG.module.set_query_codebook()
            self.netG.module.load_state_dict(ckpt, strict=False)

        optim_params = self.netG.module.train_parameters()
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction=opt['train']['l_pix_reduction']).to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction=opt['train']['l_pix_reduction']).to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss(reduction=opt['train']['l_pix_reduction']).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000, 15000, 20000, 25000, 30000],
                                                             gamma=0.5))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            elif train_opt['lr_scheme'] == "CosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_opt['niter'],
                                                                                      eta_min=train_opt['eta_min']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()

        if self.stage == 0:
            output, codebook_loss, _, _ = self.netG(self.real_H)
            self.fake_H = output
            l_pix = self.cri_pix(self.fake_H, self.real_H)
            l_final = l_pix + codebook_loss
        elif self.stage == 1:
            with torch.no_grad():
                _, _, indices_list_gt, bottle_list_gt = self.netHQ(self.real_H)
            alpha_list, bottle_list = self.netG(self.var_L)
            loss_alpha = 0
            for index, (alpha, indices_gt) in enumerate(zip(alpha_list, indices_list_gt)):
                if index > 0:
                    break
                loss_alpha += F.l1_loss(alpha, indices_gt)

            loss_bottle = 0
            for bottle, bottle_gt in zip(bottle_list, bottle_list_gt):
                loss_bottle += F.l1_loss(bottle, bottle_gt)
            l_final = 0 * loss_alpha + loss_bottle
            self.log_dict['l_alpha'] = 0
            self.log_dict['l_bottle'] = loss_bottle.item()
            l_pix = torch.zeros_like(l_final)
        elif self.stage == 2:
            self.fake_H = self.netG(self.var_L)
            l_pix = self.cri_pix(self.fake_H, self.real_H)
            l_final = l_pix
        else:
            raise NotImplementedError()

        l_final.backward()

        self.optimizer_G.step()

        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.stage == 0:
                ans = self.netG(self.real_H)
                self.fake_H = ans[0]
            elif self.stage == 1:
                ans =self.netG(self.var_L)
                self.fake_H = ans[0]
            elif self.stage == 2:
                self.fake_H = self.netG(self.var_L)
            else:
                raise NotImplementedError()
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del self.real_H
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)



