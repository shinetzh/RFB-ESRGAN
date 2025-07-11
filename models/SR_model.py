import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        # if opt['dist']:
        # self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        # # else:
        self.netG = DataParallel(self.netG)
        # print network
        self.load()

        self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        # print(data['LQ'].size())
        self.var_L = data.to(self.device)  # LQ
        # if need_GT:
        #     self.real_H = data['GT'].to(self.device)  # GT


    def test(self):
        # self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        # self.netG.train()
        return self.fake_H


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
