import torch
from models.archs.RRDBNet_RFB2_convup_nearest import RRDBNet_RFB2_convup_nearest

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'RRDBNet_rfb2_convup_nearest':
        netG = RRDBNet_RFB2_convup_nearest(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG