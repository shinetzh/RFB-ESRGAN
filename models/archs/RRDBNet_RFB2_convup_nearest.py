import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.archs.common_rfb import BasicRFB_a as rfbconv
from models.archs.common_rfb import Upsampler

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)




class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.alpha = torch.tensor([1.0], requires_grad=True).to(torch.device('cuda'))
        # self.alpha = torch.tensor([1.0], requires_grad=True)
        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.alpha + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        self.alpha = torch.tensor([1.0], requires_grad=True).to(torch.device('cuda'))
        # self.alpha = torch.tensor([1.0], requires_grad=True)
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * self.alpha + x



class ResidualDenseBlock_5C_rfb(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C_rfb, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = rfbconv(nf, gc, scale=1)
        self.conv2 = rfbconv(nf + gc, gc, scale=1)
        self.conv3 = rfbconv(nf + 2 * gc, gc, scale=1)
        self.conv4 = rfbconv(nf + 3 * gc, gc, scale=1)
        self.conv5 = rfbconv(nf + 4 * gc, nf, scale=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.alpha = torch.tensor([1.0], requires_grad=True).to(torch.device('cuda'))
        # self.alpha = torch.tensor([1.0], requires_grad=True)
        # initialization
        # initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
        #                              0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.alpha + x



class RRDB_rfb(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB_rfb, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C_rfb(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C_rfb(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C_rfb(nf, gc)
        self.alpha = torch.tensor([1.0], requires_grad=True).to(torch.device('cuda'))
        # self.alpha = torch.tensor([1.0], requires_grad=True)
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * self.alpha + x


class RRDBNet_RFB2_convup_nearest(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32): #(3,3,64,23,32)
        super(RRDBNet_RFB2_convup_nearest, self).__init__()
        RRDB_block_f_rfb = functools.partial(RRDB_rfb, nf=nf, gc=gc)
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, 15)
        self.RRDB_trunk_rfb = make_layer(RRDB_block_f_rfb, 8)
        self.trunk_conv = rfbconv(nf, nf, scale=1)
        #### upsampling
        self.upconv1 = rfbconv(nf, nf, scale=1)
        # sub pixel
        self.upsampler2 = Upsampler(rfbconv, 2, nf, act=False)
        self.upconv2 = rfbconv(nf, nf, scale=1)
        self.upconv3 = rfbconv(nf, nf, scale=1)
        # sub pixel
        self.upsampler4 = Upsampler(rfbconv, 2, nf, act=False)
        self.upconv4 = rfbconv(nf, nf, scale=1)

        # # define tail module
        # modules_tail = [
        #     Upsampler(rfbconv, 16, nf, act=False),
        #     rfbconv(nf, 3, scale=1)]

        # self.tail = nn.Sequential(*modules_tail)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.alpha = torch.tensor([1.0], requires_grad=True).to(torch.device('cuda'))
        # self.alpha = torch.tensor([1.0], requires_grad=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.RRDB_trunk(fea)
        trunk = self.RRDB_trunk_rfb(trunk)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk*self.alpha
        # nearest interplote
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # sub pixel
        fea = self.lrelu(self.upconv2(self.upsampler2(fea)))
        # nearest interplote
        fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # sub pixel
        fea = self.lrelu(self.upconv4(self.upsampler4(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out