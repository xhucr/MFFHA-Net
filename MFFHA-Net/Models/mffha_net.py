#!/usr/bin/env python
# encoding: utf-8
"""
Ms RED network.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.layers.mdfm_sc_att import scale_atten_convblock
import queue
import numpy as np


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def downsample_soft():
    return SoftPooling2D(2, 2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool

class NSNPU(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(NSNPU, self).__init__()
        self.nsnpu = nn.Sequential(
                                nn.ELU(),
                                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.nsnpu(input)


class HSBlock(nn.Module):
    '''
    替代3x3卷积
    '''

    def __init__(self, in_ch, s=4):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1, self.s):
            if i == 1:
                channels = in_ch
                acc_channels = channels // 2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels = in_ch + acc_channels
                acc_channels = channels // 2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
        self.initialize_weights()

    #snp
    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            x0 = x[i]
            seq = self.module_list[i]
            y = seq(x0)
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]

class HSBlock_rfb(nn.Module):
    '''
    替代3x3卷积
    '''

    def __init__(self, in_ch, s=4, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock_rfb, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1, self.s):
            if i == 1:
                channels = in_ch
                acc_channels = channels // 2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels = in_ch + acc_channels
                acc_channels = channels // 2
            self.module_list.append(
                self.conv_bn_relu(in_ch=channels, out_ch=channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, bias=bias))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv_hs(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, #
                 bn=True, bias=False):
        super(BasicConv_hs, self).__init__()
        self.out_channels = out_planes
        self.conv = HSBlock_rfb(in_planes, s=4, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)#123
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None
        # self.relu = nn.LeakyReLU()

    def forward(self, x):

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB_hs(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB_hs, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                         relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                         dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            # 5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                         dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out


class RFB_hs_att(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB_hs_att, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                         relu=False),
            GA(2 * inter_planes)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                         dilation=visual + 1, relu=False),
            GA(2 * inter_planes)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                         dilation=2 * visual + 1, relu=False),
            GA(2 * inter_planes)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out

class ResEncoder_hs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder_hs, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = HSBlock(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            # nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            # nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out



class GA(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(GA, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.local_attention = local_att_test(ker=(14, 20))
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        # a = self.local_attention(x)
        cab = self.cab(x)
        out = sab + cab
        return out


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class dilation(nn.Module):
    def __init__(self, channel):
        super(dilation, self).__init__()
        self.channel = channel
        self.ratio_conv1 = nn.Conv2d(self.channel, self.channel, 3, 1, 1, 1)
        self.ratio_conv2 = nn.Conv2d(self.channel, self.channel, 3, 1, 3, 3)
        self.ratio_conv3 = nn.Conv2d(self.channel, self.channel, 3, 1, 5, 5)
        self.conv_link = nn.Conv2d(self.channel * 3, self.channel, 1, 1)
        self.Norm = nn.BatchNorm2d(self.channel)
        self.Relu = nn.ReLU(inplace=True)


    def forward(self, X):
        X1 = self.ratio_conv1(X)
        X2 = self.ratio_conv2(X)
        X3 = self.ratio_conv3(X)
        X = torch.cat((X1, X2, X3), 1)
        X = self.conv_link(X)
        X = self.Norm(X)
        X = self.Relu(X)
        return X


class NSNP_MEFM(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(NSNP_MEFM, self).__init__()
        self.level = level
        self.dim = [256, 128, 64, 32]
        self.inter_dim = self.dim[self.level]
        if level == 0:  # 256

            self.conv_down32 = nn.Sequential(
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1),
                nn.Conv2d(32,64,3,2,1)

            )
            self.conv_down64 = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.Conv2d(64, 128, 3, 2, 1)

            )
            self.conv_down96 = nn.Sequential(
                nn.BatchNorm2d(96),
                nn.LeakyReLU(0.1),
                nn.Conv2d(96, 192, 3, 2, 1)

            )
            self.conv_down128= nn.Sequential(
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                nn.Conv2d(128, 256, 3, 2, 1)

            )
            self.conv_link64 = add_conv(128, 64, 1, 1)
            self.conv_link128 = add_conv(256, 128, 1, 1)
            self.conv_link256 = add_conv(512, 256, 1, 1)
            #self.conv_link384 = add_conv(768, 256, 1, 1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.ratio_32 = dilation(32)
            self.ratio_64 = dilation(64)
            self.ratio_128 = dilation(128)
            self.ratio_256 = dilation(256)
            self.upsample8 = nn.Upsample(size=(56, 80), mode='bilinear')
            self.upsample16 = nn.Upsample(size=(112, 160), mode='bilinear')
            self.upsample32 = nn.Upsample(size=(224, 320), mode='bilinear')
            self.Relu = nn.ReLU(inplace=False)
            self.conv_384_256 = nn.Conv2d(384, 256, 3, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        #self.weight_levels = nn.Conv2d(compress_c * 3, 4, kernel_size=1, stride=1, padding=0)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        if self.level == 0:
            # xlevel0    #256
            level_0_R = x_level_0
            level_0_ratio_rec = self.ratio_256(level_0_R)
            level_0_R = level_0_R + level_0_ratio_rec

            # xlevel1    #128
            level_1_R = x_level_1
            level_1_ratio_rec = self.ratio_128(x_level_1)
            level_1_R = level_1_R + level_1_ratio_rec

            # xlevel2    #64
            level_2_R = x_level_2
            level_2_ratio_rec = self.ratio_64(x_level_2)
            level_2_R = level_2_R + level_2_ratio_rec

            # xlevel3    #32
            level_3_R = x_level_3
            level_3_ratio_rec = self.ratio_32(x_level_3)
            level_3_R = level_3_R + level_3_ratio_rec

            #
            level_down = self.conv_down32(level_3_R)
            level_2_conv1 = torch.cat((level_2_R, level_down), 1)
            level_2_conv1 = self.conv_link64(level_2_conv1)
            level_2_conv1_up = self.upsample32(level_2_conv1)

            level_down = self.conv_down64(level_2_conv1)
            level_1_cat1 = torch.cat((level_1_R, level_down), 1)
            level_1_conv1 = self.conv_link128(level_1_cat1)
            level_1_conv1_up = self.upsample16(level_1_conv1)

            level_down = self.conv_down128(level_1_conv1)
            level_0_cat1 = torch.cat((level_0_R, level_down), 1)
            level_0_conv1 = self.conv_link256(level_0_cat1)
            level_0_conv1_up = self.upsample8(level_0_conv1)

            level_3_cat2 = torch.cat((level_3_R, level_2_conv1_up), 1)
            level_3_cat2_conv = self.conv_down96(level_3_cat2)
            level_2_cat2 = torch.cat((level_2_conv1, level_1_conv1_up), 1)
            level_1_cat2 = torch.cat((level_1_conv1, level_0_conv1_up), 1)
            level_1_cat2_up = self.upsample16(level_1_cat2)

            level_2_cat3 = torch.cat((level_2_cat2, level_3_cat2_conv), 1)
            level_2_cat3_pool = self.pool(level_2_cat3)

            level_2_l_add = level_2_cat3 + level_1_cat2_up
            level_1_l_add = level_2_cat3_pool + level_1_cat2

            level_2_final = self.conv_384_256(level_2_l_add)
            level_1_final = self.conv_384_256(level_1_l_add)
            level_2_next = self.pool(level_2_final)
            level_1_next = self.pool(level_1_final)
            level_2_next_next = self.pool(level_2_next)
            level_0_conv1 = self.Relu(level_0_conv1)
            level_1_next = self.Relu(level_1_next)
            level_2_next_next = self.Relu(level_2_next_next)

        level_0_weight_v = self.weight_level_0(level_0_conv1)
        level_1_weight_v = self.weight_level_1(level_1_next)
        level_2_weight_v = self.weight_level_2(level_2_next_next)
        # level_3_weight_v = self.weight_level_3(level_3_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_conv1 * levels_weight[:, 0:1, :, :] + \
                            level_1_next * levels_weight[:, 1:2, :, :] + \
                            level_2_next_next * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class LA(nn.Module):
    def __init__(self,ch):
        super(LA, self).__init__()
        self.pool2 = nn.MaxPool2d(2, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool3 = nn.AvgPool2d(3, 1, 1)
        # self.pool5 = nn.MaxPool2d(5, 1, 2)
        self.droupout = nn.Dropout(0.4)
        self.sigmoid = map()

    def forward(self, X):
        x_p_1 = self.pool2(self.pool1(X))
        x_p_2 = self.pool3(X)
        x_p_3 = self.droupout(self.pool2(self.pool1(x_p_1)))
        x_p_1 = self.droupout(x_p_1)
        x_p = x_p_1 + x_p_2 + x_p_3 + X
        x_p = self.sigmoid(x_p)
        return x_p


class WinSpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):  # 512-（14*20）    256-(28*40)     128-(56*80)     64-(112*160)    32-(224*320)
        super(WinSpatialAttentionBlock, self).__init__()
        self.channel = in_channels
        # self.win = [2,4,8,16,32]
        # self.win_n = self.win[512//in_channels]
        self.gate1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.gate2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.query = nn.Sequential(
            # nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            # nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gate_link = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, win_n=2, type=1, shift_h=0, shift_w=0):  # win_n = 2,4,8,16
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        win_n_list = [2, 4, 8, 16]
        assert win_n in win_n_list, "ERROR WINDOW NUMBER"

        # 1
        if type == 1:
            x_0 = self.win(x, win_n)
            x_1 = self.shift(x, shift_h, shift_w, win_n)
            return (x_0 + x_1) / 2

        # 2
        if type == 2:
            pass

            x_0 = self.win(x, win_n)
            x_1 = self.shift(x, shift_h, shift_w, win_n)
            x_r = (x_0 + x_1) / 2
            win_n = win_n // 2
            shift_h = shift_h * 2
            shift_w = shift_w * 2
            while win_n != 1:
                x_0 = self.win(x, win_n)
                x_1 = self.shift(x, shift_h, shift_w, win_n)
                x_r = (x_0 + x_1) / 2 * self.gate2 + x_r * self.gate1
                shift_h = shift_h * 2
                shift_w = shift_w * 2
                win_n = win_n // 2
            return x_r

    def win(self, x, win_n):
        B, C, H, W = x.size()
        H_f, W_f = H // win_n, W // win_n
        i = 0
        x0 = []
        while i < win_n:
            j = 0
            while j < win_n:
                x0.append(x[:, :, i * H_f:(i + 1) * H_f, j * W_f:(j + 1) * W_f])
                j = j + 1
            i = i + 1
        while i < win_n ** 2:
            x0[i] = self.att(x0[i])
            i = i + 1
        i = 0
        x1 = []
        while i < win_n ** 2:
            x_0 = x0[i]
            i = i + 1
            j = 0
            while j < win_n - 1:
                x_0 = torch.cat((x_0, x0[i]), 3)
                i = i + 1
                j = j + 1
            x1.append(x_0)
        x_0 = x1[0]
        i = 1
        while i < win_n:
            x_0 = torch.cat((x_0, x1[i]), 2)
            i = i + 1
        return x_0

    def shift(self, x, shift_h, shift_w, win_n):
        x0 = []
        x_r = []
        B, C, H, W = x.size()
        H_f, W_f = H // win_n, W // win_n
        if shift_h == 0:
            shift_h = H_f // 2
        if shift_w == 0:
            shift_w = W_f // 2
        i = 0
        while i < win_n - 1:
            j = 0
            while j < win_n - 1:
                x0.append(x[:, :, shift_h + i * H_f:shift_h + (i + 1) * H_f, shift_w + j * W_f:shift_w + (j + 1) * W_f])
                j = j + 1
            i = i + 1
        i = 0
        while i < (win_n - 1) ** 2:
            x0[i] = self.att(x0[i])
            i = i + 1

        i = 0
        x1 = []
        while i < (win_n - 1) ** 2:
            x_0 = x0[i]
            i = i + 1
            j = 0
            while j < win_n - 2:
                x_0 = torch.cat((x_0, x0[i]), 3)
                i = i + 1
                j = j + 1
            x1.append(x_0)
        x_0 = x1[0]
        i = 1
        while i < win_n - 1:
            x_0 = torch.cat((x_0, x1[i]), 2)
            i = i + 1

        x_r.append(x[:, :, 0:shift_h, 0:shift_w])
        x_r.append(x[:, :, 0:shift_h, shift_w:W - W_f + shift_w])
        x_r.append(x[:, :, 0:shift_h, W - W_f + shift_w:W])
        x_r.append(x[:, :, shift_h:H - H_f + shift_h, 0:shift_w])
        x_r.append(x[:, :, shift_h:H - H_f + shift_h, W - W_f + shift_w:W])
        x_r.append(x[:, :, H - H_f + shift_h:H, 0:shift_w])
        x_r.append(x[:, :, H - H_f + shift_h:H, shift_w:W - W_f + shift_w])
        x_r.append(x[:, :, H - H_f + shift_h:H, W - W_f + shift_w:W])
        i = 0
        while i < 8:
            x_r[i] = self.att(x_r[i])
            i = i + 1
        x_r1 = torch.cat((x_r[0], x_r[1], x_r[2]), 3)
        x_r2 = torch.cat((x_r[3], x_0, x_r[4]), 3)
        x_r3 = torch.cat((x_r[5], x_r[6], x_r[7]), 3)
        x_0 = torch.cat((x_r1, x_r2, x_r3), 2)
        return x_0

    def att(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class LG_A(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(LG_A, self).__init__()
        self.sab = WinSpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out

class map(nn.Module):
    def __init__(self,M_min=0.0000001,padding=0,ker=(2,2),is_up=True,add=2):
        super(map,self).__init__()
        #self.sigmoid = M_sigmoid()
        self.max_pool = nn.MaxPool2d(ker,padding=padding)
        self.avg_pool = nn.AvgPool2d(ker,padding=padding,count_include_pad=False)
        # self.softmax = nn.Softmax()
        self.up_samp = nn.Upsample(scale_factor=ker,mode='nearest')
        self.M = M_min
        self.is_up =is_up
        self.add = add
        self.ker = ker



    def forward(self,x):
        B, C, H, W = x.size()
        sc = H//self.ker[0]
        i=0
        add = 0.5
        while(i!=5):
            if sc == 1:
                break
            sc=sc//2
            i=i+1
            add=add*add
        x0 = torch.abs(x)
        x1 =(x0+x)*0.5
        m_x=self.max_pool(x0)
        x_max = self.max_pool(x1)
        x_avg = self.avg_pool(x1)
        a_ = ((x_max+x_avg)/(x_max-x_avg + self.M)+(x_max/(x_avg+self.M)))*0.5+self.add*add
        # a_ = a_**2
        if self.is_up==True:
            x_max = self.up_samp(x_max)
            x_m = self.up_samp(m_x)
            a_ = self.up_samp(a_)
        #a = (x/x_max)**2 * ((x_max+x_avg)/(x_max-x_avg + self.M)+(x_max/(x_avg+self.M)))/2
        # a= torch.div(x,(x_max+self.M))**2*(torch.div(x_max+x_avg,x_max-x_avg+self.M)+torch.div(x_max,x_avg+self.M))*0.5
        a = (x1/(x_max+self.M))*(x1/(x_max+self.M)) * a_ +(x/(x_m+self.M))*(x0/(x_m+self.M))
        # a = torch.pow(x / (x_max + self.M),2)  * a_

        x_wight = self.sigmoid(a)
        return x_wight

    def sigmoid(self, x):
        return 1/(1+torch.exp(-x))


class MFFHA_Net(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(MFFHA_Net, self).__init__()
        self.out_size = (224, 320)
        self.enc_input = ResEncoder_hs(channels, 32)
        self.encoder1 = RFB_hs(32, 64)
        self.encoder2 = RFB_hs(64, 128)
        self.encoder3 = RFB_hs(128, 256)
        self.encoder4 = RFB_hs_att(256, 512)
        self.downsample = downsample_soft()
        self.g_att = GA(512)


        self.decoder4 = RFB_hs_att(512, 256)
        self.decoder3 = RFB_hs(256, 128)
        self.decoder2 = RFB_hs(128, 64)
        self.decoder1 = RFB_hs(64, 32)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)

        self.assf_fusion4 = NSNP_MEFM(level=0)

        self.snpu4 = NSNPU(in_size=256, out_size=4, scale_factor=self.out_size)
        self.snpu3 = NSNPU(in_size=128, out_size=4, scale_factor=self.out_size)
        self.snpu2 = NSNPU(in_size=64, out_size=4, scale_factor=self.out_size)
        self.snpu1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        #self.scale_att = scale_atten_convblock(in_size=32, out_size=4)
        self.final = nn.Conv2d(4, classes, kernel_size=1)

        self.gamma_shift = nn.Parameter(torch.zeros(1))
        self.lg_att = LG_A(512)
        self.l_att = LA(512)
        self.gamma_local = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        enc_input = self.enc_input(x)  # [16, 3, 224, 320]-->[16, 32, 224, 320]
        down1 = self.downsample(enc_input)  # [16, 32, 112, 160]
        enc1 = self.encoder1(down1)  # [16, 64, 112, 160]
        down2 = self.downsample(enc1)  # [16, 64, 56, 80]
        enc2 = self.encoder2(down2)  # [16, 128, 56, 80]
        down3 = self.downsample(enc2)  # [16, 128, 28, 40]
        enc3 = self.encoder3(down3)  # [16, 256, 28, 40]
        fused1 = self.assf_fusion4(enc3, enc2, enc1, enc_input)
        down4 = self.downsample(fused1)  # [16, 256, 14, 20]
        input_feature = self.encoder4(down4)  # [16, 512, 14, 20]

        attention = self.ga(input_feature)
        local_attention = self.l_att(input_feature)
        local_attention = input_feature * self.gamma_local * local_attention
        shift_attenntion = self.lg_att(input_feature) * self.gamma_shift
        attention_fuse = input_feature + attention +  local_attention + shift_attenntion# [16, 512, 14, 18]

        up4 = self.deconv4(attention_fuse)  # [16, 256, 28, 36]
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)
        #

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        dsv4 = self.snpu4(dec4)  # [16, 4, 224, 320]
        dsv3 = self.snpu3(dec3)
        dsv2 = self.snpu2(dec2)
        dsv1 = self.snpu1(dec1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)  # [16, 16, 224, 320]
        out = self.scale_att(dsv_cat)  # [16, 4, 224, 300]
        out = self.final(out)
        final = F.sigmoid(out)
        return final
