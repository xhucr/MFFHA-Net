import torch
import torch.nn as nn
from torch.nn import functional as F


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


# # SE block add to U-net
def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)


# # CBAM Convolutional block attention module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,strides=None,padding=0,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,strides,padding,ceil_mode,count_include_pad,divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool
        

def downsample_soft():
    return SoftPooling2D(2, 2)
    
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'sp']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        #self.relu = nn.ReLU(True)

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'sp':
                sf_pool_f = SoftPooling2D((x.size(2), x.size(3)), (x.size(2), x.size(3)))
                sf_pool = sf_pool_f(x)
                channel_att_raw = self.mlp(sf_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scalecoe = F.sigmoid(channel_att_sum)
        channel_att_sum = channel_att_sum.reshape(channel_att_sum.shape[0], 4, 4)
        avg_weight = torch.mean(channel_att_sum, dim=2).unsqueeze(2)
        avg_weight = avg_weight.expand(channel_att_sum.shape[0], 4, 4).reshape(channel_att_sum.shape[0], 16)
        #scale =nn.Sigmoid(avg_weight)
        scale = F.sigmoid(avg_weight).unsqueeze(2).unsqueeze(3).expand_as(x)

        # u = torch.abs(channel_att_sum)
        # with torch.no_grad():
        #     x1 = (u+channel_att_sum)*0.5
        #     x_m =torch.max(u,1)[0]
        #     x_max = torch.max(x1,1)[0]
        #     x_avg = torch.mean(x1,1)
        #     limit_0 = 1e-7
        #     sc = ((x_max+x_avg)/(x_max-x_avg + limit_0)+(x_max/(x_avg+limit_0)))*0.5+1
        #     sc = sc.unsqueeze(1).expand_as(channel_att_sum)
        #     x_max = x_max.unsqueeze(1).expand_as(channel_att_sum)
        #     x_m = x_m.unsqueeze(1).expand_as(channel_att_sum)
        #     m = (x1/(x_max+limit_0))*(x1/(x_max+limit_0))*sc + (channel_att_sum/(x_m+limit_0))*(u/(x_m+limit_0))
        #     scale = (1/(1+torch.exp(-m))).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale, scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)    # broadcasting
        # spa_scale = scale.expand_as(x)
        # print(spa_scale.shape)
        return x * scale, scale

class MapF(nn.Module):
    def __init__(self,M_min=0.0000001,padding=0,ker=(2,2),is_up=True,add=2):
        super(MapF,self).__init__()
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
        with torch.no_grad():
            return F.sigmoid(x)

class SpatialAtten(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1):
        super(SpatialAtten, self).__init__()
        self.conv1 = BasicConv(in_size, out_size, kernel_size, stride=stride,
                               padding=(kernel_size-1) // 2, relu=True)#16431
        self.conv2 = BasicConv(out_size, out_size, kernel_size=1, stride=stride,
                               padding=0, relu=True, bn=False)

    def forward(self, x):
        residual = x
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        # spatial_att = F.sigmoid(x_out).unsqueeze(4).permute(0, 1, 4, 2, 3)
        u3 = x_out.size(2)
        u2 = x_out.size(3)
        spatial_att = MapF(ker=(u3, u2))(x_out).unsqueeze(4).permute(0, 1, 4, 2, 3)
        spatial_att = spatial_att.expand(spatial_att.shape[0], 4, 4, spatial_att.shape[3], spatial_att.shape[4]).reshape(
                                        spatial_att.shape[0], 16, spatial_att.shape[3], spatial_att.shape[4])
        x_out = residual * spatial_att

        x_out = x_out + residual

        return x_out, spatial_att

class Scale_atten_block_softpool(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'sp'], no_spatial=False):
        super(Scale_atten_block_softpool, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialAtten(gate_channels, gate_channels //reduction_ratio)

    def forward(self, x):
        x_out, ca_atten = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, sa_atten = self.SpatialGate(x_out)

        return x_out, ca_atten, sa_atten


class scale_atten_convblock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, use_cbam=True, no_spatial=False, drop_out=False):
        super(scale_atten_convblock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(in_size, out_size)
        self.bn3 = nn.BatchNorm2d(out_size)

        if use_cbam:
            self.cbam = Scale_atten_block_softpool(in_size, reduction_ratio=4, no_spatial=self.no_spatial)  # out_size
            #self.cbam = Scale_atten_block_softpool(in_size, reduction_ratio=8, no_spatial=self.no_spatial)  # out_size
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out, scale_c_atten, scale_s_atten = self.cbam(x)

            # scale_c_atten = nn.Sigmoid()(scale_c_atten)
            # scale_s_atten = nn.Sigmoid()(scale_s_atten)
            # scale_atten = channel_atten_c * spatial_atten_s

        # scale_max = torch.argmax(scale_atten, dim=1, keepdim=True)
        # scale_max_soft = get_soft_label(input_tensor=scale_max, num_class=8)
        # scale_max_soft = scale_max_soft.permute(0, 3, 1, 2)
        # scale_atten_soft = scale_atten * scale_max_soft

        out = out + residual
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out