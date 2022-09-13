import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Parameter

 # type: GN, BN, IN, LN
 # Specifically, GN with GN GN_2 GN_4 ...
def getNormModuleND(dimension = 3, norm_type = 'GN'):

    if norm_type is None:
        return None, None

    norm_ND = None
    params = []

    if 'GN' in norm_type:
        norm_ND = nn.GroupNorm
        if norm_type == 'GN':
            params.append(2) # default number of groups in GN
        else:
            eles = norm_type.split('_')
            params.append(int(eles[1]))
    else:
        if dimension == 1:
            norm_ND = nn.BatchNorm1d
        elif dimension == 2:
            norm_ND = nn.BatchNorm2d
        elif dimension == 3:
            norm_ND = nn.BatchNorm3d
    
    return norm_ND, params

class convNormAct(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, kernel_size, stride, padding, p_mode='same'):

        super(convNormAct, self).__init__()

        norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
        if p_mode == 'same':
            self.conv =  nn.Conv3d(in_cs, out_cs, kernel_size, stride, padding, bias = False)
        if p_mode == 'sym':
            self.conv =  nn.Conv3d(in_cs, out_cs, kernel_size, stride, 0, bias = False)

        if norm_type is not None:
            self.norm = norm_3d(*(norm_params + [out_cs]))
        else:
            self.norm = nn.Sequential()
        
        self.relu = nn.ReLU(inplace=True)
        self.p_mode = p_mode
        self.padding = padding

    def forward(self, x):
        
        if self.p_mode == 'sym' and self.padding == 1:
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        
        return x

class unetResBlockV1(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, downsample):

        super(unetResBlockV1, self).__init__()

        stride = 2 if downsample else 1

        if in_cs != out_cs or stride != 1:
            norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
            if norm_type is not None:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_cs, out_cs, 1, stride, 0, bias = False),
                    norm_3d(*(norm_params + [out_cs])),
                )
            else:
                self.shortcut = nn.Conv3d(in_cs, out_cs, 1, stride, 0, bias = False)
        else:
            self.shortcut = nn.Sequential() 

        self.layer1 = convNormAct(in_cs, out_cs, norm_type, 3, stride, 1, is_act = True)
        self.layer2 = convNormAct(out_cs, out_cs, norm_type, 3, 1, 1, is_act = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        identity = x

        x = self.layer2(self.layer1(x))
        x = x + self.shortcut(identity)
        x = self.relu(x)
        
        return x

class unetVggBlock(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, p_mode='same', num_convs = 2): 

        super(unetVggBlock, self).__init__()

        convs = []
        convs.append(convNormAct(in_cs, out_cs, norm_type, 3, 1, 1, p_mode))
        for _ in range(num_convs-1):
            convs.append(convNormAct(out_cs, out_cs, norm_type, 3, 1, 1, p_mode))

        self.layer = nn.Sequential(*convs)

    def forward(self, x):

        return self.layer(x)


class convMultiN(nn.Module):

    def __init__(self, in_cs, norm_type, kernel_size, stride, dilation):

        super(convMultiN, self).__init__()

        self.conv1 = nn.Conv3d(in_cs   , in_cs//2, kernel_size, stride, dilation, dilation, bias = False)
        self.conv2 = nn.Conv3d(in_cs//2, in_cs//4, kernel_size, stride, dilation, dilation, bias = False)
        self.conv3 = nn.Conv3d(in_cs//4, in_cs//8, kernel_size, stride, dilation, dilation, bias = False)
        self.conv4 = nn.Conv3d(in_cs//8, in_cs//8, kernel_size, stride, dilation, dilation, bias = False)
        
        norm_3d, norm_params = getNormModuleND(dimension = 3, norm_type = norm_type)
        
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_cs*2, in_cs*2, 1, 1, 0, bias = False),
            norm_3d(*(norm_params + [in_cs*2])),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x1):
        
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)

        x = torch.cat([x1,x2,x3,x4,x5],dim=1)

        x = self.final_conv(x)

        return x

class unetVggBlockMultiN(nn.Module):

    def __init__(self, in_cs, out_cs, norm_type, padding=1, dilation=1, p_mode='same'):

        super(unetVggBlockMultiN, self).__init__()

        self.conv1 = convNormAct(in_cs, out_cs // 2, norm_type, 3, 1, padding, p_mode=p_mode)
        self.conv2 = convMultiN(out_cs // 2, norm_type, 3, 1, dilation=dilation)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x
        
class unetConvBlock(nn.Module):
    
    '''
    Three types of convolutions are supported.
    vgg: with control of number of convolution (num_convs, default 2) in one vgg conv layer
    resV1: resnet with post activation
    resV2: resnet with pre activation
    '''

    def __init__(self, in_cs, out_cs, norm_type, num_convs = 2, conv_type = 'vgg', p_mode = 'same'): 

        super(unetConvBlock, self).__init__()

        if conv_type == 'vgg':
            self.layer = unetVggBlock(in_cs, out_cs, norm_type, p_mode, num_convs)
        elif conv_type == 'resV1':
            self.layer = unetResBlockV1(in_cs, out_cs, norm_type, downsample=False)
        elif conv_type == 'vgg_multi_d':
            self.layer = unetVggBlockMultiN(in_cs, out_cs, norm_type, padding=1, dilation=2, p_mode=p_mode)
        elif conv_type == 'vgg_multi_n':
            self.layer = unetVggBlockMultiN(in_cs, out_cs, norm_type, padding=1, dilation=1, p_mode=p_mode)
            
    def forward(self, x):

        return self.layer(x)

class unetDownSample(nn.Module):

    def __init__(self, channels, down_type = 'conv', p_mode = 'same', norm_type = None):

        super(unetDownSample, self).__init__()

        if down_type == 'conv':
            if p_mode == 'same':
                self.down = nn.Conv3d(channels, channels, 3, 2, 1, bias = False)
            elif p_mode == 'sym':
                self.down = nn.Conv3d(channels, channels, 3, 2, 0, bias = False)
        if down_type == 'maxpool':
            if p_mode == 'same':
                self.down = nn.MaxPool3d(kernel_size = 2, padding = 1)
            elif p_mode == 'sym':
                self.down = nn.MaxPool3d(kernel_size = 2, padding = 0)

        self.p_mode = p_mode

    def forward(self, x):
        
        if self.p_mode == 'sym':
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')

        return self.down(x)

class unetUpConv(nn.Module):

    def __init__(self, in_cs, out_cs, is_deconv, upsample_type, norm_type):

        super(unetUpConv, self).__init__()

        if is_deconv:
            if norm_type is not None:
                self.up = nn.Sequential(
                    convNormAct(in_cs,out_cs,norm_type,1,1,0),
                    nn.ConvTranspose3d(out_cs, out_cs, kernel_size=2, stride=2, padding=0, bias=False)            
                )
            else:
                self.up = nn.ConvTranspose3d(in_cs, out_cs, kernel_size=2, stride=2, padding=0, bias=False)
        else:
            if norm_type is not None:
                if upsample_type == 'nearest':
                    self.up = nn.Sequential(
                        convNormAct(in_cs,out_cs,norm_type,1,1,0),
                        nn.Upsample(scale_factor=2),   
                    )         
                else:
                    self.up = nn.Sequential(
                        convNormAct(in_cs,out_cs,norm_type,1,1,0),
                        nn.Upsample(scale_factor=2, mode = upsample_type, align_corners = True),   
                    )   
            else:
                if upsample_type == 'nearest':
                    self.up = nn.Sequential(
                        nn.Conv3d(in_cs, out_cs, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2),   
                    )         
                else:
                    self.up = nn.Sequential(
                        nn.Conv3d(in_cs, out_cs, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Upsample(scale_factor=2, mode = upsample_type, align_corners = True),   
                    )

    
    def forward(self, x):

        return self.up(x)

class unetPadCat(nn.Module):

    def __init__(self):

        super(unetPadCat, self).__init__()

    def forward(self, leftIn, rightIn):
        
        rShape = rightIn.size()
        lShape = leftIn.size()
        padding = (lShape[4]-rShape[4], 0, lShape[3]-rShape[3], 0, lShape[2]-rShape[2], 0)

        pad = torch.nn.ConstantPad3d(padding, 0)
        rightIn = pad(rightIn)

        return torch.cat([leftIn, rightIn], 1)

class unetUpPadCatConv(nn.Module):
    
    def __init__(self, left_cs, right_cs, is_deconv, norm_type, p_mode='same', conv_type='vgg', upsample_type='nearest'):

        super(unetUpPadCatConv, self).__init__()

        self.up = unetUpConv(right_cs, right_cs // 2, is_deconv, upsample_type, norm_type)
        self.padCat = unetPadCat()
        self.conv = unetConvBlock(right_cs, left_cs, norm_type, conv_type = conv_type, p_mode=p_mode)
    
    def forward(self, left_x, right_x):

        right_x = self.up(right_x)
        x = self.padCat(left_x, right_x)
        del left_x, right_x

        return self.conv(x)