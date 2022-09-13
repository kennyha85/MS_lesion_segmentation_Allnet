import torch
import torch.nn as nn

from models.moduleSep import unetConvBlock, unetDownSample, unetUpPadCatConv

class unetResV1Offset(nn.Module):
    def __init__(self,
        in_channels = 3,
        norm_type = 'BN',
        conv_type = 'resV1', 
    ):
        super(unetResV1Offset, self).__init__()
        
        filters = [32, 64, 128, 256, 512]
        # filters = [16, 32, 64, 128, 256]

        # initial convolution
        self.in_cs = in_channels
        self.init_conv = nn.Conv3d(in_channels, filters[0],3,1,1, bias=False)
        self.dropout = nn.Dropout3d(0.2, inplace=True)

        # downsampling
        self.conv1 = unetConvBlock(filters[0], filters[0], norm_type, conv_type=conv_type)
        self.down1 = unetDownSample(filters[0], down_type = 'conv', norm_type=norm_type)

        self.conv2 = unetConvBlock(filters[0], filters[1], norm_type, conv_type=conv_type)
        self.down2 = unetDownSample(filters[1], down_type = 'conv', norm_type=norm_type)
        
        self.conv3 = unetConvBlock(filters[1], filters[2], norm_type, conv_type=conv_type)
        self.down3 = unetDownSample(filters[2], down_type = 'conv', norm_type=norm_type)

        self.conv4 = unetConvBlock(filters[2], filters[3], norm_type, conv_type=conv_type)
        self.down4 = unetDownSample(filters[3], down_type = 'conv', norm_type=norm_type)
        
        self.center = nn.Sequential(
            unetConvBlock(filters[3], filters[4], norm_type, conv_type=conv_type),
            unetConvBlock(filters[4], filters[4], norm_type, conv_type=conv_type)
        )

        # upsampling
        self.up_concat4 = unetUpPadCatConv(filters[3], filters[4], False, norm_type, conv_type=conv_type)
        self.up_concat3 = unetUpPadCatConv(filters[2], filters[3], False, norm_type, conv_type=conv_type)
        self.up_concat2 = unetUpPadCatConv(filters[1], filters[2], False, norm_type, conv_type=conv_type)
        self.up_concat1 = unetUpPadCatConv(filters[0], filters[1], False, norm_type, conv_type=conv_type)

        self.final = nn.Sequential(
            nn.Conv3d(filters[0], 1,1,bias=False),
        )     

        self.up_ofx = unetUpPadCatConv(filters[0], filters[1], False, norm_type, conv_type=conv_type)
        self.final_ofx = nn.Sequential(
            nn.Conv3d(filters[0], 1,1,bias=False),        
        ) 
        
        self.up_ofy = unetUpPadCatConv(filters[0], filters[1], False, norm_type, conv_type=conv_type)
        self.final_ofy = nn.Sequential(
            nn.Conv3d(filters[0], 1,1,bias=False),        
        )

        self.up_ofz = unetUpPadCatConv(filters[0], filters[1], False, norm_type, conv_type=conv_type)
        self.final_ofz = nn.Sequential(
            nn.Conv3d(filters[0], 1,1,bias=False),        
        )

        self.up_vwt = unetUpPadCatConv(filters[0], filters[1], False, norm_type, conv_type=conv_type)
        self.final_vwt = nn.Sequential(
            nn.Conv3d(filters[0], 1,1,bias=False),
            nn.ReLU(inplace=True),        
        )


    def forward(self, x):

        x = self.init_conv(x[:,:self.in_cs ,:,:,:])
        x = self.dropout(x)

        conv1 = self.conv1(x)
        x = self.down1(conv1)
        conv2 = self.conv2(x)
        x = self.down2(conv2)
        conv3 = self.conv3(x)
        x = self.down3(conv3)
        conv4 = self.conv4(x)
        x = self.down4(conv4)
        x = self.center(x)

        x = self.up_concat4(conv4, x)
        del conv4
        x = self.up_concat3(conv3, x)
        del conv3
        x = self.up_concat2(conv2, x)
        del conv2

        v = self.up_concat1(conv1, x)
        v = self.final(v)

        ofx = self.up_ofx(conv1, x)
        ofx = self.final_ofx(ofx)
        
        ofy = self.up_ofy(conv1, x)
        ofy = self.final_ofy(ofy)
        
        ofz = self.up_ofz(conv1, x)
        ofz = self.final_ofz(ofz)
        
        vwt = self.up_vwt(conv1, x)
        vwt = self.final_vwt(vwt)

        return [v,[ofx,ofy,ofz,vwt]]