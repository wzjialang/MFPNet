
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from seresnet50 import se_resnet50

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(BasicConvBlock, self).__init__()
        
        if out_channels is None:
                out_channels = in_channels
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d( in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d( out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )        
    
    def forward(self,x):
        x=self.conv(x)
        return x

class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

# Channel Attention Algorithm (CAA)
class CAA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg','max']):
        super(CAA, self).__init__()
        self.num=1
        self.gate_channels = gate_channels

        self.conv_fc1 = nn.Sequential(
            nn.Conv2d(in_channels=gate_channels, out_channels=gate_channels//reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=gate_channels//reduction_ratio, out_channels=gate_channels, kernel_size=1, bias=False),
        )
        self.conv_fc2 = nn.Sequential(
            nn.Conv2d(in_channels=gate_channels, out_channels=gate_channels//reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=gate_channels//reduction_ratio, out_channels=gate_channels, kernel_size=1, bias=False),
        )
        self.conv= nn.Sequential(
            nn.Conv2d(gate_channels,gate_channels,kernel_size=(2,1),bias=False),
            nn.Sigmoid()
        )

        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        b,c,h,w=x.size()

        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.conv_fc1(avg_pool).view(b,c,self.num,self.num)
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.conv_fc2(max_pool).view(b,c,self.num,self.num)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = torch.cat([channel_att_sum, channel_att_raw],dim=2)
        
        channel_weight=self.conv(channel_att_sum)
        scale = nn.functional.upsample_bilinear(channel_weight, [h, w])

        return x * scale

# Multidirectional Adaptive Feature Fusion Module (MAFFM)
class MAFFM(nn.Module):
    def __init__(self, num_channels, conv_channels):
        super(MAFFM, self).__init__()

        # Conv layers
        self.conv5 = BasicConvBlock(num_channels)
        self.conv4 = BasicConvBlock(num_channels)
        self.conv3 = BasicConvBlock(num_channels)
        self.conv2 = BasicConvBlock(num_channels)
        self.conv1 = BasicConvBlock(num_channels)

        self.conv5_1 = BasicConvBlock(num_channels)
        self.conv4_1 = BasicConvBlock(num_channels)
        self.conv3_1 = BasicConvBlock(num_channels)
        self.conv2_1 = BasicConvBlock(num_channels)
        self.conv1_1 = BasicConvBlock(num_channels)

        self.conv1_down = BasicConvBlock(num_channels)
        self.conv2_down = BasicConvBlock(num_channels)
        self.conv3_down = BasicConvBlock(num_channels)
        self.conv4_down = BasicConvBlock(num_channels)
        self.conv5_down = BasicConvBlock(num_channels)

        # Feature scaling layers
        self.p4_upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p1_upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p3_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Channel compression layers
        self.p5_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[4], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.p4_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.p3_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.p2_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.p1_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

        # CAA
        self.csac_p1_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p2_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p3_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p4_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p5_0=CAA(num_channels,reduction_ratio=1)

        self.csac_p1_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p2_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p3_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p4_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p5_1=CAA(num_channels,reduction_ratio=1)

        self.csac_p1_2=CAA(num_channels,reduction_ratio=1)
        self.csac_p2_2=CAA(num_channels,reduction_ratio=1)
        self.csac_p3_2=CAA(num_channels,reduction_ratio=1)
        self.csac_p4_2=CAA(num_channels,reduction_ratio=1)


        self.csac_p51_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p41_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p31_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p21_0=CAA(num_channels,reduction_ratio=1)
        
        self.csac_p51_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p41_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p31_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p21_1=CAA(num_channels,reduction_ratio=1)


        self.csac_p52_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p42_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p32_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p22_0=CAA(num_channels,reduction_ratio=1)
        self.csac_p12_0=CAA(num_channels,reduction_ratio=1)

        self.csac_p52_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p42_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p32_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p22_1=CAA(num_channels,reduction_ratio=1)
        self.csac_p12_1=CAA(num_channels,reduction_ratio=1)

        self.csac_p42_2=CAA(num_channels,reduction_ratio=1)
        self.csac_p32_2=CAA(num_channels,reduction_ratio=1)
        self.csac_p22_2=CAA(num_channels,reduction_ratio=1)
        self.csac_p12_2=CAA(num_channels,reduction_ratio=1)

    def forward(self, inputs):
        p1_pre, p2_pre, p3_pre, p4_pre, p5_pre, p1_now, p2_now, p3_now, p4_now, p5_now = inputs

        p1_in_pre = self.p1_down_channel(p1_pre)
        p1_in_now = self.p1_down_channel(p1_now)

        p2_in_pre = self.p2_down_channel(p2_pre)
        p2_in_now = self.p2_down_channel(p2_now)

        p3_in_pre = self.p3_down_channel(p3_pre)
        p3_in_now = self.p3_down_channel(p3_now)

        p4_in_pre = self.p4_down_channel(p4_pre)
        p4_in_now = self.p4_down_channel(p4_now)

        p5_in_pre = self.p5_down_channel(p5_pre)
        p5_in_now = self.p5_down_channel(p5_now)
        
        # Multidirectional Fusion Pathway (MFP) + Adaptive Weighted Fusion (AWF)
        # Up
        p5_in=self.conv5(self.csac_p5_0(p5_in_now)+self.csac_p5_1(p5_in_pre))
        p4_in=self.conv4(self.csac_p4_0(p4_in_now)+self.csac_p4_1(p4_in_pre)+self.csac_p4_2(self.p4_upsample(p5_in)))
        p3_in=self.conv3(self.csac_p3_0(p3_in_now)+self.csac_p3_1(p3_in_pre)+self.csac_p3_2(self.p3_upsample(p4_in)))
        p2_in=self.conv2(self.csac_p2_0(p2_in_now)+self.csac_p2_1(p2_in_pre)+self.csac_p2_2(self.p2_upsample(p3_in)))
        p1_in=self.conv1(self.csac_p1_0(p1_in_now)+self.csac_p1_1(p1_in_pre)+self.csac_p1_2(self.p1_upsample(p2_in)))
        # Down
        p1_1 = self.conv1_down(p1_in)
        p2_1 = self.conv2_down(self.csac_p21_0(p2_in) + self.csac_p21_1(self.p2_downsample(p1_1)))
        p3_1 = self.conv3_down(self.csac_p31_0(p3_in) + self.csac_p31_1(self.p3_downsample(p2_1)))
        p4_1 = self.conv4_down(self.csac_p41_0(p4_in) + self.csac_p41_1(self.p4_downsample(p3_1)))
        p5_1 = self.conv5_down(self.csac_p51_0(p5_in) + self.csac_p51_1(self.p5_downsample(p4_1)))
        # Up
        p5_2 = self.conv5_1(self.csac_p52_0(p5_in) + self.csac_p52_1(p5_1))
        p4_2 = self.conv4_1(self.csac_p42_0(p4_in) + self.csac_p42_1(p4_1)+self.csac_p42_2(self.p4_upsample_1(p5_2)))
        p3_2 = self.conv3_1(self.csac_p32_0(p3_in) + self.csac_p32_1(p3_1)+self.csac_p32_2(self.p3_upsample_1(p4_2)))
        p2_2 = self.conv2_1(self.csac_p22_0(p2_in) + self.csac_p22_1(p2_1)+self.csac_p22_2(self.p2_upsample_1(p3_2)))
        p1_2 = self.conv1_1(self.csac_p12_0(p1_in) + self.csac_p12_1(p1_1)+self.csac_p12_2(self.p1_upsample_1(p2_2)))

        return p1_2

class DECODER(nn.Module):
    def __init__(self, in_ch, classes):
        super(DECODER, self).__init__()
        self.conv1 = nn.Conv2d(
            in_ch, in_ch//4, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(
            in_ch//4, in_ch//8, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(
            in_ch//8, classes*4, kernel_size=1)

        self.ps3 = nn.PixelShuffle(2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.ps3(x)
        
        return x

class MFPNET(nn.Module):
    def __init__(self, classes):
        super(MFPNET, self).__init__()

        self.se_resnet50 = se_resnet50(pretrained=True, strides = (1,2,2,2))
        self.stage1 = nn.Sequential(self.se_resnet50.conv1, self.se_resnet50.bn1, self.se_resnet50.relu)
        self.stage2 = nn.Sequential(self.se_resnet50.maxpool, self.se_resnet50.layer1)
        self.stage3 = nn.Sequential(self.se_resnet50.layer2)
        self.stage4 = nn.Sequential(self.se_resnet50.layer3)
        self.stage5 = nn.Sequential(self.se_resnet50.layer4)
        
        self.maffm=MAFFM(256,[64,256,512,1024,2048])
        self.dec = DECODER(256, classes)

    def encoder(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return x1, x2, x3, x4, x5

    def forward(self, x_prev, x_now):
        p1_t1, p2_t1, p3_t1, p4_t1, p5_t1 = self.encoder(x_prev)
        p1_t2, p2_t2, p3_t2, p4_t2, p5_t2 = self.encoder(x_now)
        features_t1_t2 = (p1_t1, p2_t1, p3_t1, p4_t1, p5_t1, p1_t2, p2_t2, p3_t2, p4_t2, p5_t2)

        x_fuse=self.maffm(features_t1_t2)
        dis_map=self.dec(x_fuse)
        result = torch.argmax(dis_map, dim = 1, keepdim = True)

        return dis_map, result

if __name__ == "__main__":
    model = MFPNET(classes = 2)
    
    # # Example for using Perceptual Similarity Module
    # from vgg import Vgg19

    # criterion_perceptual = nn.MSELoss()
    # criterion_perceptual.cuda()
    # vgg= Vgg19().cuda()

    # for epoch in range(300):
    #     for i, (data_prev, data_now, label) in enumerate(loader_train, 0):
    #         model.train()
    #         model.zero_grad()
    #         optimizer.zero_grad()
    #         img_prev_train, img_now_train, label_train = data_prev.cuda(), data_now.cuda(), label.cuda()

    #         out_train1, _ = model(img_prev_train, img_now_train)
            
    #         # Perceptual Similarity Module (PSM) 
    #         out_train_softmax2d = F.softmax(out_train1,dim=1)
    #         an_change = out_train_softmax2d[:,1,:,:].unsqueeze(1).expand_as(img_prev_train)
    #         an_unchange = out_train_softmax2d[:,0,:,:].unsqueeze(1).expand_as(img_prev_train)
    #         label_change = label_train.expand_as(img_prev_train).type(torch.FloatTensor).cuda()
    #         label_unchange = 1-label_change
    #         an_change = an_change*label_change
    #         an_unchange = an_unchange*(1-label_change)

    #         an_change_feature = vgg(an_change)
    #         gt_feature = vgg(label_change)   
    #         an_unchange_feature = vgg(an_unchange)
    #         gt_feature_unchange = vgg(label_unchange)
            
    #         perceptual_loss_change = criterion_perceptual(an_change_feature[0], gt_feature[0])
    #         perceptual_loss_unchange = criterion_perceptual(an_unchange_feature[0], gt_feature_unchange[0])
    #         perceptual_loss = perceptual_loss_change + perceptual_loss_unchange