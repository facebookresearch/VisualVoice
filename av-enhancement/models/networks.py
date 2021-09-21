#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d, kernel_size=4):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv])

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, outermost=False):
        super(up_conv,self).__init__()
        if not outermost:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(2.,1.)),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
                )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(2.,1.)),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.Sigmoid()
                )
    def forward(self,x):
        x = self.up(x)
        return x
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        #customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) #features before pooling

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        else:
            return x

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.view(x.size(0), -1, 1, 1)
        else:
            return x.view(x.size(0), -1, 1, 1)

    def forward_multiframe(self, x, pool=True):
        (B, T, C, H, W) = x.size()
        x = x.contiguous()
        x = x.view(B * T, C, H, W)
        x = self.feature_extraction(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x 

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
        
        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.view(x.size(0), -1, 1, 1)
        else:
            return x.view(x.size(0), -1, 1, 1)
        return x

class VisualVoiceUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, audioVisual_feature_dim=1152):
        super(VisualVoiceUNet, self).__init__()
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = conv_block(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = conv_block(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer8 = conv_block(ngf * 8, ngf * 8)
        self.frequency_pool = nn.MaxPool2d([2,1])
        self.audionet_upconvlayer1 = up_conv(audioVisual_feature_dim, ngf * 8)
        self.audionet_upconvlayer2 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer5 = up_conv(ngf * 16, ngf *4)
        self.audionet_upconvlayer6 = up_conv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer8 = unet_upconv(ngf * 2, output_nc, True)
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, audio_mix_spec, visual_feat, activation='Sigmoid'):
        audio_conv1feature = self.audionet_convlayer1(audio_mix_spec[:,:,:-1,:])
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv3feature = self.frequency_pool(audio_conv3feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv4feature = self.frequency_pool(audio_conv4feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv5feature = self.frequency_pool(audio_conv5feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv6feature = self.frequency_pool(audio_conv6feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = self.frequency_pool(audio_conv7feature)
        audio_conv8feature = self.audionet_convlayer8(audio_conv7feature)
        audio_conv8feature = self.frequency_pool(audio_conv8feature)
        audioVisual_feature = torch.cat((visual_feat, audio_conv8feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv7feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv6feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv5feature), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv4feature), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv3feature), dim=1))
        audio_upconv7feature = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv2feature), dim=1))
        prediction = self.audionet_upconvlayer8(torch.cat((audio_upconv7feature, audio_conv1feature), dim=1))
        if activation == 'Sigmoid':
            pred_mask = self.Sigmoid(prediction)
        elif activation == 'Tanh':
            pred_mask = self.Tanh(prediction)
        return pred_mask