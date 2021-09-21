#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import random
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from torch.autograd import Variable

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt

        #initialize model and criterions
        self.net_lipreading, self.net_identity, self.net_unet, self.net_classifier = nets

    def forward(self, input):
        audio_spec_A1 =  input['audio_spec_A1']
        audio_spec_A2 =  input['audio_spec_A2']
        audio_spec_B =  input['audio_spec_B']
        audio_mix_spec1 =  input['audio_spec_mix1']
        audio_mix_spec2 =  input['audio_spec_mix2']

        mouthroi_A1 = input['mouthroi_A1']
        mouthroi_A2 = input['mouthroi_A2']
        mouthroi_B = input['mouthroi_B']

        identityframe_A = input['frame_A']
        identityframe_B = input['frame_B']

        # calculate ground-truth masks
        gt_masks_A1_real = (audio_spec_A1[:,0,:,:] * audio_mix_spec1[:,0,:,:] + audio_spec_A1[:,1,:,:] * audio_mix_spec1[:,1,:,:]) / (audio_mix_spec1[:,0,:,:] * audio_mix_spec1[:,0,:,:] + audio_mix_spec1[:,1,:,:] * audio_mix_spec1[:,1,:,:] + 1e-30)
        gt_masks_A1_imag = (audio_spec_A1[:,1,:,:] * audio_mix_spec1[:,0,:,:] - audio_spec_A1[:,0,:,:] * audio_mix_spec1[:,1,:,:]) / (audio_mix_spec1[:,0,:,:] * audio_mix_spec1[:,0,:,:] + audio_mix_spec1[:,1,:,:] * audio_mix_spec1[:,1,:,:] + 1e-30)
        gt_masks_A1 = torch.cat((gt_masks_A1_real.unsqueeze(1), gt_masks_A1_imag.unsqueeze(1)), 1)
        gt_masks_A2_real = (audio_spec_A2[:,0,:,:] * audio_mix_spec2[:,0,:,:] + audio_spec_A2[:,1,:,:] * audio_mix_spec2[:,1,:,:]) / (audio_mix_spec2[:,0,:,:] * audio_mix_spec2[:,0,:,:] + audio_mix_spec2[:,1,:,:] * audio_mix_spec2[:,1,:,:] + 1e-30)
        gt_masks_A2_imag = (audio_spec_A2[:,1,:,:] * audio_mix_spec2[:,0,:,:] - audio_spec_A2[:,0,:,:] * audio_mix_spec2[:,1,:,:]) / (audio_mix_spec2[:,0,:,:] * audio_mix_spec2[:,0,:,:] + audio_mix_spec2[:,1,:,:] * audio_mix_spec2[:,1,:,:] + 1e-30)
        gt_masks_A2 = torch.cat((gt_masks_A2_real.unsqueeze(1), gt_masks_A2_imag.unsqueeze(1)), 1)
        gt_masks_B1_real = (audio_spec_B[:,0,:,:] * audio_mix_spec1[:,0,:,:] + audio_spec_B[:,1,:,:] * audio_mix_spec1[:,1,:,:]) / (audio_mix_spec1[:,0,:,:] * audio_mix_spec1[:,0,:,:] + audio_mix_spec1[:,1,:,:] * audio_mix_spec1[:,1,:,:] + 1e-30)
        gt_masks_B1_imag = (audio_spec_B[:,1,:,:] * audio_mix_spec1[:,0,:,:] - audio_spec_B[:,0,:,:] * audio_mix_spec1[:,1,:,:]) / (audio_mix_spec1[:,0,:,:] * audio_mix_spec1[:,0,:,:] + audio_mix_spec1[:,1,:,:] * audio_mix_spec1[:,1,:,:] + 1e-30)
        gt_masks_B1 = torch.cat((gt_masks_B1_real.unsqueeze(1), gt_masks_B1_imag.unsqueeze(1)), 1)
        gt_masks_B2_real = (audio_spec_B[:,0,:,:] * audio_mix_spec2[:,0,:,:] + audio_spec_B[:,1,:,:] * audio_mix_spec2[:,1,:,:]) / (audio_mix_spec2[:,0,:,:] * audio_mix_spec2[:,0,:,:] + audio_mix_spec2[:,1,:,:] * audio_mix_spec2[:,1,:,:] + 1e-30)
        gt_masks_B2_imag = (audio_spec_B[:,1,:,:] * audio_mix_spec2[:,0,:,:] - audio_spec_B[:,0,:,:] * audio_mix_spec2[:,1,:,:]) / (audio_mix_spec2[:,0,:,:] * audio_mix_spec2[:,0,:,:] + audio_mix_spec2[:,1,:,:] * audio_mix_spec2[:,1,:,:] + 1e-30)
        gt_masks_B2 = torch.cat((gt_masks_B2_real.unsqueeze(1), gt_masks_B2_imag.unsqueeze(1)), 1)

        # mask compression
        gt_masks_A1.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)
        gt_masks_A2.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)
        gt_masks_B1.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)
        gt_masks_B2.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)

        if self.opt.compression_type == 'hyperbolic':
            K = self.opt.hyperbolic_compression_K
            C = self.opt.hyperbolic_compression_C
            gt_masks_A1 = K * (1 - torch.exp(-C * gt_masks_A1)) / (1 + torch.exp(-C * gt_masks_A1))
            gt_masks_A2 = K * (1 - torch.exp(-C * gt_masks_A2)) / (1 + torch.exp(-C * gt_masks_A2))
            gt_masks_B1 = K * (1 - torch.exp(-C * gt_masks_B1)) / (1 + torch.exp(-C * gt_masks_B1))
            gt_masks_B2 = K * (1 - torch.exp(-C * gt_masks_B2)) / (1 + torch.exp(-C * gt_masks_B2))
        elif self.opt.compression_type == 'sigmoidal':
            a = self.opt.sigmoidal_compression_a
            b = self.opt.sigmoidal_compression_b
            gt_masks_A1 = 1 / (1 + torch.exp(-a * gt_masks_A1 + b))
            gt_masks_A2 = 1 / (1 + torch.exp(-a * gt_masks_A2 + b))
            gt_masks_B1 = 1 / (1 + torch.exp(-a * gt_masks_B1 + b))
            gt_masks_B2 = 1 / (1 + torch.exp(-a * gt_masks_B2 + b))

        # pass through visual stream and extract lipreading features
        lipreading_feature_A1 = self.net_lipreading(Variable(mouthroi_A1, requires_grad=False), self.opt.num_frames)
        lipreading_feature_A2 = self.net_lipreading(Variable(mouthroi_A2, requires_grad=False), self.opt.num_frames)
        lipreading_feature_B = self.net_lipreading(Variable(mouthroi_B, requires_grad=False), self.opt.num_frames)
        # pass through visual stream and extract identity features
        if self.opt.number_of_identity_frames == 1:
            identity_feature_A = self.net_identity(Variable(identityframe_A, requires_grad=False))
            identity_feature_B = self.net_identity(Variable(identityframe_B, requires_grad=False))
        else:
            identity_feature_A = self.net_identity.forward_multiframe(Variable(identityframe_A, requires_grad=False))
            identity_feature_B = self.net_identity.forward_multiframe(Variable(identityframe_B, requires_grad=False))
        if self.opt.l2_feature_normalization:
            identity_feature_A = F.normalize(identity_feature_A, p=2, dim=1)
            identity_feature_B = F.normalize(identity_feature_B, p=2, dim=1)
        output = {}
        output['identity_feature_A'] = identity_feature_A
        output['identity_feature_B'] = identity_feature_B

        # what type of visual feature to use
        identity_feature_A = identity_feature_A.repeat(1, 1, 1, lipreading_feature_A1.shape[-1])
        identity_feature_B = identity_feature_B.repeat(1, 1, 1, lipreading_feature_B.shape[-1])
        if self.opt.visual_feature_type == 'both':
            visual_feature_A1 = torch.cat((identity_feature_A, lipreading_feature_A1), dim=1)
            visual_feature_A2 = torch.cat((identity_feature_A, lipreading_feature_A2), dim=1)
            visual_feature_B = torch.cat((identity_feature_B, lipreading_feature_B), dim=1)
        elif self.opt.visual_feature_type == 'lipmotion':
            visual_feature_A1 = lipreading_feature_A1
            visual_feature_A2 = lipreading_feature_A2
            visual_feature_B = lipreading_feature_B
        elif self.opt.visual_feature_type == 'identity':
            visual_feature_A1 = identity_feature_A
            visual_feature_A2 = identity_feature_A
            visual_feature_B = identity_feature_B

        # audio-visual feature fusion through UNet and predict mask
        if self.opt.compression_type == 'hyperbolic':
            scalar = self.opt.hyperbolic_compression_K
            activation = 'Tanh'
        elif self.opt.compression_type == 'none':
            scalar = self.opt.mask_clip_threshold
            activation = 'Tanh'
        elif self. opt.compression_type == 'sigmoidal':
            scalar = 1
            activation = 'Sigmoid'
            
        mask_prediction_A1 = scalar * self.net_unet(audio_mix_spec1, visual_feature_A1, activation)
        mask_prediction_A2 = scalar * self.net_unet(audio_mix_spec2, visual_feature_A2, activation)
        mask_prediction_B1 = scalar * self.net_unet(audio_mix_spec1, visual_feature_B, activation)
        mask_prediction_B2 = scalar * self.net_unet(audio_mix_spec2, visual_feature_B, activation)

        # calculate loss weighting coefficient
        if self.opt.weighted_loss:
            weight1 = torch.log1p(torch.norm(audio_mix_spec1[:,:,:-1,:], p=2, dim=1)).unsqueeze(1).repeat(1,2,1,1)
            weight1 = torch.clamp(weight1, 1e-3, 10)
            weight2 = torch.log1p(torch.norm(audio_mix_spec2[:,:,:-1,:], p=2, dim=1)).unsqueeze(1).repeat(1,2,1,1)
            weight2 = torch.clamp(weight2, 1e-3, 10)
        else:
            weight1 = None
            weight2 = None

        if self.opt.compression_type == 'hyperbolic':
            K = self.opt.hyperbolic_compression_K
            C = self.opt.hyperbolic_compression_C
            mask_prediction_A1 = - torch.log((K - mask_prediction_A1) / (K + mask_prediction_A1)) / C
            mask_prediction_A2 = - torch.log((K - mask_prediction_A2) / (K + mask_prediction_A2)) / C
            mask_prediction_B1 = - torch.log((K - mask_prediction_B1) / (K + mask_prediction_B1)) / C
            mask_prediction_B2 = - torch.log((K - mask_prediction_B2) / (K + mask_prediction_B2)) / C
        elif self.opt.compression_type == 'sigmoidal':
            a = self.opt.sigmoidal_compression_a
            b = self.opt.sigmoidal_compression_b
            mask_prediction_A1 = (b - torch.log(1 / mask_prediction_A1 - 1)) / a
            mask_prediction_A2 = (b - torch.log(1 / mask_prediction_A2 - 1)) / a
            mask_prediction_B1 = (b - torch.log(1 / mask_prediction_B1 - 1)) / a
            mask_prediction_B2 = (b - torch.log(1 / mask_prediction_B2 - 1)) / a

        mask_prediction_A1.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)
        mask_prediction_A2.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)
        mask_prediction_B1.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)
        mask_prediction_B2.clamp_(-self.opt.mask_clip_threshold, self.opt.mask_clip_threshold)

        pred_spec_A1_real = audio_mix_spec1[:, 0, :-1, :] * mask_prediction_A1[:, 0, :, :] - audio_mix_spec1[:, 1, :-1, :] * mask_prediction_A1[:, 1, :, :]
        pred_spec_A1_imag = audio_mix_spec1[:, 1, :-1, :] * mask_prediction_A1[:, 0, :, :] + audio_mix_spec1[:, 0, :-1, :] * mask_prediction_A1[:, 1, :, :]
        pred_spec_A2_real = audio_mix_spec2[:, 0, :-1, :] * mask_prediction_A2[:, 0, :, :] - audio_mix_spec2[:, 1, :-1, :] * mask_prediction_A2[:, 1, :, :]
        pred_spec_A2_imag = audio_mix_spec2[:, 1, :-1, :] * mask_prediction_A2[:, 0, :, :] + audio_mix_spec2[:, 0, :-1, :] * mask_prediction_A2[:, 1, :, :]
        pred_spec_B1_real = audio_mix_spec1[:, 0, :-1, :] * mask_prediction_B1[:, 0, :, :] - audio_mix_spec1[:, 1, :-1, :] * mask_prediction_B1[:, 1, :, :]
        pred_spec_B1_imag = audio_mix_spec1[:, 1, :-1, :] * mask_prediction_B1[:, 0, :, :] + audio_mix_spec1[:, 0, :-1, :] * mask_prediction_B1[:, 1, :, :]
        pred_spec_B2_real = audio_mix_spec2[:, 0, :-1, :] * mask_prediction_B2[:, 0, :, :] - audio_mix_spec2[:, 1, :-1, :] * mask_prediction_B2[:, 1, :, :]
        pred_spec_B2_imag = audio_mix_spec2[:, 1, :-1, :] * mask_prediction_B2[:, 0, :, :] + audio_mix_spec2[:, 0, :-1, :] * mask_prediction_B2[:, 1, :, :]
        pred_spec_A1 = torch.cat((pred_spec_A1_real.unsqueeze(1), pred_spec_A1_imag.unsqueeze(1)), dim=1)
        pred_spec_A2 = torch.cat((pred_spec_A2_real.unsqueeze(1), pred_spec_A2_imag.unsqueeze(1)), dim=1)
        pred_spec_B1 = torch.cat((pred_spec_B1_real.unsqueeze(1), pred_spec_B1_imag.unsqueeze(1)), dim=1)
        pred_spec_B2 = torch.cat((pred_spec_B2_real.unsqueeze(1), pred_spec_B2_imag.unsqueeze(1)), dim=1)

        #extract feature embedding
        audio_embeddings_A1_pred = self.net_classifier(pred_spec_A1)
        audio_embeddings_A2_pred = self.net_classifier(pred_spec_A2)
        audio_embeddings_B1_pred = self.net_classifier(pred_spec_B1)
        audio_embeddings_B2_pred = self.net_classifier(pred_spec_B2)
        audio_embeddings_A1_gt = self.net_classifier(audio_spec_A1[:,:,:-1,:])
        audio_embeddings_A2_gt = self.net_classifier(audio_spec_A2[:,:,:-1,:])
        audio_embeddings_B_gt = self.net_classifier(audio_spec_B[:,:,:-1,:])

        if self.opt.l2_feature_normalization:
            audio_embeddings_A1_pred = F.normalize(audio_embeddings_A1_pred, p=2, dim=1)
            audio_embeddings_A2_pred = F.normalize(audio_embeddings_A2_pred, p=2, dim=1)
            audio_embeddings_B1_pred = F.normalize(audio_embeddings_B1_pred, p=2, dim=1)
            audio_embeddings_B2_pred = F.normalize(audio_embeddings_B2_pred, p=2, dim=1)
            audio_embeddings_A1_gt = F.normalize(audio_embeddings_A1_gt, p=2, dim=1)
            audio_embeddings_A2_gt = F.normalize(audio_embeddings_A2_gt, p=2, dim=1)
            audio_embeddings_B_gt = F.normalize(audio_embeddings_B_gt, p=2, dim=1)

        output['pred_spec_1'] = torch.norm(pred_spec_A1, dim=1, p=2) + torch.norm(pred_spec_B1, dim=1, p=2)
        output['pred_spec_2'] = torch.norm(pred_spec_A2, dim=1, p=2) + torch.norm(pred_spec_B2, dim=1, p=2)
        output['mask_predictions_A1'] = mask_prediction_A1
        output['mask_predictions_A2'] = mask_prediction_A2
        output['mask_predictions_B1'] = mask_prediction_B1
        output['mask_predictions_B2'] = mask_prediction_B2
        output['audio_embedding_A1_pred'] = audio_embeddings_A1_pred
        output['audio_embedding_A2_pred'] = audio_embeddings_A2_pred
        output['audio_embedding_B1_pred'] = audio_embeddings_B1_pred
        output['audio_embedding_B2_pred'] = audio_embeddings_B2_pred
        output['audio_embedding_A1_gt'] = audio_embeddings_A1_gt
        output['audio_embedding_A2_gt'] = audio_embeddings_A2_gt
        output['audio_embedding_B_gt'] = audio_embeddings_B_gt
        output['gt_masks_A1'] = gt_masks_A1
        output['gt_masks_A2'] = gt_masks_A2
        output['gt_masks_B1'] = gt_masks_B1
        output['gt_masks_B2'] = gt_masks_B2
        output['audio_mix_spec1'] = audio_mix_spec1
        output['audio_mix_spec2'] = audio_mix_spec2
        output['weight1'] = weight1
        output['weight2'] = weight2
        return output
