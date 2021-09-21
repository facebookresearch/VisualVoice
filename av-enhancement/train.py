#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from imageio import imwrite
import scipy.io.wavfile as wavfile
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import librosa
from utils import utils
from models import criterion
import torch.nn.functional as F

def create_optimizer(nets, opt):
        (net_lipreading, net_facial_attribtes, net_unet, net_vocal_attributes) = nets
        param_groups = [{'params': net_lipreading.parameters(), 'lr': opt.lr_lipreading},
                        {'params': net_facial_attribtes.parameters(), 'lr': opt.lr_facial_attributes},
                        {'params': net_unet.parameters(), 'lr': opt.lr_unet},
                        {'params': net_vocal_attributes.parameters(), 'lr': opt.lr_vocal_attributes}]
        if opt.optimizer == 'sgd':
            return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adam':
            return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

#used to display validation loss
def display_val(model, crit, writer, index, dataset_val, opt):
        #initial results lists
        mixandseparate_losses = []
        coseparation_losses = []
        crossmodal_losses = []

        with torch.no_grad():
            for i, val_data in enumerate(dataset_val):
                output = model.forward(val_data)
                coseparation_loss = get_coseparation_loss(output, opt, crit['loss_triplet']) * opt.coseparation_loss_weight
                mixandseparate_loss = get_mixandseparate_loss(output, opt, crit['loss_mixandseparate']) * opt.mixandseparate_loss_weight
                crossmodal_loss = get_crossmodal_loss(output, opt, crit['loss_triplet']) * opt.crossmodal_loss_weight
                coseparation_losses.append(coseparation_loss.item()) 
                mixandseparate_losses.append(mixandseparate_loss.item())
                crossmodal_losses.append(crossmodal_loss.item())

        avg_coseparation_loss = sum(coseparation_losses)/len(coseparation_losses)
        avg_mixandseparate_loss = sum(mixandseparate_losses)/len(mixandseparate_losses)
        avg_crossmodal_loss = sum(crossmodal_losses)/len(crossmodal_losses)

        if opt.tensorboard:
            writer.add_scalar('data/val_coseparation_loss', avg_coseparation_loss, index)
            writer.add_scalar('data/val_mixandseparate_loss', avg_mixandseparate_loss, index)
            writer.add_scalar('data/val_crossmodal_loss', avg_crossmodal_loss, index)
        print('val mix and separate loss: %.5f' % avg_mixandseparate_loss)
        print('val coseparation loss: %.5f' % avg_coseparation_loss)
        print('val crossmodal loss: %.5f' % avg_crossmodal_loss)
        return avg_mixandseparate_loss + avg_coseparation_loss + avg_crossmodal_loss

def get_mixandseparate_loss(output, opt, loss_mixandseparate):
        gt_masks_A1 = output['gt_masks_A1']
        gt_masks_A2 = output['gt_masks_A2']
        gt_masks_B1 = output['gt_masks_B1']
        gt_masks_B2 = output['gt_masks_B2']
        mask_prediction_A1 = output['mask_predictions_A1']
        mask_prediction_A2 = output['mask_predictions_A2']
        mask_prediction_B1 = output['mask_predictions_B1']
        mask_prediction_B2 = output['mask_predictions_B2']
        weight1 = output['weight1']
        weight2 = output['weight2']
        mixandseparate_loss = loss_mixandseparate(mask_prediction_A1, gt_masks_A1[:,:,:-1,:], weight1) + loss_mixandseparate(mask_prediction_A2, gt_masks_A2[:,:,:-1,:], weight2) + loss_mixandseparate(mask_prediction_B1, gt_masks_B1[:,:,:-1,:], weight1) + loss_mixandseparate(mask_prediction_B2, gt_masks_B2[:,:,:-1,:], weight2)
        return mixandseparate_loss

def get_coseparation_loss(output, opt, loss_triplet):
    if random.random() > opt.gt_percentage:
        audio_embeddings_A1 = output['audio_embedding_A1_pred']
        audio_embeddings_A2 = output['audio_embedding_A2_pred']     
        audio_embeddings_B1 = output['audio_embedding_B1_pred']
        audio_embeddings_B2 = output['audio_embedding_B2_pred']
    else:
        audio_embeddings_A1 = output['audio_embedding_A1_gt']  
        audio_embeddings_A2 = output['audio_embedding_A2_gt']     
        audio_embeddings_B1 = output['audio_embedding_B_gt']
        audio_embeddings_B2 = output['audio_embedding_B_gt']
    
    coseparation_loss = loss_triplet(audio_embeddings_A1, audio_embeddings_A2, audio_embeddings_B1) + loss_triplet(audio_embeddings_A1, audio_embeddings_A2, audio_embeddings_B2)
    return coseparation_loss

def get_crossmodal_loss(output, opt, loss_triplet):
    identity_feature_A = output['identity_feature_A']
    identity_feature_B = output['identity_feature_B']
    if random.random() > opt.gt_percentage:
        audio_embeddings_A1 = output['audio_embedding_A1_pred']
        audio_embeddings_A2 = output['audio_embedding_A2_pred']     
        audio_embeddings_B1 = output['audio_embedding_B1_pred']
        audio_embeddings_B2 = output['audio_embedding_B2_pred']
    else:
        audio_embeddings_A1 = output['audio_embedding_A1_gt']  
        audio_embeddings_A2 = output['audio_embedding_A2_gt']     
        audio_embeddings_B1 = output['audio_embedding_B_gt']
        audio_embeddings_B2 = output['audio_embedding_B_gt']
    crossmodal_loss = loss_triplet(audio_embeddings_A1, identity_feature_A, identity_feature_B) + loss_triplet(audio_embeddings_A2, identity_feature_A, identity_feature_B) + loss_triplet(audio_embeddings_B1, identity_feature_B, identity_feature_A) + loss_triplet(audio_embeddings_B2, identity_feature_B, identity_feature_A)
    return crossmodal_loss

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

#construct data loader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

#create validation set data loader if validation_on option is set
if opt.validation_on:
        #temperally set to val to load val data
        opt.mode = 'val'
        data_loader_val = CreateDataLoader(opt)
        dataset_val = data_loader_val.load_data()
        dataset_size_val = len(data_loader_val)
        print('#validation images = %d' % dataset_size_val)
        opt.mode = 'train' #set it back

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# Network Builders
builder = ModelBuilder()
net_lipreading = builder.build_lipreadingnet(
        config_path=opt.lipreading_config_path,
        weights=opt.weights_lipreadingnet,
        extract_feats=opt.lipreading_extract_feature)
#if identity feature dim is not 512, for resnet reduce dimension to this feature dim
if opt.identity_feature_dim != 512:
    opt.with_fc = True
else:
    opt.with_fc = False
net_facial_attribtes = builder.build_facial(
        pool_type=opt.visual_pool,
        fc_out = opt.identity_feature_dim,
        with_fc=opt.with_fc,
        weights=opt.weights_facial)  
net_unet = builder.build_unet(
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        audioVisual_feature_dim=opt.audioVisual_feature_dim,
        identity_feature_dim=opt.identity_feature_dim,
        weights=opt.weights_unet)
net_vocal_attributes = builder.build_vocal(
        pool_type=opt.audio_pool,
        input_channel=2,
        with_fc=opt.with_fc,
        fc_out = opt.identity_feature_dim,
        weights=opt.weights_vocal)
nets = (net_lipreading, net_facial_attribtes, net_unet, net_vocal_attributes)
print(nets)

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)

# Set up optimizer
optimizer = create_optimizer(nets, opt)

cudnn.benchmark = True

# Set up loss functions
if opt.triplet_loss_type == 'tripletCosine':
    loss_triplet = criterion.TripletLossCosine(opt.margin)
elif opt.triplet_loss_type == 'triplet':
    loss_triplet = criterion.TripletLoss(opt.margin)
if opt.mask_loss_type == 'L1':
    loss_mixandseparate = criterion.L1Loss()
elif opt.mask_loss_type == 'L2':
    loss_mixandseparate = criterion.L2Loss()
if(len(opt.gpu_ids) > 0):
    loss_triplet.cuda(opt.gpu_ids[0])
    loss_mixandseparate.cuda(opt.gpu_ids[0])
crit = {'loss_triplet': loss_triplet, 'loss_mixandseparate': loss_mixandseparate}


#initialization
total_batches = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_coseparation_loss = []
batch_mixandseparate_loss = []
batch_crossmodal_loss = []
best_err = float("inf")

for epoch in range(1 + opt.epoch_count, opt.niter+1):
        torch.cuda.synchronize()
        epoch_start_time = time.time()

        if(opt.measure_time):
                iter_start_time = time.time()
        for i, data in enumerate(dataset):
                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_loaded_time = time.time()

                total_batches += 1

                #forward pass
                model.zero_grad()
                output = model.forward(data)

                #compute loss                
                #mix and separate loss
                mixandseparate_loss = get_mixandseparate_loss(output, opt, loss_mixandseparate) * opt.mixandseparate_loss_weight
                #coseparation loss
                coseparation_loss = get_coseparation_loss(output, opt, loss_triplet) * opt.coseparation_loss_weight
                #crossmodal loss
                crossmodal_loss = get_crossmodal_loss(output, opt, loss_triplet) * opt.crossmodal_loss_weight
                loss = mixandseparate_loss + coseparation_loss + crossmodal_loss

                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_forwarded_time = time.time()
                #store losses for this batch
                batch_coseparation_loss.append(coseparation_loss.item())
                batch_mixandseparate_loss.append(mixandseparate_loss.item())
                batch_crossmodal_loss.append(crossmodal_loss.item())

                optimizer.zero_grad()
                crossmodal_loss.backward(retain_graph=True)
                coseparation_loss.backward(retain_graph=True)
                mixandseparate_loss.backward()
                optimizer.step()

                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_model_backwarded_time = time.time()

                if(opt.measure_time):
                        torch.cuda.synchronize()
                        iter_model_backwarded_time = time.time()
                        data_loading_time.append(iter_data_loaded_time - iter_start_time)
                        model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                        model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

                if(total_batches % opt.display_freq == 0):
                        print('Display training progress at (epoch %d, total_batches %d)' % (epoch, total_batches))
                        avg_coseparation_loss = sum(batch_coseparation_loss)/len(batch_coseparation_loss)
                        avg_mixandseparate_loss = sum(batch_mixandseparate_loss)/len(batch_mixandseparate_loss)
                        avg_crossmodal_loss = sum(batch_crossmodal_loss)/len(batch_crossmodal_loss)

                        print('mix-and-separate loss: %.5f, co-separation loss: %.5f, crossmodal loss: %.5f' \
                            % (avg_mixandseparate_loss, avg_coseparation_loss, avg_crossmodal_loss))
                        batch_coseparation_loss = []
                        batch_mixandseparate_loss = []
                        batch_crossmodal_loss = []
                        if opt.tensorboard:
                            writer.add_scalar('data/coseparation_loss', avg_coseparation_loss, i)
                            writer.add_scalar('data/mixandseparate_loss', avg_mixandseparate_loss, i)
                            writer.add_scalar('data/crossmodal_loss', avg_crossmodal_loss, i)
                        if(opt.measure_time):
                                print('average data loading time: %.3f' % (sum(data_loading_time)/len(data_loading_time)))
                                print('average forward time: %.3f' % (sum(model_forward_time)/len(model_forward_time)))
                                print('average backward time: %.3f' % (sum(model_backward_time)/len(model_backward_time)))
                                data_loading_time = []
                                model_forward_time = []
                                model_backward_time = []
                        print('end of display \n')

                if(total_batches % opt.save_latest_freq == 0):
                        print('saving the latest model (epoch %d, total_batches %d)' % (epoch, total_batches))
                        torch.save(net_lipreading.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'lipreading_latest.pth'))
                        torch.save(net_facial_attribtes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'facial_latest.pth'))
                        torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'unet_latest.pth'))
                        torch.save(net_vocal_attributes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'vocal_latest.pth'))

                if(total_batches % opt.validation_freq == 0 and opt.validation_on):
                        model.eval()
                        opt.mode = 'val'
                        print('Display validation results at (epoch %d, total_batches %d)' % (epoch, total_batches))
                        val_err = display_val(model, crit, writer, total_batches, dataset_val, opt)
                        print('end of display \n')
                        model.train()
                        opt.mode = 'train'
                        #save the model that achieves the smallest validation error
                        if val_err < best_err:
                            best_err = val_err
                            print('saving the best model (epoch %d, total_batches %d) with validation error %.3f\n' % (epoch, total_batches, val_err))
                            torch.save(net_lipreading.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'lipreading_best.pth'))
                            torch.save(net_facial_attribtes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'facial_best.pth'))
                            torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'unet_best.pth'))
                            torch.save(net_vocal_attributes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'vocal_best.pth'))

                #decrease learning rate
                if(total_batches in opt.lr_steps):
                        decrease_learning_rate(optimizer, opt.decay_factor)
                        print('decreased learning rate by ', opt.decay_factor)

                if(opt.measure_time):
                        torch.cuda.synchronize()
                        iter_start_time = time.time()