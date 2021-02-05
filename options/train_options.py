#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of displaying average loss and accuracy')
		self.parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')
		self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
		self.parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor for learning rate')
		self.parser.add_argument('--tensorboard', type=bool, default=False, help='use tensorboard to visualize loss change ')
		self.parser.add_argument('--measure_time', type=bool, default=False, help='measure time of different steps during training')
		self.parser.add_argument('--niter', type=int, default=1, help='# of epochs to train, set to 1 because we are doing random sampling from the whole dataset')
		self.parser.add_argument('--num_batch', default=30000, type=int, help='number of batches to train')
		self.parser.add_argument('--validation_on', type=bool, default=False, help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=500, help='frequency of testing on validation set')
		self.parser.add_argument('--validation_batches', type=int, default=10, help='number of batches to test for validation')
		self.parser.add_argument('--num_visualization_examples', type=int, default=5, help='number of examples to visualize')		

		#model arguments
		self.parser.add_argument('--visual_pool', type=str, default='maxpool', help='avg or max pool for visual stream feature')
		self.parser.add_argument('--audio_pool', type=str, default='maxpool', help="avg or max pool for audio stream feature")
		self.parser.add_argument('--weights_facial', type=str, default='', help="weights for facial attributes net")
		self.parser.add_argument('--weights_unet', type=str, default='', help="weights for unet")
		self.parser.add_argument('--weights_vocal', type=str, default='', help="weights for vocal attributes net")
		self.parser.add_argument('--weights_lipreadingnet', type=str, default='', help="weights for lipreading net")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=3, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=3, help="output spectrogram number of channels")
		self.parser.add_argument('--coseparation_loss_weight', default=0.05, type=float, help='weight for coseparation loss')
		self.parser.add_argument('--mixandseparate_loss_weight', default=1, type=float, help='weight for reconstruction loss')
		self.parser.add_argument('--crossmodal_loss_weight', default=0.05, type=float, help='weight for crossmodal loss')
		self.parser.add_argument('--mask_loss_type', default='L1', type=str, choices=('L1', 'L2', 'BCE'), help='type of loss on mask')
		self.parser.add_argument('--triplet_loss_type', default='tripletCosine', type=str, choices=('tripletCosine', 'triplet'), help='type of triplet loss')
		self.parser.add_argument('--weighted_loss', action='store_true', help="weighted loss")
		self.parser.add_argument('--log_freq', type=bool, default=False, help="whether use log-scale frequency")		
		self.parser.add_argument('--mask_thresh', default=0.5, type=float, help='mask threshold for binary mask')
		self.parser.add_argument('--lipreading_config_path', type=str, help='path to the config file of lipreading')
		self.parser.add_argument('--audioVisual_feature_dim', type=int, default=1280, help="dimension of audioVisual feature map")
		self.parser.add_argument('--identity_feature_dim', type=int, default=64, help="dimension of identity feature map")
		self.parser.add_argument('--visual_feature_type', default='both', type=str, choices=('lipmotion', 'identity', 'both'), help='type of visual feature to use')
		self.parser.add_argument('--lipreading_extract_feature',default=False,action='store_true',help="whether use features extracted from 3d conv")
		self.parser.add_argument('--number_of_identity_frames', type=int, default=1, help="number of identity frames to use")
		self.parser.add_argument('--compression_type', type=str, default='hyperbolic', choices=('hyperbolic', 'sigmoidal', 'none'), help="type of compression on masks")
		self.parser.add_argument('--hyperbolic_compression_K', type=int, default=10, help="hyperbolic compression K")
		self.parser.add_argument('--hyperbolic_compression_C', type=float, default=0.1, help="hyperbolic compression C")
		self.parser.add_argument('--sigmoidal_compression_a', type=float, default=0.1, help="sigmoidal compression a")
		self.parser.add_argument('--sigmoidal_compression_b', type=int, default=0, help="sigmoidal compression b")
		self.parser.add_argument('--mask_clip_threshold', type=int, default=5, help="mask_clip_threshold")
		self.parser.add_argument('--l2_feature_normalization',default=False,action='store_true',help="whether l2 nomalizing identity/audio features")
		self.parser.add_argument('--gt_percentage', type=float, default=0.5, help="percentage to use gt embeddings")

		#preprocessing
		self.parser.add_argument('--scale_w',nargs='+',help='Scale width of the video',default=[128],type=int)
		self.parser.add_argument('--scale_h',nargs='+',help='Scale height oft the video',default=[128],type=int)
		self.parser.add_argument("--crop_size",type=int,default=112,help="Final image scale",)
		self.parser.add_argument('--normalization',default=False,action='store_true',help="Should we use input normalization?")
		self.parser.add_argument('--audio_augmentation',default=False,action='store_true', help='whether to augment input audio')
		self.parser.add_argument('--audio_normalization',default=False,action='store_true',help="whether to normalize audio?")

		#optimizer arguments
		self.parser.add_argument('--lr_lipreading', type=float, default=0.0001, help='learning rate for lipreading stream')
		self.parser.add_argument('--lr_facial_attributes', type=float, default=0.0001, help='learning rate for identity stream')
		self.parser.add_argument('--lr_unet', type=float, default=0.001, help='learning rate for unet')
		self.parser.add_argument('--lr_vocal_attributes', type=float, default=0.001, help='learning rate for audio classifier')
		self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[10000, 20000], help='steps to drop LR in training samples')
		self.parser.add_argument('--optimizer', default='sgd', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=0.0001, type=float, help='weights regularizer')
		self.parser.add_argument('--margin', default=0.5, type=float, help='margin for triplet loss')

		self.mode = 'train'
