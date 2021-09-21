#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import librosa
from scipy.io import wavfile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from options.test_real_options import TestRealOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram_complex, load_mouthroi, get_preprocessing_pipelines, load_frame
from utils import utils
from utils.lipreading_preprocess import *
from facenet_pytorch import MTCNN

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return rms / desired_rms, samples

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def get_separated_audio(net_lipreading, net_facial_attributes, net_unet, spec_mix, segment_mouthroi, frames, opt):
	lipreading_feature = net_lipreading.forward(segment_mouthroi, opt.num_frames)

	#extract identity feature
	if opt.number_of_identity_frames == 1:
		identity_feature = net_facial_attributes.forward(frames)
	else:
		identity_feature = net_facial_attributes.forward_multiframe(frames)
		
	if opt.l2_feature_normalization:
		identity_feature = F.normalize(identity_feature, p=2, dim=1)
		
	# what type of visual feature to use
	identity_feature = identity_feature.repeat(1, 1, 1, lipreading_feature.shape[-1])
	if opt.visual_feature_type == 'both':
		visual_feature = torch.cat((identity_feature, lipreading_feature), dim=1)
	elif opt.visual_feature_type == 'lipmotion':
		visual_feature = lipreading_feature
	elif opt.visual_feature_type == 'identity':
		visual_feature = identity_feature
		
	if opt.compression_type == 'hyperbolic':
		scalar = opt.hyperbolic_compression_K
		activation = 'Tanh'
	elif opt.compression_type == 'none':
		scalar = opt.mask_clip_threshold
		activation = 'Tanh'
	elif opt.compression_type == 'sigmoidal':
		scalar = 1
		activation = 'Sigmoid'
		
	mask_prediction = scalar * net_unet.forward(spec_mix, visual_feature, activation)
	mask_prediction.clamp_(-opt.mask_clip_threshold, opt.mask_clip_threshold)
	spec_mix = spec_mix.detach().cpu().numpy()
	pred_masks = mask_prediction.detach().cpu().numpy()
	
	pred_spec_real = spec_mix[0, 0, :-1] * pred_masks[0, 0] - spec_mix[0, 1, :-1] * pred_masks[0, 1] 
	pred_spec_imag = spec_mix[0, 1, :-1] * pred_masks[0, 0] + spec_mix[0, 0, :-1] * pred_masks[0, 1]
	pred_spec_real = np.concatenate((pred_spec_real, spec_mix[0,0,-1:,:]), axis=0)
	pred_spec_imag = np.concatenate((pred_spec_imag, spec_mix[0,1,-1:,:]), axis=0)
	preds_wav = utils.istft_reconstruction_from_complex(pred_spec_real, pred_spec_imag, hop_length=opt.hop_size, length=int(opt.audio_length * opt.audio_sampling_rate))
	return preds_wav

def main():
	#load test arguments
	opt = TestRealOptions().parse()
	opt.device = torch.device("cuda")

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
	net_facial_attributes = builder.build_facial(
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

	nets = (net_lipreading, net_facial_attributes, net_unet, net_vocal_attributes)
	print(nets)

	# construct our audio-visual model
	model = AudioVisualModel(nets, opt)
	model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
	model.to(opt.device)
	model.eval()
	mtcnn = MTCNN(keep_all=True, device=opt.device)

	lipreading_preprocessing_func = get_preprocessing_pipelines()['test']
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
	vision_transform_list = [transforms.ToTensor()]
	if opt.normalization:
		vision_transform_list.append(normalize)
	vision_transform = transforms.Compose(vision_transform_list)

	for speaker_index in range(opt.number_of_speakers):
		mouthroi_path = os.path.join(opt.mouthroi_root, 'speaker' + str(speaker_index+1) + '.npz')
		facetrack_path = os.path.join(opt.facetrack_root, 'speaker' + str(speaker_index+1) + '.mp4')

		#load data
		mouthroi = load_mouthroi(mouthroi_path)
		sr, audio = wavfile.read(opt.audio_path)
		print("sampling rate of audio: ", sr)
		if len((audio.shape)) == 2:
			audio = np.mean(audio, axis=1) #convert to mono if stereo
		audio = audio / 32768
		audio = clip_audio(audio)
		audio_length = len(audio)

		if opt.reliable_face:
			best_score = 0
			for i in range(10):
				frame = load_frame(facetrack_path)
				boxes, scores = mtcnn.detect(frame)
				if scores[0] > best_score:
					best_frame = frame			
			frames = vision_transform(best_frame).squeeze().unsqueeze(0).cuda()
		else:
			frame_list = []
			for i in range(opt.number_of_identity_frames):
				frame = load_frame(facetrack_path)
				frame = vision_transform(frame)
				frame_list.append(frame)
			frame = torch.stack(frame_list).squeeze().unsqueeze(0).cuda()
		

		#perform separation over the whole audio using a sliding window approach
		sliding_window_start = 0
		overlap_count = np.zeros((audio_length))
		sep_audio = np.zeros((audio_length))
		avged_sep_audio = np.zeros((audio_length))
		samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
		while sliding_window_start + samples_per_window < audio_length:
			sliding_window_end = sliding_window_start + samples_per_window

			#get audio spectrogram
			segment_audio = audio[sliding_window_start:sliding_window_end]
			if opt.audio_normalization:
				normalizer, segment_audio = audio_normalize(segment_audio, desired_rms=opt.desired_rms)
			else:
				normalizer = 1

			audio_spec = generate_spectrogram_complex(segment_audio, opt.window_size, opt.hop_size, opt.n_fft)            
			audio_spec = torch.FloatTensor(audio_spec).unsqueeze(0).cuda()

			#get mouthroi
			frame_index_start = int(round(sliding_window_start / opt.audio_sampling_rate * 25))
			segment_mouthroi = mouthroi[frame_index_start:(frame_index_start + opt.num_frames), :, :]
			segment_mouthroi = lipreading_preprocessing_func(segment_mouthroi)
			segment_mouthroi = torch.FloatTensor(segment_mouthroi).unsqueeze(0).unsqueeze(0).cuda()
				
			reconstructed_signal = get_separated_audio(net_lipreading, net_facial_attributes, net_unet, audio_spec, segment_mouthroi, frames, opt)
			reconstructed_signal = reconstructed_signal * normalizer
			sep_audio[sliding_window_start:sliding_window_end] = sep_audio[sliding_window_start:sliding_window_end] + reconstructed_signal

			#update overlap count
			overlap_count[sliding_window_start:sliding_window_end] = overlap_count[sliding_window_start:sliding_window_end] + 1
			sliding_window_start = sliding_window_start + int(opt.hop_length * opt.audio_sampling_rate)
				
		#deal with the last segment
		segment_audio = audio[-samples_per_window:]
		if opt.audio_normalization:
			normalizer, segment_audio = audio_normalize(segment_audio, desired_rms=opt.desired_rms)
		else:
			normalizer = 1

		audio_spec = generate_spectrogram_complex(segment_audio, opt.window_size, opt.hop_size, opt.n_fft)
		audio_spec = torch.FloatTensor(audio_spec).unsqueeze(0).cuda()
		#get mouthroi
		frame_index_start = int(round((len(audio) - samples_per_window) / opt.audio_sampling_rate * 25)) - 1
		segment_mouthroi = mouthroi[-opt.num_frames:, :, :]
		segment_mouthroi = lipreading_preprocessing_func(segment_mouthroi)
		segment_mouthroi = torch.FloatTensor(segment_mouthroi).unsqueeze(0).unsqueeze(0).cuda()
		reconstructed_signal = get_separated_audio(net_lipreading, net_facial_attributes, net_unet, audio_spec, segment_mouthroi, frames, opt)
		reconstructed_signal = reconstructed_signal * normalizer
		sep_audio[-samples_per_window:] = sep_audio[-samples_per_window:] + reconstructed_signal

		#update overlap count
		overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1
		
		#divide the aggregated predicted audio by the overlap count
		avged_sep_audio =  clip_audio(np.divide(sep_audio, overlap_count))

		#output separated audios
		if not os.path.isdir(opt.output_dir_root):
			os.mkdir(opt.output_dir_root)
		librosa.output.write_wav(os.path.join(opt.output_dir_root, 'speaker' + str(speaker_index+1) + '.wav'), avged_sep_audio, opt.audio_sampling_rate)

if __name__ == '__main__':
    main()
