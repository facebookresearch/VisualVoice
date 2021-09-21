#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# A naive implementation of face tracking leveraging a face detector 

import os
import argparse
import utils
import face_alignment
from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw

def face2head(boxes, scale=1.5):
    new_boxes = []
    for box in boxes:
        width = box[2] - box[0]
        height= box[3] - box[1]
        width_center = (box[2] + box[0]) / 2
        height_center = (box[3] + box[1]) / 2
        square_width = int(max(width, height) * scale)
        new_box = [width_center - square_width/2, height_center - square_width/2, width_center + square_width/2, height_center + square_width/2]
        new_boxes.append(new_box)
    return new_boxes

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--detect_every_N_frame', type=int, default=8)
    parser.add_argument('--scalar_face_detection', type=float, default=1.5)
    parser.add_argument('--number_of_speakers', type=int, default=2)
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    utils.mkdirs(os.path.join(args.output_path, 'faces'))

    landmarks_dic = {}
    faces_dic = {}
    boxes_dic = {}
    for i in range(args.number_of_speakers):
        landmarks_dic[i] = []
        faces_dic[i] = []
        boxes_dic[i] = []

    mtcnn = MTCNN(keep_all=True, device=device)
    
    video = mmcv.VideoReader(args.video_input_path)
    print("Video statistics: ", video.width, video.height, video.resolution, video.fps)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    print('Number of frames in video: ', len(frames))
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')
        
        # Detect faces
        if i % args.detect_every_N_frame == 0:
            boxes, _ = mtcnn.detect(frame)
            boxes = boxes[:args.number_of_speakers]
            boxes = face2head(boxes, args.scalar_face_detection)
        else:
            boxes = [boxes_dic[j][-1] for j in range(args.number_of_speakers)]

        # Crop faces and save landmarks for each speaker
        if len(boxes) != args.number_of_speakers:
            boxes = [boxes_dic[j][-1] for j in range(args.number_of_speakers)]
        
        for j,box in enumerate(boxes):
            face = frame.crop((box[0], box[1], box[2], box[3])).resize((224,224))
            preds = fa.get_landmarks(np.array(face))
            if i == 0:
                faces_dic[j].append(face)
                landmarks_dic[j].append(preds)
                boxes_dic[j].append(box)
            else:
                iou_scores = []
                for b_index in range(args.number_of_speakers):
                    last_box = boxes_dic[b_index][-1]
                    iou_score = bb_intersection_over_union(box, last_box)
                    iou_scores.append(iou_score)
                box_index = iou_scores.index(max(iou_scores))
                faces_dic[box_index].append(face)
                landmarks_dic[box_index].append(preds)
                boxes_dic[box_index].append(box)
    
    for s in range(args.number_of_speakers):
        frames_tracked = []
        for i, frame in enumerate(frames):
            # Draw faces
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            draw.rectangle(boxes_dic[s][i], outline=(255, 0, 0), width=6) 
            # Add to frame list
            frames_tracked.append(frame_draw)
        dim = frames_tracked[0].size
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
        video_tracked = cv2.VideoWriter(os.path.join(args.output_path, 'video_tracked' + str(s+1) + '.mp4'), fourcc, 25.0, dim)
        for frame in frames_tracked:
            video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video_tracked.release()

    # Save landmarks
    for i in range(args.number_of_speakers):    
        utils.save2npz(os.path.join(args.output_path, 'landmark', 'speaker' + str(i+1)+'.npz'), data=landmarks_dic[i])
        dim = face.size
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
        speaker_video = cv2.VideoWriter(os.path.join(args.output_path, 'faces', 'speaker' + str(i+1) + '.mp4'), fourcc, 25.0, dim)
        for frame in faces_dic[i]:
            speaker_video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        speaker_video.release()

    # Output video path
    parts = args.video_input_path.split('/')
    video_name = parts[-1][:-4]
    if not os.path.exists(os.path.join(args.output_path, 'filename_input')):
        os.mkdir(os.path.join(args.output_path, 'filename_input'))
    csvfile = open(os.path.join(args.output_path, 'filename_input', str(video_name) + '.csv'), 'w')
    for i in range(args.number_of_speakers):
        csvfile.write('speaker' + str(i+1)+ ',0\n')
    csvfile.close()


if __name__ == '__main__':
    main()
