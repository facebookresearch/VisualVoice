#!/bin/bash
ffmpeg -i ./test_videos/noisy_restaurant.mp4 -filter:v fps=fps=25 ./test_videos/noisy_restaurant25fps.mp4
mv ./test_videos/noisy_restaurant25fps.mp4 ./test_videos/noisy_restaurant.mp4
python ./utils/detectFaces.py --video_input_path ./test_videos/noisy_restaurant.mp4 --output_path ./test_videos/noisy_restaurant/ --number_of_speakers 1 --scalar_face_detection 1.3 --detect_every_N_frame 25
ffmpeg -i ./test_videos/noisy_restaurant.mp4 -vn -ar 16000 -ac 1 -ab 192k -f wav ./test_videos/noisy_restaurant/noisy_restaurant.wav
python ./utils/crop_mouth_from_video.py --video-direc ./test_videos/noisy_restaurant/faces/ --landmark-direc ./test_videos/noisy_restaurant/landmark/ --save-direc ./test_videos/noisy_restaurant/mouthroi/ --convert-gray --filename-path ./test_videos/noisy_restaurant/filename_input/noisy_restaurant.csv
