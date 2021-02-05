#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Acknowledgement: Codes are borrowed from Bruno Korbar

import av
import gc
import warnings

_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 20

# remove warnings
av.logging.set_level(av.logging.ERROR)

class VideoReader(object):
    """
    Simple wrapper around PyAV that exposes a few useful functions for
    dealing with video reading.
    """
    def __init__(self, video_path, sampling_rate=1, decode_lossy=False, audio_resample_rate=None):
        """
        Arguments:
            video_path (str): path of the video to be loaded
        """
        self.container = av.open(video_path)
        self.sampling_rate = sampling_rate
        self.resampler = None
        if audio_resample_rate is not None:
            self.resampler = av.AudioResampler(rate=audio_resample_rate)
        
        if self.container.streams.video:
            # enable multi-threaded video decoding
            if decode_lossy:
                warnings.warn('VideoReader| thread_type==AUTO can yield potential frame dropping!', RuntimeWarning)
                self.container.streams.video[0].thread_type = 'AUTO'
            self.video_stream = self.container.streams.video[0]
        else:
            self.video_stream = None
 
    def seek(self, offset, backward=True, any_frame=False):
        stream = self.video_stream
        self.container.seek(offset, any_frame=any_frame, backward=backward, stream=stream)

    def _occasional_gc(self):
        # there are a lot of reference cycles in PyAV, so need to manually call
        # the garbage collector from time to time
        global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
        _CALLED_TIMES += 1
        if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
            gc.collect()

    def _read_video(self, offset, num_frames):
        self._occasional_gc()
        self.seek(offset, backward=True, any_frame=False)
        video_frames = []
        count = 0
        for idx, frame in enumerate(self.container.decode(video=0)):
            if frame.pts < offset:
                continue
            video_frames.append(frame)
            if count >= num_frames - 1:
                break
            count += 1
        return video_frames
    
    def _read_entire_audio(self, offset):
        # read all the rest of the audio starting from offset
        self._occasional_gc()
        if not self.container.streams.audio:
            return []
        video_time_base =  self.container.streams.video[0].time_base
        audio_time_base = self.container.streams.audio[0].time_base
        audio_offset = int(offset * video_time_base / audio_time_base)
        self.container.seek(audio_offset, backward=True, any_frame=False, stream=self.container.streams.audio[0])
        audio_frames = []
        for idx, frame in enumerate(self.container.decode(audio=0)):
            if self.resampler is not None:
                frame = self._resample_audio_frame(frame)
            audio_frames.append(frame)
        return audio_frames

    def read(self, offset, num_frames):
        if self.container is None:
            return [], []
        num_frames = self.sampling_rate * num_frames
        video_frames = self._read_video(offset, num_frames)
        try:
            audio_frames = self._read_entire_audio(offset)
        except av.AVError:
            audio_frames = []
        #hacky way to deal with special cases
        if len(video_frames) == 0:
            video_frames = self._read_video(0, num_frames)
            audio_frames = self._read_entire_audio(0)
        #hacky way to fill in the video_frames by copying the last frame
        while len(video_frames) != num_frames:
            video_frames.append(video_frames[-1])
        return video_frames, audio_frames

    def _compute_video_stats(self):
        if self.video_stream is None or self.container is None:
            return 0
        num_of_frames = self.container.streams.video[0].frames
        self.seek(0, backward=False)
        count = 0
        time_base = 512
        for p in self.container.decode(video=0):
            count = count + 1
            if count == 1:
                start_pts = p.pts
            elif count == 2:
                time_base = p.pts - start_pts
                break
        return start_pts,time_base,num_of_frames