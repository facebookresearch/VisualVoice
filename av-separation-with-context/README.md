This subdirectory contains the code for our model tailored to two-speaker speech separation: we concatenate the visual features of both speakers in the mixture with the audio feature to guide the separation process. This leads to slightly better performance due to the additional context information of the other speaker provided. The demo/training/testing procedures are similar to the version without context.

### Pre-trained models
```
wget http://dl.fbaipublicfiles.com/VisualVoice/av-speech-separation-model-with-context/facial_best.pth
wget http://dl.fbaipublicfiles.com/VisualVoice/av-speech-separation-model-with-context/lipreading_best.pth
wget http://dl.fbaipublicfiles.com/VisualVoice/av-speech-separation-model-with-context/unet_best.pth
wget http://dl.fbaipublicfiles.com/VisualVoice/av-speech-separation-model-with-context/vocal_best.pth
```
