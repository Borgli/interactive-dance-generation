# Interactive Dance Generation

The repository for ... (we will see what happens to this)

## Installation instructions

If you want to use this application download the dependencies from the ```requirements.txt```. Most notably, you will need to install ```flask```, ```pytorch```, and ```detectron2``` (https://github.com/facebookresearch/detectron2).

The models for inference can be downloaded from:\
|        |            | 
|--------|------------|
| 15 FPS FACT| https://drive.google.com/file/d/1P6RoqLPbPKdnEl8cMfBPFFXjyZqQIvCS/view?usp=share_link |
| 30 FPS FACT| https://drive.google.com/file/d/11_27OuwQB0bAzWuOXreRoLJvHRRNgBRS/view?usp=share_link |

The models should be placed in a ```./checkpoints``` in the main repository.

## How to use

After the requirements are installed and the models downloaded. You need a webcam (or camera) connected to your computer. A decent GPU - this is not required, but highly advised to allow for fast inference.

Start the program by
```
python app.py -f 30 -device cuda:0
```
where ```-f``` denotes the desired fps and accepts 30 and 15 (default 30), and ```-device``` denotes the devide the model should be run on (default cuda:0). Note, 15 fps runs faster than 30 fps and *gpu* (cuda:x) is highly recommended over *cpu*.

With all of that taken care of, get a large space, and make sure your entire body fits in the video frame. It is important that your entire body fits within the frame to allow for stable predictions. Lastly, hit the record button and enjoy. You are given 10 seconds to get in place and dance, your last 3-4 seconds will be used for dance prediction. 