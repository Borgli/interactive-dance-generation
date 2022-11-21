
"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""
import pickle

import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob

FRAME_FEEDBACK = [0, 0]


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


class Infer(object):

    def __init__(
            self,
            dance_model,
            cfg_model,
            fps: int = 30,
            seq_length: int = 20,
            inp_frames: int = 90,
            device: str = "cuda:0",
            cut: int = 5,
    ) -> None:

        self.fps = fps
        self.seq_length = seq_length
        self.inp_frames = inp_frames
        self.device = device
        self.cut = cut
        self.dance_model = dance_model.to(self.device)

        with open('music/music_features.pickle', 'rb') as handle:
            self.music_features = pickle.load(handle)

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg_model))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_model)
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)

    def get_resolution(self, filename):
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)

    def read_video(self, filename):
        w, h = self.get_resolution(filename)
        # w, h = 1920, 1080

        command = ['ffmpeg',
                   '-i', filename,
                   '-f', 'image2pipe',
                   '-pix_fmt', 'bgr24',
                   '-vsync', '0',
                   '-vcodec', 'rawvideo', '-']

        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))

    def infer_keypoints(self, im_or_folder):
        global FRAME_FEEDBACK
        keypoints = list()
        sscores = list()

        if os.path.isdir(im_or_folder):
            im_list = glob.iglob(im_or_folder + '/*.mp4')
        else:
            im_list = [im_or_folder]

        for video_name in im_list:
            # out_name = os.path.join(
            # output_dir, os.path.basename(video_name)
            # )
            video_frames = list(self.read_video(video_name))
            video_length = len(video_frames)
            print('Processing {}'.format(video_name))
            print(f"Total of {self.inp_frames} frames")
            FRAME_FEEDBACK = [0, self.inp_frames]

            boxes = []
            segments = []

            assert len(
                video_frames) - self.cut >= self.inp_frames + self.cut, "The number of recorded frames are less than what the model expects"
            # Cut out the last 5 frames
            for frame_i, im in enumerate(video_frames[-self.inp_frames - self.cut: - self.cut]):
                t = time.time()
                outputs = self.predictor(im)['instances'].to('cpu')

                print('Frame {}/{} processed in {:.3f}s'.format(frame_i, video_length, time.time() - t))

                FRAME_FEEDBACK = [frame_i, self.inp_frames]

                has_bbox = False
                if outputs.has('pred_boxes'):
                    bbox_tensor = outputs.pred_boxes.tensor.numpy()
                    if len(bbox_tensor) > 0:
                        has_bbox = True
                        scores = outputs.scores.numpy()[:, None]
                        bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
                if has_bbox:
                    kps = outputs.pred_keypoints.numpy()[np.argmax(scores[:, 0])]  # [N,num_keypoint,(x,y,score)]
                    kps = np.expand_dims(kps, 0)
                    w = im.shape[1] / 2
                    h = im.shape[0] / 2
                    kps_x = (kps[:, :, 0] - w) / w
                    kps_y = (kps[:, :, 1] - h) / h
                    kps_xy = np.concatenate((kps_x, kps_y), axis=0)
                    kps_xy = np.transpose(kps_xy)
                else:
                    kps = []
                    bbox_tensor = []

                # Mimic Detectron1 format
                cls_boxes = [[], bbox_tensor]
                cls_keyps = [[], kps]

                boxes.append(cls_boxes)
                segments.append(None)
                keypoints.append(kps_xy)

            # Video resolution
            metadata = {
                'w': im.shape[1],
                'h': im.shape[0],
            }
        return np.array(keypoints), video_length

    def infer_dance(self, keypoints: np.ndarray, music: str, video_length: int):
        global FRAME_FEEDBACK
        start = keypoints
        keypoints = torch.from_numpy(keypoints)
        keypoints = keypoints.reshape(self.inp_frames, 34).unsqueeze(0).to(self.device)

        factor = 60 / self.fps  # Original fps divided by video fps

        mf = self.music_features[music]
        music_features = mf[int(factor * video_length - self.inp_frames - self.cut)::int(factor)]

        inp = {
            "audio_input": torch.from_numpy(music_features).unsqueeze(0).to(self.device),
            "motion_input": keypoints,
        }
        self.dance_model.eval()
        with torch.no_grad():
            out = self.dance_model.infer_auto_regressive(inp, steps=int(self.seq_length * self.fps), step_size=1)
            out = out.to("cpu").squeeze(0).reshape(int(self.seq_length * self.fps), 17, 2).numpy()
        seq = np.concatenate([start, out], axis=0)
        FRAME_FEEDBACK = [FRAME_FEEDBACK[0] + 1, FRAME_FEEDBACK[1]]
        return seq

    def infer(self, im_or_folder, music):
        keypoints, video_length = self.infer_keypoints(im_or_folder)
        dance = self.infer_dance(keypoints, music, video_length)
        with open('videos/keypoints.json', 'w') as f:
            json.dump(dance.tolist(), f)


if __name__ == '__main__':
    from app import fetch_model
    import json

    model = fetch_model("/home/jon/2D/30fps/2022-11-13/epoch_150/checkpoint-epoch150.pth")
    infer = Infer(
        dance_model=model,
        cfg_model="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
        fps=30,
        seq_length=20,
        inp_frames=90,
        device="cuda:0",
        cut=5,
    )
    keypoints, video_length = infer.infer_keypoints("videos/")
    dance = infer.infer_dance(keypoints=keypoints, video_length=video_length, music="mBR0", )
    with open('videos/keypoints.json', 'w') as f:
        json.dump(dance.tolist(), f)
