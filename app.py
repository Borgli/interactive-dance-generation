import json
import os
import time
from pathlib import Path
from threading import Thread

import cv2
import torch
from flask import Flask, render_template, Response, request, send_file

import infer_keypoints as ik
from fact.config import audio_config, motion_config, multi_model_config
from fact.fact import FACTModel
from infer_keypoints import Infer

global rec_frame, switch, rec, out, music_id, video_name, device

switch = 1
rec = 0
music_id = -1
video_name = ''
device = "cuda:0"


def fetch_model(model_path: str, fps: int = 30, out_dim: int = 34, pred_length: int = 20):
    audio_config.transformer.intermediate_size = 1024
    motion_config.transformer.intermediate_size = 1024
    multi_model_config.transformer.intermediate_size = 1024
    multi_model_config.transformer.num_hidden_layers = 4

    motion_config.feature_dim = 34

    if fps == 30:
        # 30 FPS
        audio_config.sequence_length = 180
        motion_config.sequence_length = 90
        multi_model_config.sequence_length = 180
    elif fps == 15:
        # 15 FPS
        audio_config.sequence_length = 120
        motion_config.sequence_length = 60
        multi_model_config.sequence_length = 120
    else:
        # 10 FPS
        audio_config.sequence_length = 80
        motion_config.sequence_length = 40
        multi_model_config.sequence_length = 80

    model = FACTModel(audio_config, motion_config, multi_model_config, out_dim=out_dim, pred_length=pred_length)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device)


model = fetch_model("/home/jon/2D/30fps/2022-11-13/epoch_150/checkpoint-epoch150.pth")
global infer
infer = Infer(
    dance_model=model,
    cfg_model="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
    fps=30,
    seq_length=20,
    inp_frames=90,
    device="cuda:0",
    cut=5,
)

model.eval()
# make directories to save video and keypoints
try:
    os.mkdir('./videos')
    os.mkdir('./keypoints')
except OSError as error:
    pass

# instantiate flask app  
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while (rec):
        time.sleep(1 / 30)  # 20 FPS
        out.write(cv2.cvtColor(rec_frame, cv2.COLOR_BGR2RGB))


def return_dict():
    # Dictionary to store music file information TODO: add more music
    dict_here = [
        {'id': 1, 'name': 'Hip Hop', 'link': 'music/mMH0.wav', 'genre': 'General'},
        {'id': 2, 'name': 'House', 'link': 'music/mHO0.wav', 'genre': 'General'},
        {'id': 3, 'name': 'Lock', 'link': 'music/mLO0.wav', 'genre': 'General'},
        {'id': 4, 'name': 'Break', 'link': 'music/mBR0.wav', 'genre': 'General'},
    ]
    return dict_here


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if (rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


def generate(stream_id):  # generate sound
    # global stream_id
    data = return_dict()
    for item in data:
        if item['id'] == stream_id:
            song = item['link']
    with open(song, "rb") as fwav:
        data = fwav.read(1024)
        while data:
            yield data
            data = fwav.read(1024)


# Route to render GUI
@app.route('/')
def index():
    stream_entries = return_dict()
    songs = [[entry["id"], f"{entry['genre']}: {entry['name']}"] for entry in stream_entries]
    return render_template('index.html', entries=stream_entries, songs=songs)


@app.route('/results')
def results():
    return render_template('results.html', music_id=music_id)


# Route to render video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to manipulate video
@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global rec, switch, camera, music_id
    stream_entries = return_dict()
    songs = [[entry["id"], f"{entry['genre']}: {entry['name']}"] for entry in stream_entries]

    if request.method == 'POST':
        if request.form.get('stop') == 'Stop/Start':
            if switch:  # stop video stream
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:  # start video stream
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') is not None:
            global out, video_name
            rec = not rec
            if rec:
                music_id = int(request.form.get('songindex'))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_name = os.path.sep.join(['videos', 'video.mp4'])
                w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(video_name, fourcc, 20.0, (w, h))
                thread = Thread(target=record, args=[out, ])
                thread.start()
                # TODO: play music at the same time
            else:
                out.release()
                input_dir = video_name
                keypoint_thread = Thread(target=infer.infer,
                                         args=[input_dir, Path(stream_entries[music_id - 1]["link"]).stem])
                keypoint_thread.start()

                return render_template('index.html', entries=stream_entries,
                                       songs=songs, loading=True, music_id=music_id)

        else:
            music_id = int(request.form.get('songindex'))

    elif request.method == 'GET':
        return render_template('index.html', entries=stream_entries, songs=songs, music_id=music_id)

    return render_template('index.html', entries=stream_entries, songs=songs, rec=rec, music_id=music_id)


@app.route('/poll')
def poll_keypoints_processing():
    return Response(json.dumps(ik.FRAME_FEEDBACK))


@app.route('/get_keypoints')
def return_keypoints():
    with open("videos/keypoints.json") as f:
        return Response(f.read())


# Route to stream music
@app.route('/<int:stream_id>')
def streamwav(stream_id):
    return Response(generate(stream_id), mimetype="audio/wav")


@app.route('/music/<int:sound_id>')
def get_music_file(sound_id):
    stream_entries = return_dict()
    music_file = stream_entries[sound_id - 1]
    return send_file(music_file["link"])


if __name__ == '__main__':
    try:
        app.run()
    finally:
        camera.release()
        cv2.destroyAllWindows()
