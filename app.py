from flask import Flask, Response, render_template, jsonify
import tensorflow as tf
import cv2
import time
import argparse
import pafy
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

app = Flask(__name__)
video = cv2.VideoCapture(0)

keypoint1 = ""
keypoint2 = ""

# ////////////////////////////////////////////////////////////////////////
def webcam_main_vid():
    with tf.compat.v1.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
            # cap=cv2.VideoCapture(1)
        else:
            cap = cv2.VideoCapture(args.cam_id)
            # cap=cv2.VideoCapture(1)

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords1 = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            # for ii, score in enumerate(pose_scores):
            #     print("************")
            #     print(ii, "----------", score)

            global keypoint1
            keypoint_coords1 *= output_scale
            keypoint1 = np.array(keypoint_coords1[0])

            overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords1,
            min_pose_score=0.15, min_part_score=0.1)

            # overlay_image = cv2.resize(overlay_image, (0,0), fx=0.8, fy=0.8)
            
            ret, jpeg = cv2.imencode('.jpg', overlay_image)

            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # yield "data:" + keypoint1 + "\n\n"

# ////////////////////////////////////////////////////////////////////////
def recorded_main_vid():
    with tf.compat.v1.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        url = "https://www.youtube.com/watch?v=2HTvZp5rPrg&t=7s"
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture()
        cap.open(best.url)
        cap.set(cv2.CAP_PROP_FPS, int(30))

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            global keypoint2

            pose_scores, keypoint_scores, keypoint_coords2 = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)
            
            keypoint_coords2 *= output_scale
            keypoint2 = np.array(keypoint_coords2[0])
            # print("camera", np.array(keypoint_coords2[0]).shape)

            # for ii, score in enumerate(pose_scores):
            #     print("************")
            #     print(ii, "----------", score)
            # print(pose_scores)

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords2,
                min_pose_score=0.15, min_part_score=0.1)

            # overlay_image = cv2.resize(overlay_image, (0,0), fx=0.8, fy=0.8)

            ret, jpeg = cv2.imencode('.jpg', overlay_image)

            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # print('Average FPS: ', frame_count / (time.time() - start))

# ////////////////////////////////////////////////////////////////////////

def webcam_main_key():
    compare = 0
    for i in range(17):
        for j in range(2):
            compare += (keypoint1[i][j] - keypoint2[i][j])

    score = compare/34
    yield "data:" + str(score) + "\n\n"
 
# ////////////////////////////////////////////////////////////////////////

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam_route_key')
def webcam_sub_key():
    # return Response(webcam_main(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(webcam_main_key(), mimetype='text/event-stream')
    # return  '{} {}'.format(Response(webcam_main(), mimetype='text/event-stream'), "lastname")

@app.route('/webcam_route_vid')
def webcam_sub_vid():
    return Response(webcam_main_vid(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recorded_route')
def recorded_sub():
    return Response(recorded_main_vid(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    app.run(debug=True)

# frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)