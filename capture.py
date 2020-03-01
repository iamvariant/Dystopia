# USAGE
# python capture.py --ip 0.0.0.0 --port 8000

# import the necessary packages
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from imutils.video import VideoStream
from flask import Flask, render_template, Response
import threading
import argparse
import datetime
import imutils
import time
import cv2
import backend
from backend import preprocess_image, extract_faces, add_face, remove_face, identify_face, identify_all, facedetect

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def capture_frame():
    # grab global references to the video stream, output frame, and
     # lock variables
    global outputFrame, lock

    while(True):
        ret, frame = cap.read()
        if frame is None: print("help lah")
        else: print(frame.shape)
        # frame = cv2.flip(frame, 1)
        # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
    return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
        # # construct the argument parser and parse command line arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--ip", type=str, required=True,
        #     help="ip address of the device")
        # ap.add_argument("-o", "--port", type=int, required=True,
        #     help="ephemeral port number of the server (1024 to 65535)")
        # ap.add_argument("-f", "--frame-count", type=int, default=32,
        #     help="# of frames used to construct the background model")
        # args = vars(ap.parse_args())

        # start a thread that will perform frame capture
        t = threading.Thread(target=capture_frame)
        t.daemon = True
        t.start()
        
        # start the flask app
        app.run(debug=True)

