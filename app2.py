# Credit: We used https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
# as an example for creating this application!

# USAGE
# python app2.py --ip localhost --port 5000

import argparse
import datetime
import threading
import time
import numpy as np
import tensorflow
from flask_login import login_required
import IoT_influxdb
import send_sms
import easygui
from PIL import Image, ImageOps
from flask import Flask, render_template, Response, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
import os
from imutils.video import VideoStream
from datetime import timedelta

import cv2

# @app.shell_context_processor
# def make_shell_context():
#     return {'db': app.db, 'User': User, 'Post': Post}

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
touchlock = 0

outputFrame = None
lock = threading.Lock()

from auth_decorator import login_required
# dotenv setup
from dotenv import load_dotenv

load_dotenv()

# initialize a flask object
app = Flask(__name__)

# For sign feature /////////////////////////////////////////////////////////////////////////
# Session config
app.secret_key = ''
app.config['SESSION_COOKIE_NAME'] = 'google-login-session'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)
# oauth config
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='',
    client_secret='',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    # This is only needed if using openId to fetch user info
    client_kwargs={'scope': 'openid email profile'},
)
# //////////////////////////////////////////////////////////////////////////////////////////////

# initialize the video stream
cap = VideoStream(src=0).start()
time.sleep(2.0)

# initialize to nothing
warningText = ""

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


@app.route("/")
@login_required
def index():
    # return the rendered template
    email = dict(session)['profile']['email']
    easygui.msgbox(f'Hello, you are logged in as {email}!', title="simple gui")
    return render_template("index.html")
    # return f'Hello, you are logged in as {email}!'
    # return render_template("index.html")


# For SignIn feature /////////////////////////////////////////////////////
@app.route('/login')
def login():
    google = oauth.create_client('google')  # create the google oauth client
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/authorize')
def authorize():
    google = oauth.create_client('google')  # create the google oauth client
    token = google.authorize_access_token()
    # resp = oauth.twitter.get('account/verify_credentials.json')
    resp = google.get('userinfo', token=token)
    user_info = resp.json()
    # do something with the token and profile
    user = oauth.google.userinfo()  # uses openid endpoint to fetch user info
    # Here you use the profile/user data that you got and query your database find/register the user
    # and set ur own data in the session not the profile from google
    session['profile'] = user_info
    session.permanent = True  # make the session permanant so it keeps existing after broweser gets closed
    return redirect('/')


# ////////////////////////////////////////////////////////////////////////

def touched_face():
    IoT_influxdb.on_message(1)
    send_sms.sendMessage()


def stopped_touching_face():
    IoT_influxdb.on_message(0)


def stream_video():
    # grab global references to the video stream, output frame, and
    # lock variables
    global cap, outputFrame, lock, touchlock, warningText

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream
        frame = cap.read()

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # put warning text on frame if they touched their face
        cv2.putText(frame, warningText, (125, frame.shape[0] - 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # normalize the captured frame
        normalizedframe = np.zeros((800, 800))
        normalizedframe = cv2.normalize(frame, normalizedframe, 0, 255, cv2.NORM_MINMAX)

        # Convert the frame into a PIL image so that it can be used by the TensorFlow Model
        im = Image.fromarray(np.uint8(normalizedframe) * 255)

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(im, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference model to determine if the user touched their face or not
        prediction = model.predict(data)

        # If the user touches their face---------------------------------------------------------
        if touchlock == 0:
            if prediction[-1][0] < .9:
                touched_face()
                touchlock = 1
                warningText = "FACE TOUCH WARNING!!!!"
        else:
            if prediction[-1][0] > .9:
                stopped_touching_face()
                touchlock = 0
                warningText = ""

        # print results of model
        print(prediction)

        # acquire the lock, set the output frame, pass the frame to the model, and release the lock
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
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform video capture streaming
    t = threading.Thread(target=stream_video)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
cap.stop()


@app.route('/logout')
def logout():
    for key in list(session.keys()):
        session.pop(key)
    return redirect('/')
# if __name__ == '__main__':
#    app.run()
