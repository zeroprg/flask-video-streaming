#!/usr/bin/env python
from importlib import import_module
import os
import threading
from flask import Flask, render_template, Response
import time
from screen_statistics import Screen_statistic
#initialise Screen statistic object
scrn_stats = Screen_statistic()

import pi_object_detection as object_detector

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera


# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)

@app.before_first_request
def activate_job():
    def run_job():
        p_get_frame = object_detector.start()
        p_get_frame.join()
        while True:
            print("Run recurring task")
            time.sleep(3)

    thread = threading.Thread(target=run_job)
    thread.start()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect():
    """Video streaming generator function."""
    i=0
    while True:
        #frame = camera.get_frame()
        time.sleep(1)
        iterable = object_detector.imgs[i]
        i +=1
        i %= object_detector.MAX_BUFFER
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
        yield iterable
        yield b'\r\n'

def gen_push():
    """Parameters streaming generator function."""
    while True:
        time.sleep(1)
        #uncomment as soon will be ready
        #x = ( int(time.time()) % 3 )
        #parameter = (fields_st).encode('ascii') 
        #yield ( parameter )
  

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    # gen(Camera()),
    return Response( detect(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/params_feed')
def params_feed():
    """Parameters streaming route. Put this in the src attribute of an img tag."""
    return Response( detect(), #gen_push(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':

    #app.run(host='0.0.0.0', threaded=True)
    # debug mode    
    app.run(debug=True, use_debugger=False, use_reloader=False)
    
