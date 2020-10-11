# import the necessary packages

from multiprocessing import Process
from multiprocessing import Queue
import os
import time
import logging
import json
import db

from classifier_yolo import Detection

from flask import Flask, render_template, Response, request, redirect, jsonify, send_from_directory
from flask_cors import cross_origin, CORS

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)
logger.debug('DEBUG mode')


###################### Flask API #########################
app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/urls": {"origins": "http://localhost:5000"}})


@app.route('/static/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static', 'js'), filename)


@app.route('/')
def index():
    """Video streaming home page."""
    start = request.args.get('start')
    if start == None:
        start = 0
    start = int(start)
    video_urls = []
    img_paginator = IMG_PAGINATOR
    conn = db.create_connection(SQLITE_DB)
    images_filenames = []
    for i in range(len(videos)):
        video_urls.append((videos[i][0], 'video_feed?cam=' + str(videos[i][0])))
        images_filenames.append(db.select_last_frames(conn, i, IMG_PAGINATOR))
    logger.info("video_urls: {}".format(video_urls))
    """ Delete old frames wil be here """
    resp =  Response(render_template('index.html', **locals()), mimetype='text/html')
    resp.set_cookie('SameSite', 'None', secure=True)
    return resp


@app.route('/moreparams')
def moreparams():
    """ Read list of json files or return one specific  for specific time """
    hour_back1 = request.args.get('hour_back1', default=1, type=int)
    hour_back2 = request.args.get('hour_back2', default=0, type=int)
    cam = request.args.get('cam', default=0, type=int)
    if hour_back1 != '':
        hour_back1 = int(hour_back1)
    else:
        hour_back1 = 1  # default value: 60 min back
    now_in_seconds = time.time()*1000
    if hour_back2 != '': now_in_seconds = now_in_seconds - int(hour_back2) * 60 * 60
    print("cam: {}, hour_back:{}, now_in_seconds:{}".format(cam, hour_back1, now_in_seconds))

    params = gen_params(cam=cam, hours=hour_back1, currentime=now_in_seconds)
    return Response(params, mimetype='text/plain')


@app.route('/moreimgs')
def moreimgs():
    """ Read list of json files or return one specific  for specific time """
    direction = request.args.get('direction', default=1, type=int)
    start = request.args.get('start', default=0, type=int)
    cam = request.args.get('cam', default=0, type=int)

    if direction < 0:
        if start >= IMG_PAGINATOR:
            start = start - IMG_PAGINATOR
        else:
            start = 0
    conn = db.create_connection(SQLITE_DB)
    rows = db.select_last_frames(conn, cam, IMG_PAGINATOR, offset=start, as_json=True)
    #    ret = json.dumps(rows)
    return Response(rows, mimetype='text/plain')


@app.route('/imgs_at_time')
def imgs_at_time():
    """ Read list of json files or return one specific  for specific time """
    seconds = request.args.get('time', default=time.time(), type=int)
    delta = request.args.get('delta', default=10, type=int)
    cam = request.args.get('cam', default=0, type=int)
    return Response(gen_array_of_imgs(cam, delta=delta, currentime=seconds), mimetype='text/plain')


def gen_array_of_imgs(cam, delta=10, currentime=time.time()*1000):
    time1 = currentime - delta
    time2 = currentime + delta
    conn = db.create_connection(SQLITE_DB)
    rows = db.select_frame_by_time(conn, cam, time1, time2)
    x = json.dumps(rows)
    return x


def gen(camera):
    """Video streaming generator function."""
    try:
        while True:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except GeneratorExit:
        pass


def detect(cam):
    """Video streaming generator function."""
    label = ''
    try:
        # logger.debug('imagesQueue:', imagesQueue.empty())
        while True:
            while (not imagesQueue[cam].empty()):
                frame = imagesQueue[cam].get(block=True)
                iterable = cv2.imencode('.jpg', frame)[1].tobytes()
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + iterable + b'\r\n'
    except GeneratorExit:
        pass


def gen_params(cam=0, hours=1, currentime=time.time()):
    """Parameters streaming generator function."""
    time1 = currentime - hours * 60 * 60
    print("time1: {} now: {}".format(time1, currentime))
    conn = db.create_connection(SQLITE_DB)
    ls = db.select_statistic_by_time(conn, cam, time1, currentime)
    ret = json.dumps(ls)  # , indent = 4)
    logger.debug(ret)
    return ret


def ping_video_url(url):
    """ Ping url """
    try:
        vs = cv2.VideoCapture(url)
        flag, frame = vs.read()
        ret = flag
    except:
        ret = False
    return flag


@app.route('/urls', methods=['GET', 'POST'])
@cross_origin(origin='http://localhost:5000')
def urls():
    """Add/Delete/Update a new video url, list all availabe urls."""
    list_url = request.args.get('list', default=None)
    add_url = request.args.get('add', default=None)
    delete_url = request.args.get('delete', default=None)
    update_url = request.args.get('update', default=None)
    if add_url is not None:
        if ping_video_url(add_url):
            initialize_video_streams(add_url)
            start_one_stream_processes(cam=len(videos) - 1)
            # return index() #redirect("/")
            return Response('{"message":"URL added  successfully , video start processing"}', mimetype='text/plain')
    if list_url is not None:
        return Response(json.dumps(videos), mimetype='text/plain')
    if delete_url is not None:
        for video in videos:
            if video[0] == delete_url:
                videos.remove(video)
                return Response('{"message":"URL deleted successfully"}', mimetype='text/plain')
    if update_url is not None:
        index = request.args.get('index', default=None)
        if index is not None:
            videos[index][1] == update_url
            return Response('{"message":"URL updated successfully"}', mimetype='text/plain')


@app.route('/video_feed', methods=['GET'])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    # gen(Camera()),
    cam = request.args.get('cam', default=0, type=int)
    return Response(detect(int(cam)),  # mimetype='text/event-stream')
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/params_feed')
def params_feed():
    """Parameters streaming route. Put this in the src attribute of an img tag."""
    hours = request.args.get('hour_back1', default=1)
    start_hour = request.args.get('hour_back2', default=0)
    currentime = time.time() - int(start_hour) * 60 * 60
    return Response(gen_params(hours, currentime=currentime),
                    mimetype='text/plain')



if (__name__ == '__main__'):
    start()
    app.run(host='0.0.0.0', threaded=True)  # debug = True ) #
