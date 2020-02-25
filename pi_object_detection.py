
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue
from files import *

import os
from urllib.request import urlopen
import threading
import io
import imagehash
import numpy as np
import argparse
import imutils
import time
import dhash
import glob
import logging

import sqlite3

from PIL import Image, ImageEnhance
from time import gmtime, strftime
import datetime
import cv2
import json

#from screen_statistics import Screen_statistic
import db
from objCountByTimer import ObjCountByTimer
import base64

from flask import Flask, render_template, Response, request,redirect,jsonify
from flask_cors  import cross_origin, CORS


logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)
logger.debug('DEBUG mode')


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
LOOKED1 = {"car":[],"person":[],"bus":[]}

subject_of_interes = ["car","person", "bus"]
hashes = {}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


DRAW_RECTANGLES = True
DELETE_FILES_LATER = 6*60*60 # sec  (8hours)
ENCODING = "utf-8"
NUMBER_OF_FILES = 10
HASH_DELTA = 60 # bigger number  more precise object's count
IMAGES_BUFFER = 45
RECOGNZED_FRAME = 1
THREAD_NUMBERS  = 5 #must be less then 4 for PI
videos = []
camleft = []
camright =[]
piCameraResolution = (1024,768) #(640,480)  #(1920,1080) #(1080,720) # (1296,972)
piCameraRate=10
IMG_PAGINATOR = 40
SQLITE_DB = "framedata.db"

class CameraMove:
    def __init__(self, move_left,move_right, timestep=10):
        if move_left==None or move_right == None: return 
        self.timestep = timestep
        self.move_left =  move_left   #'http://www.google.com' # move_left 
        self.move_right = move_right #'http://www.google.com' #move_right
        self.t1 = threading.Timer(timestep, self.cameraLoop)
        self.t1.start()
        
    def cameraLoop(self):
        logger.debug(self.move_left)
        os.system(self.move_left) #urlopen(self.move_left)
        time.sleep(5.0)
        os.system(self.move_left) #urlopen(self.move_left)
        time.sleep(5.0)
        os.system(self.move_left) #urlopen(self.move_left)
        time.sleep(10.0)
        os.system(self.move_right) #urlopen(self.move_right)
        time.sleep(5.0)
        os.system(self.move_right) #urlopen(self.move_right)
        time.sleep(5.0)
        os.system(self.move_right) #urlopen(self.move_right)
        time.sleep(2.0)
        time.sleep(20.0)
        self.t1 = threading.Timer(self.timestep, self.cameraLoop) 

        self.t1.start()


class Trace(dict):
    def __init__(self):
        dict.__init__(self)
        self.cam = 0
        self.x = 0
        self.y = 0
        self.name = ''
        self.text = ''
        self.filenames = []
    def toJSON(self):
            return json.dumps(self, default=lambda o: o.__dict__,
                sort_keys=True, indent=4)

def getParametersJSON(hashes, cam):
    ret =[]
    for key in hashes:
        #logging.debug(images[key])
        trace = Trace()
        trace.name = key
        trace.cam = cam
        tm = int(time.time()) #strftime("%H:%M:%S", localtime())
        trace.hashcodes = hashes[key].toString()
        trace.x = tm
        #last = len(hashes[key].counted) -1
        trace.y = hashes[key].getCountedObjects()
        trace.text =  str(trace.y )+ ' ' + key + '(s)'
        ret.append(trace.__dict__) # used for proper JSON generation (dictionary)
        #ret.append(trace)
        #logging.debug( trace.__dict__ )
    return ret



class ImageHashCodesCountByTimer(ObjCountByTimer):
    def equals(self,hash1, hash2):
        delta = dhash.get_num_bits_different(hash1, hash2)
        return delta < HASH_DELTA



def do_statistic(conn,cam,hashes):
  # Do some statistic work here
    params = getParametersJSON(hashes, cam)
    db.insert_statistic(conn,params)


DIMENSION_X = 300
DIMENSION_Y = 300

def classify_frame(inputQueue,rectanglesQueue, confidence, prototxt, model, cam):
        hashes = {}
        conn = db.create_connection(SQLITE_DB)
        # keep looping
        frame = None
        label2 = "No data"
        # child Thread reference
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        # specify the target device as the Myriad processor on the NCS
        #net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        p_classifier = None
        while True:

                # check to see if there is a frame in our input queue
                #while not inputQueue.empty():
                # grab the frame from the input queue, resize it, and
                # construct a blob from it
                #logger.debug('inputQueue.qsize()',inputQueue.qsize())
                #print(inputQueue)
                #print(rectanglesQueue)
                try:
                   frame = inputQueue.get(block=False)
                except: continue
                _frame = cv2.resize(frame, (DIMENSION_X, DIMENSION_Y))
                blob = cv2.dnn.blobFromImage(_frame, 0.007843,
                        (DIMENSION_X, DIMENSION_Y), (127.5,127.5,127.5), True)
                # set the blob as input to our deep learning object
                # detector and obtain the detections
                net.setInput(blob)
                detections = net.forward()
                # loop over the detections
                (fH, fW) = frame.shape[:2]
                #logger.debug(detections)
                if detections is not None:
                        # loop over the detections
                        for i in np.arange(0, detections.shape[2]):
                                # extract the confidence (i.e., probability) associated
                                # with the prediction
                                # filter out weak detections by ensuring the `confidence`
                                # is greater than the minimum confidence
                                if  detections[0, 0, i, 2] < confidence: continue

                                # otherwise, extract the index of the class label from
                                # the `detections`, then compute the (x, y)-coordinates
                                # of the bounding box for the object
                                idx = int(detections[0, 0, i, 1])
                                if idx > len(CLASSES)-1:continue
                                key = CLASSES[idx]
                                logger.debug("idx: " + str(idx) + " key:" + key)
                                if not key in LOOKED1: continue
                                dims = np.array([fW, fH, fW, fH])
                                box = detections[0, 0, i, 3:7] * dims
                                (startX, startY, endX, endY) = box.astype("int")

                                # draw the prediction on the frame
                                if idx > len(CLASSES)-1:continue
                                key = CLASSES[idx]
                                #if not key in IMAGES: continue
                                #use 20 pixels from the top for labeling
                                crop_img_data = frame[startY-20:endY, startX:endX]
                                #label = "Unknown"
                                hash=0
                                try:
                                    crop_img = Image.fromarray(crop_img_data)
                                    #crop_img = cv2.cvtColor( crop_img, cv2.COLOR_RGB2GRAY )
                                    #crop_img = ImageEnhance.Contrast(crop_img)
                                    hash = dhash.dhash_int(crop_img)
                                except: continue # pass

                                logger.debug("cam:" + str(cam)+ ", key:" + str(key) + " ,hash:" + str(hash))
                                logger.debug(hashes)
                                if not key in LOOKED1: continue
                                diffr = 1
                                if (hashes).get(key, None)== None:
                                    # count objects for last sec, last 5 sec and last minute
                                    logger.debug('ImageHashCodesCountByTimer init by hash: {}'.format(hash))
                                    hashes[key] = ImageHashCodesCountByTimer(1,30, (3,10,30))
                                    if hashes[key].add(hash) == False:  continue
                                    #filename = str(cam)+'_' + key +'_'+ str(hash)+ '.jpg'
                                else:
                                     #if not is_hash_the_same(hash,hashes[key]): hashes[key].add(hash)
                                     if hashes[key].add(hash) == False:  continue
                                     label2 =''
                                     for key in hashes:
                                        if hashes[key].getCountedObjects() == 0: continue
                                        label2 += ' ' + key+':' + str(hashes[key].getCountedObjects())
                                        #label2 += key+'(s):' + str(hashes[key].counted[0]) + ',' + str(hashes[key].counted[1]) + ',' + str(hashes[key].counted[2]) + ' '


                                label1 = "{}: {:.2f}%".format(key,confidence * 100)
                                #logger.debug('------------------- Rectangle placed in buffer ------------------------')
                                rectanglesQueue.put((label1, (startX-25, startY-25), (endX+25, endY+25),label2))
                                logger.debug((label1, (startX-25, startY-25), (endX+25, endY+25)))

                                # process further only  if image is really different from other ones
                                if key in subject_of_interes:
                                    #use it if you 100% sure you need save this image on disk
                                    #filename = str(cam)+'_' + key +'_'+ str(hash)+ '.jpg'
                                    x_dim = endX-startX
                                    y_dim = endY-startY
                                    fontScale = min(y_dim, x_dim)/280
                                    if fontScale > 4: cv2.putText(crop_img_data,str(datetime.datetime.now().strftime('%H:%M %d/%m/%y')),(1,15),cv2.FONT_HERSHEY_SIMPLEX,fontScale,(0,255,0),1)
                                    now = datetime.datetime.now()
                                    day = "{date:%Y-%m-%d}".format(date=now)
                                    db.insert_frame(conn, hash, day, time.time(), key, crop_img_data, x_dim, y_dim, cam)
                                    #cv2.imwrite(IMAGES_FOLDER + filename,crop_img_data)
                                    #logger.info("Persisting ,filename: " + IMAGES_FOLDER + filename)
                                do_statistic(conn, cam, hashes)
#  draw metadata on screen
def draw_metadata_onscreen(frame, rectanglesQueue,label2):
    #logger.debug('!!!!!!!!!!!!!!!! Display rectangles!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    while not rectanglesQueue.empty():
        try:
            (label1,dot1,dot2,label2) = rectanglesQueue.get(block=False)
            if DRAW_RECTANGLES: 
               cv2.rectangle(frame, dot1, dot2, (0,255,0), 1)
               cv2.putText(frame, label1, (dot1[0], dot1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        except: continue
    return label2
    
 
def change_res(camera, width, height):
    camera.set(3, width)
    camera.set(4, height)


def get_frame(video_urls,inputQueue, imagesQueue, rectanglesQueue, cam):
    # loop over the frames from the video stream
    detections = None
    label2 = 'No data'
    if 'picam' == video_urls[1]:
        video_s = VideoStream(usePiCamera=True,resolution=piCameraResolution,framerate=piCameraRate).start()
        time.sleep(2.0)
    else:
        # grab the frame from the threaded video stream, resize it, and
        # grab its dimensions
        video_s = cv2.VideoCapture(video_urls[1])



    while  True:
        flag,frame = video_s.read()
        if not flag:
            video_s = cv2.VideoCapture(video_urls[1])
            continue

        inputQueue.put(frame)

        if imagesQueue.qsize()>IMAGES_BUFFER-20:
           imagesQueue.get()
        if inputQueue.qsize()>IMAGES_BUFFER-20:
           inputQueue.get()

        label2 = draw_metadata_onscreen(frame, rectanglesQueue, label2)
        cv2.putText(frame, label2, (10,23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        # if perfomance issue on Raspberry Pi comment it
        if not "not_show_in_window" in args.keys():
            cv2.imshow("Camera" + str(cam), frame)
            key=cv2.waitKey(1) & 0xFF

        imagesQueue.put(frame)

    if (__name__ == '__main__'):
    # stop the timer and display FPS information
        fps.stop()
        logger.debug("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        logger.debug("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def fetchImagesFromQueueToVideo(filename, imagesQueue):
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
    #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    # fourcc = 0x00000021
    #logger.debug(fourcc)
    #out = cv2.VideoWriter(filename,fourcc, 29.0, size, False)  # 'False' for 1-ch instead of 3-ch for color
    #logger.debug(out)
    #fgbg= cv2.createBackgroundSubtractorMOG2()
    #logger.debug(fgbd)
    while(imagesQueue.qsize() > 2):
    #    fgmask = imagesQueue.get() #fgbg.apply(imagesQueue.get())
         imagesQueue.get()
         #np.save(filename,imagesQueue.get())
    #    out.write(fgmask)
        #cv2.imshow('img',fgmask)
    #out.release()


def destroy():
# stop the timer and display FPS information
    fps.stop()
    logger.debug("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    logger.debug("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    #conn.close()

separator = "="
args = {}

# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue =  []
rectanglesQueue = []
imagesQueue = []

#catchedObjQueue = Queue()

detections = None
vs = None

fps = None
p_get_frame = None


def configure(args):
    # construct the argument parse and parse the arguments
# I named config file as  file config.txt and stored it 
# in the same directory as the script

    with open('config.txt') as f:
        for line in f:
            if separator in line:
                # Find the name and value by splitting the string
                name, value = line.split(separator, 1)
                # Assign key value pair to dict
                # strip() removes white space from the ends of strings
                args[name.strip()] = value.strip()

    #global scrn_stats
    #scrn_stats = Screen_statistic(paramsQueue)

    ap = argparse.ArgumentParser()
    ap.add_argument("-nw","--not_show_in_window",required=False,
            help="video could be shown in window.", action="store_true", default=False)
    ap.add_argument("-v","--video_file", required=False,
            help="video file , could be access to remote location." )
    ap.add_argument("-p", "--prototxt", required=False,
            help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=False,
            help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, required=False,
            help="minimum probability to filter weak detections")
    more_args = vars(ap.parse_args())

    more_args =  {k: v for k, v in more_args.items() if v is not None}
    #if more_args["confidence"] == 0.0:more_args["confidence"] = args["confidence"]

    args.update(more_args)

    SHOW_VIDEO = not args["not_show_in_window"]
    logger.info("SHOW_VIDEO="+str(SHOW_VIDEO))

    logger.debug(args)



def startOneStreamProcesses( confidence, prototxt, model, cam):
        p_get_frame = Process(target=get_frame, args=(videos[cam],inputQueue[cam],imagesQueue[cam],rectanglesQueue[cam],cam))
        p_get_frame.daemon = True
        p_get_frame.start()

        p_classifier = Process(target=classify_frame, args=(inputQueue[cam],rectanglesQueue[cam], float(confidence), prototxt, model, cam))
        p_classifier.daemon = True
        p_classifier.start()


def start():
    # load our serialized model from disk
    configure(args)
    logger.info("[INFO] loading model...")
    # construct a child process *indepedent* from our main process of
    # execution
    logger.info("[INFO] starting process...")
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    logger.info("[INFO] starting video stream...")

    initialize_video_streams()

    for cam in range(len(videos)):
        p_get_frame = Process(target=get_frame, args=(videos[cam],inputQueue[cam],imagesQueue[cam],rectanglesQueue[cam],cam))
        p_get_frame.daemon = True
        p_get_frame.start()
       #p_get_frame.join()
        p_classifier = Process(target=classify_frame, args=(inputQueue[cam],rectanglesQueue[cam], float(args["confidence"]), args["prototxt"], args["model"], cam))
        p_classifier.daemon = True
        p_classifier.start()
        time.sleep(2.0)
       #p_classifier.join()
        logger.info("p_classifiers for cam:" +str(cam)+ " started")
        cam += 1
    return p_get_frame

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
def initialize_video_streams(url=None):
    i = 0
    arg = None
    right = None
    left = None
    if url is not None:
        arg = url
        i = len(videos)
#  initialise picam or IPCam
    else:
        arg = args.get('video_file'+ str(i),None)
    while arg is not None:
        if not (i,arg) in videos:
            camright.append(args.get('cam_right'+ str(i),None))
            camleft.append(args.get('cam_left'+ str(i),None))
            CameraMove(camright[i],camleft[i])
            videos.append((str(i),arg))
            imagesQueue.append(Queue(maxsize=IMAGES_BUFFER+5))
            inputQueue.append(Queue(maxsize=IMAGES_BUFFER+5))
            rectanglesQueue.append(Queue(maxsize=IMAGES_BUFFER+5))
            i+=1
            arg = args.get('video_file'+ str(i),None)



    # Start process
    time.sleep(3.0)
    fps = FPS().start()
    


###################### Flask API #########################
app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/urls": {"origins": "http://localhost:5000"}})

#api = Api(app)
#api.decorators=[cors.crossdomain(origin='*')]


@app.route('/static/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static', 'js'),   filename)

@app.route('/')
def index():
    """Video streaming home page."""
    start = request.args.get('start')
    if start == None: start = 0
    start = int(start)
    video_urls = []
    img_paginator = IMG_PAGINATOR
    conn = db.create_connection(SQLITE_DB)
    images_filenames = []
    for i in range(len(videos)):
        video_urls.append((videos[i][0],'video_feed?cam='+str(videos[i][0])))
        images_filenames.append(db.select_last_frames(conn,i,IMG_PAGINATOR))
    logger.info("video_urls: {}".format(video_urls))
    """ Delete old frames wil be here """
    return render_template('index.html', **locals())

@app.route('/moreparams')
def moreparams():
    """ Read list of json files or return one specific  for specific time """
    hour_back1 = request.args.get('hour_back1', default=1, type = int)
    hour_back2 = request.args.get('hour_back2', default=0, type = int)
    cam = request.args.get('cam', default=0, type = int)
    if hour_back1 != '': hour_back1 = int(hour_back1)
    else: hour_back1 = 1 # default value: 60 min back
    now_in_seconds = time.time()
    if hour_back2 != '': now_in_seconds = now_in_seconds - int(hour_back2)*60*60
    print("cam: {}, hour_back:{}, now_in_seconds:{}".format(cam, hour_back1, now_in_seconds))

    params = gen_params(cam=cam, hours=hour_back1, currentime = now_in_seconds)
    return Response( params, mimetype='text/plain')

@app.route('/moreimgs')
def moreimgs():
    """ Read list of json files or return one specific  for specific time """
    direction = request.args.get('direction', default=1, type = int)
    start =     request.args.get('start', default=0, type = int)
    cam =       request.args.get('cam', default=0, type = int)

    if direction < 0 : 
        if start >= IMG_PAGINATOR: start = start - IMG_PAGINATOR
        else: start = 0
    conn = db.create_connection(SQLITE_DB)
    rows =  db.select_last_frames(conn, cam, IMG_PAGINATOR, offset = start, as_json=True)
#    ret = json.dumps(rows)
    return Response( rows,  mimetype='text/plain')


@app.route('/imgs_at_time')
def imgs_at_time():
    """ Read list of json files or return one specific  for specific time """
    seconds = request.args.get('time', default=time.time(), type=int)
    delta   = request.args.get('delta', default=10, type=int)
    cam = request.args.get('cam', default = 0, type = int)
    return Response( gen_array_of_imgs(cam, delta=delta, currentime=seconds), mimetype='text/plain')

def gen_array_of_imgs(cam,delta=10, currentime = time.time()):
    time1 = currentime - delta
    time2 = currentime + delta
    conn = db.create_connection(SQLITE_DB)
    rows = db.select_frame_by_time(conn, cam, time1,time2)
    x = json.dumps(rows)
    return x




def gen(camera):
    """Video streaming generator function."""
    try:
        while True:
          frame = camera.get_frame()
          yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except GeneratorExit: pass




def detect(cam):
    """Video streaming generator function."""
    try:
        #logger.debug('imagesQueue:', imagesQueue.empty())
       while True:
         while(not imagesQueue[cam].empty()):
            frame = imagesQueue[cam].get(block=True)
            # draw rectangles when run on good hardware
            iterable = cv2.imencode('.jpg', frame)[1].tobytes()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + iterable + b'\r\n'
    except GeneratorExit: pass

def gen_params(cam=0, hours=1, currentime=time.time()):
    """Parameters streaming generator function."""
    time1 = currentime - hours*60*60
    print("time1: {} now: {}".format(time1, currentime))
    conn = db.create_connection(SQLITE_DB)
    ls = db.select_statistic_by_time(conn,cam,time1,currentime)
    ret = json.dumps(ls) #, indent = 4)
    logger.debug(ret)
    return ret


def ping_video_url(url):
    """ Ping url """
    try:
        vs = cv2.VideoCapture(url)
        flag,frame = vs.read()
        ret = flag
    except:
        ret = False
    return flag


@app.route('/urls',methods=['GET','POST'])
@cross_origin(origin='http://localhost:5000')
def urls():
    """Add/Delete/Update a new video url, list all availabe urls."""
    list_url   = request.args.get('list', default=None)
    add_url    = request.args.get('add', default=None)
    delete_url = request.args.get('delete', default=None)
    update_url = request.args.get('update', default=None)
    if add_url is not None:
        if ping_video_url(add_url):
            initialize_video_streams(add_url)
            startOneStreamProcesses( args["confidence"], args["prototxt"], args["model"], cam=len(videos)-1)
            #return index() #redirect("/")
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
            return Response( '{"message":"URL updated successfully"}', mimetype='text/plain')


@app.route('/video_feed',methods=['GET'])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    # gen(Camera()),
    cam = request.args.get('cam', default = 0, type = int)
    return Response( detect(int(cam)), #mimetype='text/event-stream')
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/params_feed')
def params_feed():
    """Parameters streaming route. Put this in the src attribute of an img tag."""
    hours = request.args.get('hour_back1', default=1)
    start_hour = request.args.get('hour_back2', default=0)
    currentime = time.time() - int(start_hour)*60*60
    return Response( gen_params(hours, currentime=currentime ),
                    mimetype='text/plain')
#@app.route('/images_feed')
#def images_feed():
#    """Images streaming route. Put this in the src attribute of an img tag."""
#    return Response( gen_images(),
#                    mimetype='text/plain')

if (__name__ == '__main__'):
    start()
    app.run(host='0.0.0.0',threaded=True) # debug = True ) # 
