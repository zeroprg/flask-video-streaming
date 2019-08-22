# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue
from files import *

import os
import io
import imagehash
import numpy as np
import argparse
import imutils
import time
import dhash
import glob
import logging 

from PIL import Image, ImageEnhance
from time import gmtime, strftime
import datetime
import cv2
import json
from screen_statistics import Screen_statistic
from objCountByTimer import ObjCountByTimer
import base64

from flask import Flask, render_template, Response, request,redirect,jsonify
from flask_cors  import cross_origin, CORS
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera

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
LOOKED1 = { "car": [], "cat": [],"dog": [],"person":  []}
LOOKED2 = { "car": [], "cat": [],"dog": [], "person": []}

subject_of_interes = ["person","cat","dog"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

IMAGES_FOLDER = "static/img/"
PARAMS_FOLDER = "static/params/"

DRAW_RECTANGLES = True
DELETE_FILES_LATER = 24 * 60 * 60 # sec 
ENCODING = "utf-8"
NUMBER_OF_FILES = 10
HASH_DELTA = 45# 72
PARAMS_BUFFER = 25
IMAGES_BUFFER = 25
RECOGNZED_FRAME = 1
THREAD_NUMBERS  = 1 #must be less then 4 for PI
videos = []
piCameraResolution = (640,480) #(1296,972)
piCameraRate=24
IMG_PAGINATOR = 50

class ImageHashCodesCountByTimer(ObjCountByTimer):
    def equals(self,hash1, hash2):
        delta = dhash.get_num_bits_different(hash1, hash2)
        return delta < HASH_DELTA

def is_hash_the_same(hash,hashes):
    diffr = 0
    for _hash in hashes:
       delta = dhash.get_num_bits_different(_hash, hash)
       logger.debug("delta: " + str(delta))
       if delta < HASH_DELTA: break
       else: diffr +=1
    return len(hashes) != diffr or hash!=0


def clean_params(cam, hashes, inputQueue, filenames):
  #  delta = (time.time() - starttime) % 60.0
  #  logger.debug(delta)
  #  if delta > 9.0 :
    logger.debug("!!!!!!!!!!!!!!!!! Params initialized !!!!!!!!!!!!!!!")
    hashes = {"car":[], "person":[], "cat":[], "dog":[]}
    filenames  = {}
    logger.debug(hashes)

def do_statistic(cam,hashes, rectanglesQueue, inputQueue, filenames):
  # Do some statistic work here
    logger.debug('get_frame: hashes[' + str(cam) +']')
    logger.debug(hashes)
    logger.debug('get_frame: filenames[' + str(cam) +']')
    logger.debug(filenames)

    params = scrn_stats.refresh(hashes,filenames, cam)
    if len(params)>0:
        #params(params)
        #if paramsQueue.qsize()> PARAMS_BUFFER: continue
        logger.debug('get_frame: params' )
        logger.debug(params)
        paramsQueue.put( params )
        logger.info('inputQueue.qsize():' + str(inputQueue.qsize()) + ' paramsQueue.qsize():' + str(paramsQueue.qsize()) + ' imagesQueue[' + str(cam)+'].qsize():' + str(imagesQueue[cam].qsize()) + ' rectanglesQueue[' + str(cam)+'].qsize():' + str(rectanglesQueue.qsize()) )
    if paramsQueue.qsize() > PARAMS_BUFFER:
        persist_params(PARAMS_FOLDER + str(int(time.time())) + '.json')


def classify_frame( net, inputQueue,rectanglesQueue,cam):
        conf_threshold = float(args["confidence"])
        hashes = {}
        filenames = {}

        #starttime=time.time()
        # keep looping
        frame = None
        while True:
                #clean_params(cam,starttime)
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
                logger.debug('cam:' + str(cam))
                _frame = cv2.resize(frame, (300, 300))
                cols = frame.shape[1]
                rows = frame.shape[0]
                blob = cv2.dnn.blobFromImage(_frame, 0.007843,
                        (300, 300), (127.5,127.5,127.5), False)
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
                                confidence = detections[0, 0, i, 2]
                                # filter out weak detections by ensuring the `confidence`
                                # is greater than the minimum confidence
                                if confidence < conf_threshold: continue

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
                                crop_img_data = frame[startY:endY, startX:endX]
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
                                    print('ImageHashCodesCountByTimer init by hash: {}'.format(hash))
                                    hashes[key] = ImageHashCodesCountByTimer(1,60, (5,10,60))
                                    hashes[key].add(hash)
                                    filename = str(cam)+'_' + key +'_'+ str(hash)+ '.jpg'
                                    filenames[key] = [filename]
                                else:
                                     #if not is_hash_the_same(hash,hashes[key]): hashes[key].add(hash)
                                     print('hash: {}'.format(hash))
                                     hashes[key].add(hash)



                                if DRAW_RECTANGLES: 
                                    label = "{}: {:.2f}%".format(key,confidence * 100)
                                    #cv2.rectangle(frame, (startX-25, startY-25), (endX+25, endY+25), (0,255,0), 2)
                                    #y = startY - 25 if startY - 25 > 25 else startY + 25
                                    #cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                                    # add frames with rectangles
                                    #logger.debug('------------------- Rectangle placed in buffer ------------------------')
                                    rectanglesQueue.put((label, (startX-25, startY-25), (endX+25, endY+25)))
                                    logger.debug((label, (startX-25, startY-25), (endX+25, endY+25)))
                                    #do_statistic(cam, hashes, rectanglesQueue, inputQueue, filenames)

                                # process further only  if image is really different from other ones
                                if key in subject_of_interes:
                                    #use it if you 100% sure you need save this image on disk
                                    filename = str(cam)+'_' + key +'_'+ str(hash)+ '.jpg'
                                    filenames[key].append(filename)
                                    fontScale = min(endY-startY, endX-startX)/280
                                    cv2.putText(crop_img_data,str(datetime.datetime.now().strftime('%H:%M %d/%m/%y')),(1,15),cv2.FONT_HERSHEY_SIMPLEX,fontScale,(0,255,0),1)
                                    cv2.imwrite(IMAGES_FOLDER + filename,crop_img_data)
                                    logger.info("Persisting ,filename: " + IMAGES_FOLDER + filename)

                                    #encoded =  (base64.b64encode(imgb)).decode(ENCODING)
                                    #catchedObjQueue.put( str(cam) + ";" + key + ";" +encoded)
                                    logger.debug("classify_frame: cam:" + str(cam) + "key:" + key+ ", filenames: ")
                                    logger.debug(filenames[key])
                                    #continue
                                   



def draw_metadata_onscreen(frame, rectanglesQueue):
           logger.debug('!!!!!!!!!!!!!!!! Display rectangles!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
           try:
                (label,dot1,dot2) = rectanglesQueue.get(block=False)
                cv2.rectangle(frame, dot1, dot2, (0,255,0), 1)
                cv2.putText(frame, label, (dot1[0], dot1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,128), 1)
           except: pass     



def get_frame(vss,video_urls,inputQueue, imagesQueue, rectanglesQueue, cam):
    # loop over the frames from the video stream

    detections = None
    cols,rows = 0,0
    while  True:
        logger.debug('cam:' +video_urls[1])
        if 'picam' == video_urls[1]:
            if vss == None:
                vss = VideoStream(usePiCamera=True,resolution=piCameraResolution,framerate=piCameraRate).start()
                time.sleep(2.0)
            frame = vss.read()
        else:
    # grab the frame from the threaded video stream, resize it, and
    # grab its dimensions
            flag,frame = vss.read()
            if not flag:
               vss = cv2.VideoCapture(video_urls[1])
               continue

        inputQueue.put(frame)

        if imagesQueue.qsize()>IMAGES_BUFFER-1:
           while not imagesQueue.empty(): imagesQueue.get()  
        if inputQueue.qsize()>IMAGES_BUFFER-1:
           while not inputQueue.empty(): inputQueue.get()  


        draw_metadata_onscreen(frame,rectanglesQueue)

        # if perfomance issue on Raspberry Pi comment it
        if not args["not_show_in_window"]:
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
    
def dumpQueue(paramsQueue):
    params_array = []
    while not paramsQueue.empty():
        a = paramsQueue.get()
        params_array +=a
    x = json.dumps(params_array)
    return x
    
def persist_params(filename):
    logger.info("Persisting ,filename: " + filename)
    x = dumpQueue(paramsQueue)
    f = open(filename,"w+")    

    f.write(x)
    f.close()

    return x

def destroy():
# stop the timer and display FPS information
    fps.stop()
    logger.debug("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    logger.debug("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


separator = "="
args = {}

# initialize the input queue (frames), output queue (detections),
# and the list of actual detections returned by the child process
inputQueue =  []
rectanglesQueue = []
imagesQueue = []
paramsQueue = Queue(maxsize=PARAMS_BUFFER+5)

#catchedObjQueue = Queue()

detections = None
vs = None
vss = []
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

    global scrn_stats
    scrn_stats = Screen_statistic(paramsQueue)

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
    print("SHOW_VIDEO="+str(SHOW_VIDEO))

    logger.debug(args)


def start():
    # load our serialized model from disk
    configure(args)
    logger.info("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # construct a child process *indepedent* from our main process of
    # execution
    logger.info("[INFO] starting process...")
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    logger.info("[INFO] starting video stream...")


       
    initialize_video_streams()
        
        

    for cam in range(len(vss)):
        p_get_frame = Process(target=get_frame, args=(vss[cam],videos[cam],inputQueue[cam],imagesQueue[cam],rectanglesQueue[cam],cam))
        p_get_frame.daemon = True
        p_get_frame.start()
       #p_get_frame.join()
    
    #for i in range(0,len(videos)-1):
        # Share common parameters between threads
        #while(inputQueue[i].empty()): None
        
 #       print('inputQueue[i].qsize():' + str(i) + ": " + str(inputQueue[i].qsize()))

    time.sleep(2.0)
    for cam in range(len(vss)):
        for i in range(THREAD_NUMBERS):
            # Share common parameters between threads
            p_classifier = Process(target=classify_frame, args=(net,inputQueue[cam],rectanglesQueue[cam],cam))
            p_classifier.daemon = True
            p_classifier.start()
            #p_classifier.join()
            
        
        logger.info("p_classifiers for cam:" +str(cam)+ " started")
        cam += 1
    return p_get_frame

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
def initialize_video_streams(url=None):
    i = 0
    arg = None
    if url is not None:
        arg = url
        i = len(videos)
#  initialise picam or IPCam
    else:
        arg = args.get('video_file'+ str(i),None)
    while arg is not None:
        if not (i,arg) in videos:
            if arg  == 'picam':
                vs = None
            else: 
                 vs = cv2.VideoCapture(arg)
            logger.info("[INFO] Video stream: " + str(i) + " vs:" + str(vs) )
            vss.append(vs)
            videos.append((str(i),arg))
            imagesQueue.append(Queue(maxsize=IMAGES_BUFFER+5))
            inputQueue.append(Queue(maxsize=PARAMS_BUFFER+5))
            rectanglesQueue.append(Queue(maxsize=PARAMS_BUFFER+5))
            i+=1
            arg = args.get('video_file'+ str(i),None)


    # Start process
    time.sleep(2.0)
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
    print("start", start)
    if start == None: start = 0
    start = int(start)
    video_urls = []
    img_folder = IMAGES_FOLDER
    img_paginator = IMG_PAGINATOR
    for i in range(len(videos)):
        if DRAW_RECTANGLES: video_urls.append((videos[i][0],'video_feed?cam='+str(videos[i][0])))
        else:
            video_urls.append((videos[i][0], videos[i][1]))
    """ Delete old files """
    images_filenames=[]
    delete_file_older_then(IMAGES_FOLDER, DELETE_FILES_LATER)
    delete_file_older_then(PARAMS_FOLDER, DELETE_FILES_LATER)
    images_filenames = traverse_dir( IMAGES_FOLDER, True, str(0)+"_*", start, start + IMG_PAGINATOR)
    images_filenames.extend( traverse_dir(IMAGES_FOLDER, True, str(1)+"_*", start, start + IMG_PAGINATOR ))
    
    return render_template('index.html', **locals())

@app.route('/moreparams')
def moreparams():
    """ Read list of json files or return one specific  for specific time """
    time = request.args.get('time')
    if time == '' or time is None: time = 0
    else: time = int(time)
    files = gen_array_of_params()
    from_indx = find_index(files,time)
    to_indx = len(files)
    print("from_indx:", from_indx)
    _arr = ''
    for indx_ in range(from_indx, to_indx-1):
        _arr +=  open( files[indx_], 'r').read()
    
    return Response( _arr.replace('][',','), mimetype='text/plain')
        
def gen_array_of_params():
    params_filenames =  traverse_dir(PARAMS_FOLDER, False)
    #x = json.dumps(params_filenames)
    return params_filenames

        

@app.route('/moreimgs')
def moreimgs():
    start = int(request.args.get('start'))
    cam = int(request.args.get('cam'))
    direction = int(request.args.get('direction'))
    return Response( gen_array_of_imgs(cam,start,direction), mimetype='text/plain')

def gen_array_of_imgs(cam,start,direction):
    a = start
    b = start+direction*IMG_PAGINATOR
    if start > start+direction*IMG_PAGINATOR:
        b = start
        a = start+direction*IMG_PAGINATOR

    images_filenames =  traverse_dir(IMAGES_FOLDER, True, str(cam)+"_*", a,b)
    x = json.dumps(images_filenames)
    return x




def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect(cam):
    """Video streaming generator function."""
    while True:
         #logger.debug('imagesQueue:', imagesQueue.empty())
         while(not imagesQueue[cam].empty()):
             try:
                frame = imagesQueue[cam].get(block=False)
                iterable = cv2.imencode('.jpg', frame)[1].tobytes()
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + iterable + b'\r\n'
             except: continue


def gen_params():
    """Parameters streaming generator function."""
    if paramsQueue.qsize() > PARAMS_BUFFER:
        ret = persist_params(PARAMS_FOLDER + str(int(time.time())) + '.json')
    else:
        ret = dumpQueue(paramsQueue)    
    logger.debug(ret)
    return ret
#def gen_images():
#   """Images streaming generator function."""
#   return catchedObjQueue.get()

def ping_video_url(url):
    """ Ping url """
    try:
        vs = cv2.VideoCapture(url)
        flag,frame = vs.read()
        ret = flag
    except:
        ret = False
    return flag


@app.route('/urls',methods=['GET'])
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
            #return redirect("/", code=303)
            return Response('{"message":"URL added  successfully"}', mimetype='text/plain')
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
    return Response( gen_params(),
                    mimetype='text/plain')
#@app.route('/images_feed')
#def images_feed():
#    """Images streaming route. Put this in the src attribute of an img tag."""
#    return Response( gen_images(),
#                    mimetype='text/plain')

if (__name__ == '__main__'):
    start()
    app.run(host='0.0.0.0',threaded=True) # debug = True ) # 
