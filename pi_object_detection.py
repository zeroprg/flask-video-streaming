# USAGE
# python pi_object_detection.py 
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
from files import *

import os
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

import cv2
import json
from screen_statistics import Screen_statistic
import base64

from flask import Flask, render_template, Response, request,redirect,jsonify
from flask_cors  import cross_origin, CORS

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)
logger.info('test')


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
LOOKED1 = { "car": [], "cat": [],"dog": [], "person": [], "pottedplant":[], "bottle":[], "chair":[]}
LOOKED2 = { "car": [], "cat": [],"dog": [], "person": [], "pottedplant":[], "bottle":[], "chair":[]}

subject_of_interes = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

IMAGES_FOLDER = "static/img/"
PARAMS_FOLDER = "static/params/"

DRAW_RECTANGLES = False
DELETE_FILES_LATER = 24 * 60 * 60 # sec 
ENCODING = "utf-8"
NUMBER_OF_FILES = 10
HASH_DELTA = 57
PARAMS_BUFFER =  10
IMAGES_BUFFER = 200
RECOGNZED_FRAME = 1
THREAD_NUMBERS  = 1 #must be less then 4 for PI
videos = []


IMG_PAGINATOR = 50

def classify_frame( net, inputQueue, outputQueue):
        # keep looping
        while True:
                # check to see if there is a frame in our input queue
                #while not inputQueue.empty():
                # grab the frame from the input queue, resize it, and
                # construct a blob from it
                #logger.debug('inputQueue.qsize()',inputQueue.qsize())
                #logger.debug('outputQueue.qsize()',outputQueue.qsize())
                frame = inputQueue.get()
                frame = cv2.resize(frame, (300, 300))
                cols = frame.shape[1]
                rows = frame.shape[0]
                blob = cv2.dnn.blobFromImage(frame, 0.007843,
                        (300, 300), (127.5,127.5,127.5), False)
                # set the blob as input to our deep learning object
                # detector and obtain the detections
                net.setInput(blob)
                detections = net.forward()

                # write the detections to the output queue
                outputQueue.put(detections)



def get_frame(vss,video_urls):
    # loop over the frames from the video stream
    conf_threshold = float(args["confidence"])
    detections = None
    cols,rows = 0,0
    j = PARAMS_BUFFER+1
    k = 0
    _thr = RECOGNZED_FRAME
    hashes = []
    filenames = []

    for cam in range(len(vss)):
        hashes.append(LOOKED1)
        filenames.append(LOOKED2)
    while  True:
      logger.debug(str(j)+ " len(vss): "+ str(len(vss)) )
      for cam in range(len(vss)):
	        # grab the frame from the threaded video stream, resize it, and
            # grab its imensions
            flag,frame = vss[cam].read()
            
            if not flag:
                vss[cam] = cv2.VideoCapture(video_urls[cam][1])
                continue
            #frame = imutils.resize(frame, width=640)
            inputQueue[cam].put(frame) 
            (fH, fW) = frame.shape[:2]
            # if the output queue *is not* empty, grab the detections
            detections = outputQueue[cam].get()
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
                            except: None
                                #continue
                                
                            key = CLASSES[idx]
                            
                            logger.debug("cam:", cam, "key:",key,"hash:",hash)
                            if not key in LOOKED1: continue
                            if (hashes[cam]).get(key, None)== None:
                                hashes[cam][key] = [hash]
                                filename = str(cam)+'_' + key +'_'+ str(time.time()).replace(".","_")+ '.jpg'
                                filenames[cam][key] = [filename]
                                continue
                            #_hashes = []
                            diffr = 0
                            for _hash in hashes[cam][key]:
                                delta = dhash.get_num_bits_different(_hash, hash)
                                #logger.debug("delta: ", delta)
                                if delta < HASH_DELTA: break
                                else: diffr +=1
                            # process further only  if image is really different from other ones   
                            if len(hashes[cam][key]) == diffr and hash !=0:
                                hashes[cam][key].append(hash)
                                if key in subject_of_interes:
                                    #use it if you 100% sure you need save this image on disk
                                    #filename = str(cam)+'_' + key +'_'+ str(time.time()).replace(".","_")+ '.jpg'
                                    filename = str(cam)+'_' + key +'_'+ str(hash)+ '.jpg'
                                    filenames[cam][key].append(filename)
                                    cv2.imwrite(IMAGES_FOLDER + filename,crop_img_data)
                                    imgb = crop_img_data.tobytes()
                                    
                                    encoded =  (base64.b64encode(imgb)).decode(ENCODING)
                                    #catchedObjQueue.put( str(cam) + ";" + key + ";" +encoded)
                                    logger.debug("cam:", cam, "key:", key, "filenames:", filenames[cam][key])
                                    
                            logger.debug("cam:", cam, "key:", key, "hashes:", hashes[cam][key])
                           
                            
                            if DRAW_RECTANGLES: 
                                label = "{}: {:.2f}%".format(key,confidence * 100)
                                cv2.rectangle(frame, (startX-25, startY-25), (endX+25, endY+25), (0,255,0), 2)
                                y = startY - 25 if startY - 25 > 25 else startY + 25
                                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                                                       
            
            params = scrn_stats.refresh(hashes[cam],filenames[cam], cam)
            logger.debug(params)
            #if paramsQueue.qsize()> PARAMS_BUFFER: continue
            paramsQueue.put( params )
            logger.debug('paramsQueue.qsize()',paramsQueue.qsize())
            if DRAW_RECTANGLES:
                imagesQueue[cam].put(frame)
                if imagesQueue[cam].qsize() > IMAGES_BUFFER:
                    k+=1
                    fetchImagesFromQueueToVideo(IMAGES_FOLDER+str(cam)+'_'+str(k), imagesQueue[cam],(640,480))
                    k %= NUMBER_OF_FILES
            if paramsQueue.qsize() > IMAGES_BUFFER:
                persist_params(PARAMS_FOLDER + str(int(time.time())) + '.json')

      j+=1
      if j >= PARAMS_BUFFER:
         j = 0
         for cam in range(len(vss)):
            hashes[cam] = {}
            filenames[cam] = {}
      

    if (__name__ == '__main__'):
    # stop the timer and display FPS information
        fps.stop()
        logger.debug("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        logger.debug("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def fetchImagesFromQueueToVideo(filename, imagesQueue, size):
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
    #logger.info("Persisting ,x: " + x)
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
imagesQueue = []
outputQueue = []
paramsQueue = Queue(maxsize=IMAGES_BUFFER+5)

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
            
    # show the output frame when need to test is working or not
    p_get_frame = Process(target=get_frame, args=(vss,videos))
    p_get_frame.daemon = True
    p_get_frame.start()
    
    cam = 0
    for vs in vss:        
        for i in range(THREAD_NUMBERS):
            p_classifier = Process(target=classify_frame, args=(net,inputQueue[cam],
                    outputQueue[cam],))
            p_classifier.daemon = False
            p_classifier.start()
        
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
    else:
        arg = args.get('video_file'+ str(i),None)
    while arg is not None:
        if not (i,arg) in videos:
            logger.info("[INFO] starting video stream with arg: " + arg)
            vs = cv2.VideoCapture(arg)
            logger.info("[INFO] Video stream: " + str(i) + " vs:" + str(vs) )
            vss.append(vs)
            videos.append((str(i),arg))
            imagesQueue.append(Queue())
            inputQueue.append(Queue())
            outputQueue.append(Queue())
            i+=1
            arg = args.get('video_file'+ str(i),None)

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
             iterable = imagesQueue[cam].get()
             iterable = cv2.imencode('.jpg', iterable)[1].tobytes()
             yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + iterable + b'\r\n'



def gen_params():
    """Parameters streaming generator function."""
    if paramsQueue.qsize() > IMAGES_BUFFER:
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
