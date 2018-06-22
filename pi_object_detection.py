

# USAGE
# python pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import imagehash
import numpy as np
import argparse
import imutils
import time
import dhash
from PIL import Image
from time import gmtime, strftime

import cv2
import json
from screen_statistics import Screen_statistic
from flask import Flask, render_template, Response, request
import base64
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]#,"_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_","_X_"]
LOOKED = { "car": [], "cat": [],"dog": [], "person": []}
subject_of_interes = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

VIDEO_FILENAME = 'images/camera'
NUMBER_OF_FILES = 100
HASH_DELTA = 55
PARAMS_BUFFER =  8
IMAGES_BUFFER = 100
RECOGNZED_FRAME = 1
THREAD_NUMBERS  = 3 # must be less then 4 for PI
videos = []

def classify_frame( net, inputQueue, outputQueue):
        # keep looping
        while True:
                # check to see if there is a frame in our input queue
                #while not inputQueue.empty():
                # grab the frame from the input queue, resize it, and
                # construct a blob from it
                print('inputQueue.qsize()',inputQueue.qsize())
                print('outputQueue.qsize()',outputQueue.qsize())
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
                outputQueue.put(detections)#(detections,rows,cols))



def get_frame(vss):
    # loop over the frames from the video stream
    detections = None
    cols,rows = 0,0
    j = PARAMS_BUFFER+1
    k = 0
    _thr = RECOGNZED_FRAME
    hashes = []
    for cam in range(len(vss)):
        hashes.append(LOOKED)
    while  True:
      print(j)
      for cam in range(len(vss)):
            
	    # grab the frame from the threaded video stream, resize it, and
            # grab its imensions
            flag,frame = vss[cam].read()
            
            print('cam:', cam, 'flag: ' , flag , "vs:",vs)
            if not flag:
                vss[cam] = cv2.VideoCapture(videos[cam])
                continue
            #frame = imutils.resize(frame, width=640)
            inputQueue[cam].put(frame) 
            (fH, fW) = frame.shape[:2]
            # if the output queue *is not* empty, grab the detections
            detections = outputQueue[cam].get()
            #print(detections)
    
            if detections is not None:

                    # loop over the detections
                    for i in np.arange(0, detections.shape[2]):
                            # extract the confidence (i.e., probability) associated
                            # with the prediction
                            confidence = detections[0, 0, i, 2]

                            # filter out weak detections by ensuring the `confidence`
                            # is greater than the minimum confidence
                            if confidence < args["confidence"]:continue

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
                                hash = dhash.dhash_int(crop_img)
                            except: None
                                #continue
                                
                            key = CLASSES[idx]
                            
                            print("cam:", cam, "key:",key,"hash:",hash)
                            if not key in LOOKED: continue
                            if (hashes[cam]).get(key, None)== None:
                                hashes[cam][key] = [hash]
                                continue
                            _hashes = []
                            diffr = 0
                            for _hash in hashes[cam][key]:
                                delta = dhash.get_num_bits_different(_hash, hash)
                                #print("delta: ", delta)
                                if delta < HASH_DELTA: break
                                else: diffr +=1
                            if len(hashes[cam][key]) == diffr and hash !=0:
                                hashes[cam][key].append(hash)
                            print("cam:", cam, "key:", key, "hashes:", hashes[cam][key])
                            label = "{}: {:.2f}%".format(key,confidence * 100)
                            if key in subject_of_interes:
                                #crop_img = frame[startY:endY, startX:endX]
                                cv2.imwrite('images/'+str(hash)+'.jpg',frame)

                            cv2.rectangle(frame, (startX-10, startY-10), (endX+10, endY+10),
                                    COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                                                       

            
            imagesQueue[cam].put(frame)

                           
            params = scrn_stats.refresh(hashes[cam], cam)
                   
            print(params)
            #if paramsQueue.qsize()> PARAMS_BUFFER: continue
            paramsQueue.put( params )
            print('paramsQueue.qsize()',paramsQueue.qsize())  
            if imagesQueue[cam].qsize() > IMAGES_BUFFER:
                k+=1
                fetchImagesFromQueueToVideo(VIDEO_FILENAME+str(cam)+'_'+str(k), imagesQueue[cam],(640,480))
                k %= NUMBER_OF_FILES
            if paramsQueue.qsize() > IMAGES_BUFFER:  
                fetchParamsFromQueueToDB("dbname", paramsQueue)

      j+=1
      if j >= PARAMS_BUFFER:
         j = 0
         for cam in range(len(vss)): hashes[cam] = {}
      

    if (__name__ == '__main__'):
    # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def fetchImagesFromQueueToVideo(filename, imagesQueue, size):
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
    #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    # fourcc = 0x00000021  
    #print(fourcc)
    #out = cv2.VideoWriter(filename,fourcc, 29.0, size, False)  # 'False' for 1-ch instead of 3-ch for color
    #print(out)
    #fgbg= cv2.createBackgroundSubtractorMOG2()
    #print(fgbd)
    while(imagesQueue.qsize() > 2):
    #    fgmask = imagesQueue.get() #fgbg.apply(imagesQueue.get())
         np.save(filename,imagesQueue.get())
    #    out.write(fgmask)
        #cv2.imshow('img',fgmask)   
    #out.release()
    
    
def fetchParamsFromQueueToDB(db, paramsQueue):
    _array = []
    while(paramsQueue.qsize() > 0):
        _array.append(paramsQueue.get())
    # connect to DB and store array of parameters here


def destroy():
# stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

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
paramsQueue = Queue()


detections = None
vs = None
vss = []
fps = None
p_get_frame = None

if (__name__ == '__main__'):
    # construct the argument parse and parse the arguments
# I named config file as  file config.txt and stored it 
# in the same directory as the script
    args = {}
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
    ap.add_argument("-c", "--confidence", type=float, default=0.50,
            help="minimum probability to filter weak detections")
    more_args = vars(ap.parse_args())

    more_args =  {k: v for k, v in more_args.items() if v is not None}
    #if more_args["confidence"] == 0.0:more_args["confidence"] = args["confidence"]

    args.update(more_args)

    print(args)





def start():
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # construct a child process *indepedent* from our main process of
    # execution
    print("[INFO] starting process...")
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    if args['video_file'] != '':
       p_get_frame = initialize_video_streams()
    cam = 0
    for vs in vss:        
        for i in range(THREAD_NUMBERS):
            p_classifier = Process(target=classify_frame, args=(net,inputQueue[cam],
                    outputQueue[cam],))
            p_classifier.daemon = False
            p_classifier.start()
        
        print("p_classifiers for cam:",cam, " started")
        cam += 1
    return p_get_frame

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
def initialize_video_streams():
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args['video_file'])
    print("[INFO] Video stream1: ", vs, args['video_file'])
    vss.append(vs)
    videos.append(args['video_file'])
    imagesQueue.append(Queue())
    if args.get('video_file2',None) != None:
        vs = cv2.VideoCapture(args['video_file2'])
        print("[INFO] Video stream2: ", vs, args['video_file2'])
        vss.append(vs)
        videos.append(args['video_file2'])
        imagesQueue.append(Queue())

    time.sleep(3.0)
    fps = FPS().start()
    if(vs is None):
        print("[INFO] starting video stream(s) failed.")
    else:
        print("[INFO] video stream started.")
        
    for vs in vss:
        inputQueue.append(Queue())
        outputQueue.append(Queue())
        
    # show the output frame when need to test is working or not
    p_get_frame = Process(target=get_frame, args=(vss,))
    p_get_frame.daemon = False
    p_get_frame.start()
    return p_get_frame

###################### Flask API #########################
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    video_url1 = args['video_file']
    if args.get('video_file2',None) != None:
        video_url2 = args['video_file2']             
        
    return render_template('index.html', **locals())

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect(cam):
    """Video streaming generator function."""
    while True:
         #print('imagesQueue:', imagesQueue.empty())
         while(not imagesQueue[cam].empty()):
             iterable = imagesQueue[cam].get()
             iterable = cv2.imencode('.jpg', iterable)[1].tobytes()
             yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + iterable + b'\r\n'



def gen_params():
    """Parameters streaming generator function."""
    x = []
    while not paramsQueue.empty():
        x += paramsQueue.get()
    x = json.dumps(x) 
    #print(x)
    return x

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

if (__name__ == '__main__'):
    start()
    app.run(host='0.0.0.0', threaded=True)


