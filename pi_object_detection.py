
# USAGE
# python pi_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
from time import gmtime, strftime

import cv2
import json
from screen_statistics import Screen_statistic
from flask import Flask, render_template, Response
import base64
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","__"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


"""An emulated camera implementation that streams a repeated sequence of
files 1.jpg, 2.jpg and 3.jpg at a rate of one frame per second."""
imgs =  [open(f + '.jpg', 'rb').read() for f in ['1', '2', '3']]


pnt = 0
IMAGE_BUFFER =2000
PARAMS_BUFFER = 2000
RECOGNZED_FRAME =1
THREAD_NUMBERS = 3 # must be less then 4 for PI

def classify_frame( net, inputQueue, outputQueue):
        # keep looping
        while True:
                # check to see if there is a frame in our input queue
                if not inputQueue.empty():
                        # grab the frame from the input queue, resize it, and
                        # construct a blob from it
                        print('inputQueue.qsize()',inputQueue.qsize())
                        print('outputQueue.qsize()',outputQueue.qsize())
                        frame = inputQueue.get()
                        #inputQueue.task_done()
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



def get_frame(vs):
    # loop over the frames from the video stream 
    cols,rows = 0,0
    i = 0
    while  True:
            #print("imagesQueue.qsize()", imagesQueue.qsize())
            #dictionary of detected objects
            classes = {}
	    # grab the frame from the threaded video stream, resize it, and
            # grab its imensions
            flag,frame = vs.read()        
            #print('Test 2- flag: ' , flag )
            if not flag: continue
            #frame = imutils.resize(frame, width=640)
            (fH, fW) = frame.shape[:2]
            if i < RECOGNZED_FRAME: i+= 1
            else:
                i =0
                inputQueue.put(frame)

            # if the output queue *is not* empty, grab the detections
            detections = None
            if not outputQueue.empty():
                    detections = outputQueue.get()
                    #outputQueue.task_done() 
                    #detections = object_detected[0]
                    #cols = object_detected[1]
                    #rows = object_detected[2]

            # check to see if our detectios are not None (and if so, we'll
            # draw the detections on the frame)
            #print('Test3- detections: ' , detections )
            #print("Test 2.5")
            if detections is not None:
                    # loop over the detections
                    for i in np.arange(0, detections.shape[2]):
                            # extract the confidence (i.e., probability) associated
                            # with the prediction
                            confidence = detections[0, 0, i, 2]

                            # filter out weak detections by ensuring the `confidence`
                            # is greater than the minimum confidence
                            if confidence < args["confidence"]:
                                    continue

                            # otherwise, extract the index of the class label from
                            # the `detections`, then compute the (x, y)-coordinates
                            # of the bounding box for the object
                            idx = int(detections[0, 0, i, 1])


                            #xLeftBottom = int(detections[0, 0, i, 3] * cols)
                            #yLeftBottom = int(detections[0, 0, i, 4] * rows)
                            #xRightTop   = int(detections[0, 0, i, 5] * cols)
                            #yRightTop   = int(detections[0, 0, i, 6] * rows)


                            dims = np.array([fW, fH, fW, fH])
                            box = detections[0, 0, i, 3:7] * dims
                            (startX, startY, endX, endY) = box.astype("int")

                            # draw the prediction on the frame
                            label = "{}: {:.2f}%".format(CLASSES[idx],
                                    confidence * 100)
                            cv2.rectangle(frame, (startX, startY), (endX, endY),
                                    COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            key = CLASSES[idx]
                            A = [confidence,startX,endX, startY, endY, int(time.time())]
                            #print(A)
                            if( not key in classes): classes[key] = [A]
                            else: classes[key].append(A)
                    if paramsQueue.full():
                        print("paramsQueue is full")
                        paramsQueue.get()
                        paramsQueue.put(classes)
                        continue
                    paramsQueue.put(classes)

            
            if imagesQueue.full():
                    print("imagesQueue is full")
                    imagesQueue.get()
                    #imagesQueue.task_done()
            jpg = cv2.imencode('.jpg', frame)[1].tobytes()
            imagesQueue.put(jpg)

               
            # Accumulate statistic
            #print('Test4 - scrn_stats: ' , scrn_stats )
            #p_scrn_stats = Process(scrn_stats.refresh, args=(classes,))
            #p_scrn_stats.daemon = True
            #p_scrn_stats.start()
            
            

            # show the output frame when need to test is working or not
            #---> uncommment me
            #print(pnt)
            #pnt += 1
            #pnt %= IMAGE_BUFFER
            
            #if (__name__ == '__main__'):
            #    cv2.imshow("Frame", frame)
            #    key = cv2.waitKey(1) & 0xFF
                # update the FPS counter
            #    ps.update()
            
            #--> uncomment this one when run as from other module
            #jpeg = cv2.imencode('.jpg', frame)[1].tobytes()
            #print(jpeg)
            #yield jpeg
            
            #cv2.imencode('.jpg', frame)[1].tostring()

    if (__name__ == '__main__'):
    # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            

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
inputQueue = Queue(maxsize=IMAGE_BUFFER)
outputQueue = Queue(maxsize=IMAGE_BUFFER)
imagesQueue = Queue(maxsize=IMAGE_BUFFER)
paramsQueue = Queue(maxsize=PARAMS_BUFFER)
   

imagesQueue.put(imgs[0])
imagesQueue.put(imgs[1])
imagesQueue.put(imgs[2])
imagesQueue.put(imgs[0])
imagesQueue.put(imgs[1])
imagesQueue.put(imgs[2])

detections = None
vs = None
fps = None
p_get_frame = None

if (__name__ == '__main__'):
    # construct the argument parse and parse the arguments
    #initialise Screen statistic object
    show_video = True
    global scrn_stats
    scrn_stats = Screen_statistic(paramsQueue)

    ap = argparse.ArgumentParser()
    ap.add_argument("-v","--video_file", required=False,
            help="video file , could be access to remote location." )
    ap.add_argument("-p", "--prototxt", required=True,
            help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
            help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.25,
            help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    

    
else:

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
    print(args)





def start():
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # construct a child process *indepedent* from our main process of
    # execution
    print("[INFO] starting process...")
    for i in range(THREAD_NUMBERS):
        p_classifier = Process(target=classify_frame, args=(net,inputQueue,
                outputQueue,))
        p_classifier.daemon = False
        p_classifier.start()


    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    if args['video_file'] != '':
       p_get_frame = initialize_video_stream(args['video_file'])
    return p_get_frame

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
def initialize_video_stream(video_file):
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args['video_file'])
    print("[INFO] Video stream: ", vs)
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()
    if(vs is None):
        print("[INFO] starting video stream failed.")
    else:
        print("[INFO] video stream started.")

    # show the output frame when need to test is working or not
    p_get_frame = Process(target=get_frame, args=(vs,))
    p_get_frame.daemon = False
    p_get_frame.start()
    return p_get_frame

###################### Flask API #########################
app = Flask(__name__)

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
def strToBase64(s):
    return base64.b64encode(s.encode('utf-8'))
def base64ToStr(b):
    return base64.b64decode(b).decode('utf-8')

def detect():
    """Video streaming generator function."""
    while True:
        # print('imagesQueue isEmpty:', imagesQueue.empty())
         if imagesQueue.qsize()>0:
         #imagesQueue.task_done()
             yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + imagesQueue.get() + b'\r\n'



def gen_params():
    """Parameters streaming generator function."""
    #scrn_stats.orig_classes
    #uncomment as soon will be ready
    #print('Request to params1', paramsQueue.empty())
    #while(paramsQueue.empty()): a=0
    params = None
    x ="{}"
    if(not paramsQueue.empty()):
        classes = paramsQueue.get()
        #paramsQueue.task_done()
        print(classes)
        params = scrn_stats.refresh(classes)
    #print(params)
        x = json.dumps(params)    
    return x
  

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
   
    return Response( detect(), #mimetype='text/event-stream')
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/params_feed')
def params_feed():
    """Parameters streaming route. Put this in the src attribute of an img tag."""
    return Response( gen_params(),
                    mimetype='text/plain')

if (__name__ == '__main__'):
    start()
    app.run(host='0.0.0.0', threaded=True)
    # debug mode    
    #app.run(debug=True, use_debugger=False, use_reloader=False)
    # if the `q` key was pressed, break from the loop

