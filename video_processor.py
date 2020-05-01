import zmq
import numpy as np
from imutils.video import VideoStream
import os
import argparse

import db
import cv2
import dhash
from PIL import Image
import time
import datetime
import json
from objCountByTimer import ObjCountByTimer
from multiprocessing import Process

# load the COCO class labels our YOLO model was trained on
labelsPath = "yolo-coco/coco.names"
CLASSES = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3),
                           dtype="uint8")

LOOKED1 = {"car": [], "person": [], "bus": [], "truck": [], "motorbike": [], "train":[]}
subject_of_interes = ["car", "person", "bus", "motorbike", "train"]

DNN_TARGET_MYRIAD = False
HASH_DELTA = 3  # bigger number  more precise object's count
DIMENSION_X = 416
DIMENSION_Y = 416
piCameraResolution = (640, 480)  # (1024,768) #(640,480)  #(1920,1080) #(1080,720) # (1296,972)
piCameraRate = 16
NUMBER_OF_THREADS = 2
BOX_EXTENDER = 30
SQLITE_DB = "framedata.db"
args = {}
separator = "="

class Detection:
    def __init__(self, sqlite_db, confidence, prototxt, model):
        self.confidence = confidence
        self.threshold = 0.05
        self.hashes = {}
        self.sqlite_db = sqlite_db
        self.topic_label = ''
        self.net = self.video_s = None
        # derive the paths to the YOLO weights and model configuration
        self.weightsPath = model    # "yolo-coco/yolov3.weights"
        self.configPath = prototxt   #"yolo-coco/yolov3.cfg"
     
        print("[INFO] passing prototext: {} model: {} ".format(prototxt, model))
        context = zmq.Context()
        self.receiver = context.socket(zmq.PULL)
        self.receiver.connect("tcp://localhost:5555")
        self.sender = context.socket(zmq.PUSH)
        self.sender.connect("tcp://localhost:5556")


        for i in range(NUMBER_OF_THREADS):
            p_get_frame = Process(target=self.classify)
            p_get_frame.daemon = True
            p_get_frame.start()
            time.sleep(0.0025)

    def classify(self):
        if self.net is None:
            # load our YOLO object detector trained on COCO dataset (80 classes)
            # and determine only the *output* layer names that we need from YOLO
            print("[INFO] loading YOLO from disk...")
            self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
            print("step 1.") 
            ln = self.net.getLayerNames()
            print("step 2.")
            self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            print("step 3.")  
    # specify the target device as the Myriad processor on the NCS
            if DNN_TARGET_MYRIAD:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
            else:
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Going into the while(True) loop") 
        while True:
            try:
                cam,frame = receiver.recv_pyobj()
                #frame = self.read_video_stream(self.video_s)
                print("frame" +frame.shape)
                if frame is None:
                   continue
            except:
                continue
            frame = self.classify_frame(self.net, self.ln, frame, cam)
            output_queue.put_nowait(frame)


    def classify_frame(self, net, ln, frame, cam):
        # print(" Classify frame ... --->")
        # draw at the top left corner of the screen
        cv2.putText(frame, self.topic_label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        conn = db.create_connection(self.sqlite_db)
        # _frame = cv2.resize(frame, (DIMENSION_X, DIMENSION_Y))
        # _frame = imutils.resize(frame,DIMENSION_X)

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        # blob = cv2.dnn.blobFromImage(frame, 0.003921568627451, (416, 416), swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(frame, 0.003921568627451,
                                     (DIMENSION_X, DIMENSION_Y), swapRB=True, crop=False)

        # set the blob as input to our deep learning object
        # detector and obtain the detections
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        # loop over the detections
        (H, W) = frame.shape[:2]
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        # logger.debug(detections)
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # extract the confidence (i.e., probability) associated
                # with the prediction
                # filter out weak detections by ensuring the `confidence`
                # is greater than the minimum confidence
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                key = CLASSES[classIDs[i]]
                if key not in LOOKED1:
                    continue
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0] - BOX_EXTENDER, boxes[i][1] - BOX_EXTENDER)
                (w, h) = (boxes[i][2] + 2*BOX_EXTENDER, boxes[i][3] + 2*BOX_EXTENDER)

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                # use 20 pixels from the top for labeling
                crop_img_data = frame[y: y + h , x :x + w ]
                try:
                    hash = dhash(crop_img_data)
                    now = datetime.datetime.now()
                    day = "{date:%Y-%m-%d}".format(date=now)
                    # do_statistic(conn, cam, self.hashes)
                    db.insert_frame(conn, hash, day, time.time(), key, crop_img_data, w, h, cam)
                except:
                    continue
                # draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}:".format(CLASSES[classIDs[i]])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if self.hashes.get(key, None) is None:
                    # count objects for last sec, last 5 sec and last minute
                    self.hashes[key] = ImageHashCodesCountByTimer()
                    if not self.hashes[key].add(hash):
                        continue
                else:
                    # if not is_hash_the_same(hash,hashes[key]): hashes[key].add(hash)
                    if not self.hashes[key].add(hash):
                        continue
                    label = ''
                    for key in self.hashes:
                        if self.hashes[key].getCountedObjects() == 0:
                            continue
                        label += ' ' + key + ':' + str(self.hashes[key].getCountedObjects())
                    self.topic_label = label

                    do_statistic(conn, cam, self.hashes)
        return frame



class ImageHashCodesCountByTimer(ObjCountByTimer):
    def equals(self, hash1, hash2):
        delta = hash1 - hash2
        if delta < 0:
            delta -= delta
        return delta < HASH_DELTA


def do_statistic(conn, cam, hashes):
    params = get_parameters_json(hashes, cam)
    db.insert_statistic(conn, params)

def get_parameters_json(hashes, cam):
    ret = []
    for key in hashes:
        # logging.debug(images[key])
        trace = Trace()
        trace.name = key
        trace.cam = cam
        tm = int(time.time())  # strftime("%H:%M:%S", localtime())
        trace.hashcodes = hashes[key].toString()
        trace.x = tm
        # last = len(hashes[key].counted) -1
        trace.y = hashes[key].getCountedObjects()
        trace.text = str(trace.y) + ' ' + key + '(s)'
        ret.append(trace.__dict__)  # used for proper JSON generation (dictionary)
    # ret.append(trace)
    # logging.debug( trace.__dict__ )
    return ret

class Trace(dict):
    def __init__(self):
        dict.__init__(self)
        self.cam = 0
        self.x = 0
        self.y = 0
        self.name = ''
        self.text = ''
        self.filenames = []

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

def dhash(image_data, hashSize=8):
    image_out = np.array(image_data).astype(np.uint8)

    # convert the image to grayscale and compute the hash

    image = cv2.cvtColor(image_out, cv2.COLOR_BGR2GRAY)

    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(image, (hashSize + 1, hashSize), interpolation=cv2.INTER_LINEAR)
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])



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

    # global scrn_stats
    # scrn_stats = Screen_statistic(paramsQueue)

    ap = argparse.ArgumentParser()
    ap.add_argument("-nw", "--not_show_in_window", required=False,
                    help="video could be shown in window.", action="store_true", default=False)
    ap.add_argument("-v", "--video_file", required=False,
                    help="video file , could be access to remote location.")
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=False,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, required=False,
                    help="minimum probability to filter weak detections")
    more_args = vars(ap.parse_args())

    more_args = {k: v for k, v in more_args.items() if v is not None}
    # if more_args["confidence"] == 0.0:more_args["confidence"] = args["confidence"]

    args.update(more_args)


configure(args)

Detection(SQLITE_DB, float(args["confidence"]), args["prototxt"], args["model"]);





 
