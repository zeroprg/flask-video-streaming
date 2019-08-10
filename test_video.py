from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
LOOKED1 = { "car": [], "cat": [],"dog": [],"person":  []}
LOOKED2 = { "car": [], "cat": [],"dog": [], "person": []}

subject_of_interes = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

IMAGES_FOLDER = "static/img/"
PARAMS_FOLDER = "static/params/"

DRAW_RECTANGLES = True
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

def classify_frame( net, inputQueue, outputQueue, hashes, cam):
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



def findObjectInFrame(frame):
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






camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size =(640,480))

time.sleep(0.5)
generator = iter(camera.capture_continuous(rawCapture, format="bgr", use_video_port=True))
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while True:
	frame = next(generator)
	image = frame.array
	cv2.imshow("Frame",image)
	key = cv2.waitKey(1) & 0xFF

	rawCapture.truncate(0)
	findObjectInFrame(image)
	if key==ord("q"):
		break

