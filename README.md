flask-video-streaming
=====================
This is initial pilot project for OpenSource project aka iSpy. All ideas are welcome. Repository will be open for commit by request to 
zeroprg@yahoo.com

The idea of project is simple 'how to make flexible as possible application which will be similar to iSpy: http://github.com/ispysoftware/iSpy'. 
User can specify his own video feeds through config file or command line.
To run app. type :
```
python pi_object_detection.py
```
This app working under Linux with python3.5 and working even on low perfomance Raspberry Pi 3. Under Raspberry Pi it can  easeally support 4 videostreams.
It support USB web cameras, IP camers.
Short video how it's behaive on YouTube: http://youtu.be/d2LVG4CrFzo
Major feature of application is supporting Caffe Model Nets. Currently used MobileNetSSD_deploy.caffemodel which is fastest and recognize 20 simle objects. 
Object identified by CaffeModel: 

```
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
```
Possible  to use any type of CaffeModel Net.
Application recognize all specified objects of interest.
To configure video stream just modify config.txt:

```
video_file3=http://38.101.209.29:8082/mjpg/video.mjpg?COUNTER
video_file2=http://192.168.1.100:81/videostream.cgi?loginuse=xxxxxxxxxxx
video_file1=http://192.168.1.101:81/videostream.cgi?loginuse=xxxxxxxxxxx
video_file0=picam
prototxt=MobileNetSSD_deploy.prototxt.txt
model=MobileNetSSD_deploy.caffemodel
confidence=0.7
```

To run python unit test run:

```
python3 -m unittest test_ObjCountByTimer.py 
```
