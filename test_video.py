'''
Simple check for PICamera 
'''
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2


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

