from time import sleep
import zmq
import cv2
import argparse
from imutils.video import VideoStream

piCameraResolution = (640, 480)  # (1024,768) #(640,480)  #(1920,1080) #(1080,720) # (1296,972)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_file", required=False,
                    help="video file , could be access to remote location or local video file.")
args={}
more_args = vars(ap.parse_args())

more_args = {k: v for k, v in more_args.items() if v is not None}
# if more_args["confidence"] == 0.0:more_args["confidence"] = args["confidence"]
args.update(more_args)
video_url = args.get('video_file', None)



def init_video_stream():
    if 'picam' == video_url:
        video_s = VideoStream(usePiCamera=True, resolution=piCameraResolution, framerate=piCameraRate).start()
        sleep(2.0)
    else:
        # grab the frame from the threaded video stream
        video_s = cv2.VideoCapture(video_url)
    return video_s


def read_video_stream(video_s):
    # print("Read video stream .. " + self.video_url)
    if 'picam' == video_url:
        frame = video_s.read()
    else:
        flag, frame = video_s.read()
        if not flag:
            video_s = cv2.VideoCapture(video_url)
            flag, frame = video_s.read()
            return frame
    return frame


context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:5555")
video_s = init_video_stream()
while True:
    frame = read_video_stream(video_s)
    socket.send_pyobj(frame)

