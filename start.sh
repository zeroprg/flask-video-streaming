#!/bin/bash
source /home/pi/.profile

# OpenVINO
source /home/pi/openvino/bin/setupvars.sh

# Run programm
cd /home/pi/projects/flask-video-streaming && nohup python3 pi_object_detection.py &

