#!/bin/bash
#command which will fired from dockerfile CMD ["./start_program.sh"]
pkill -f python3
cd  /home/odroid/projects/flask-video-streaming/
git pull --no-edit
bin/bash /opt/py3cv4/bin/activate.sh &&  /opt/py3cv4/bin/python3.5 pi_object_detection.py &
