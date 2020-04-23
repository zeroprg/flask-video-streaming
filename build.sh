#tar -czvf virtualenvs.tar /home/odroid/.virtualenvs/py3cv4

sudo docker build -t zeroprg/opencv-python-arm:4.1.2-py3.5 -f dockerfile-virtenv .

#rm  virtualenvs.tar
