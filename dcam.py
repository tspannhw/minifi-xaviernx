#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import time
import sys
import datetime
import subprocess
import sys
import os
import datetime
import traceback
import math
import base64
import json
import time
from time import gmtime, strftime
import random, string
import psutil
import base64
import uuid
# Importing socket library 
import socket 
import jetson.inference
import jetson.utils

import argparse
import sys
from datetime import datetime

external_IP_and_port = ('198.41.0.4', 53)  # a.root-servers.net
socket_family = socket.AF_INET

def IP_address():
        try:
            s = socket.socket(socket_family, socket.SOCK_DGRAM)
            s.connect(external_IP_and_port)
            answer = s.getsockname()
            s.close()
            return answer[0] if answer else None
        except socket.error:
            return None

# Get MAC address of a local interfaces
def psutil_iface(iface):
    # type: (str) -> Optional[str]
    import psutil
    nics = psutil.net_if_addrs()
    if iface in nics:
        nic = nics[iface]
        for i in nic:
            if i.family == psutil.AF_LINK:
                return i.address
# Random Word
def randomword(length):
 return ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ".lower()) for i in range(length))


host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
ipaddress = IP_address()

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--camera", type=str, default="/dev/video2", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create the camera and display
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()

# display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md

# process frames until user exits
while display.IsOpen():
    img, width, height = camera.CaptureRGBA()

    detections = net.Detect(img, width, height, opt.overlay)
    row = {}
    #print("detected {:d} objects in image".format(len(detections)))
    row['detected'] = str(len(detections))
    counterDetect = 0 
    for detection in detections:
        #print(detection)
        row['detect'  + str(counterDetect) + 'ClassID'] = str(detection.ClassID)
        class_desc = net.GetClassDesc(detection.ClassID)
        row['detect' + str(counterDetect) + 'Class'] = str(class_desc)
        row['detect'  + str(counterDetect) + 'Confidence'] = str(detection.Confidence)
        row['detect'  + str(counterDetect) + 'Left'] = str(detection.Left)
        row['detect'  + str(counterDetect) + 'Top'] = str(detection.Top)
        row['detect'  + str(counterDetect) + 'Right'] = str(detection.Right)
        row['detect'  + str(counterDetect) + 'Bottom'] = str(detection.Bottom)
        row['detect'  + str(counterDetect) + 'Width'] = str(detection.Width)
        row['detect'  + str(counterDetect) + 'Height'] = str(detection.Height)
        row['detect'  + str(counterDetect) + 'Area'] = str(detection.Area)
        row['detect'  + str(counterDetect) + 'Center'] = str(detection.Center)
        counterDetect = counterDetect + 1


    display.RenderOnce(img, width, height)
    display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
    # CPU Temp
    f = open("/sys/devices/virtual/thermal/thermal_zone1/temp","r")
    cputemp = str( f.readline() )
    cputemp = cputemp.replace('\n','')
    cputemp = cputemp.strip()
    cputemp = str(round(float(cputemp)) / 1000)
    cputempf = str(round(9.0/5.0 * float(cputemp) + 32))
    f.close()
    # GPU Temp
    f = open("/sys/devices/virtual/thermal/thermal_zone2/temp","r")
    gputemp = str( f.readline() )
    gputemp = gputemp.replace('\n','')
    gputemp = gputemp.strip()
    gputemp = str(round(float(gputemp)) / 1000)
    gputempf = str(round(9.0/5.0 * float(gputemp) + 32))
    f.close()

    uniqueid = 'detect_uuid_{0}_{1}'.format(randomword(7),datetime.now().strftime("%Y%m%d%H%M%S.%f"))
    uuidID = '{0}_{1}'.format(datetime.now().strftime("%Y%m%d%H%M%S.%f"),uuid.uuid4())

    row['networktime'] = net.GetNetworkTime()
    row['cputemp'] =  cputemp
    row['gputemp'] =  gputemp
    row['gputempf'] =  gputempf
    row['cputempf'] =  cputempf
    row['networkfps'] = net.GetNetworkFPS()
    row['id'] = str(uuidID)
    row['uuid'] =  uniqueid
    row['ipaddress']=ipaddress
    row['host'] = os.uname()[1]
    row['host_name'] = host_name
    row['macaddress'] = psutil_iface('wlan0')
    row['cpu'] = psutil.cpu_percent(interval=1)
    usage = psutil.disk_usage("/")
    row['diskusage'] = "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
    row['memory'] = psutil.virtual_memory().percent

    json_string = json.dumps(row)
    fa=open("/home/nvidia/nvme/logs/detectcam.log", "a+")
    fa.write(json_string + "\n")
    fa.close()
