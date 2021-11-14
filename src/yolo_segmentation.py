import os

command = "python ../yolov5/detect.py --source ../traffic.mp4 --line-thickness 2 --hide-labels --hide-conf --save-txt --project ../segmentation_results/yolo"

os.system(command)