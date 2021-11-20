import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='traffic.mp4')
args = parser.parse_args()

#command = f"python ../yolov5/detect_new.py --weights yolov5n6.pt --img 1280 --source {args.input} --line-thickness 2 --hide-labels --hide-conf --save-txt --project ../segmentation_results/yolo"
command = f"python ../yolov5/detect_new.py --source {args.input} --line-thickness 1 --hide-labels --hide-conf --save-txt --project ../segmentation_results/yolo"

os.system(command)