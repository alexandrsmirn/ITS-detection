import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='frame.png')
args = parser.parse_args()

command = f"python ../yolov5/detect.py --source {args.input} --line-thickness 1 --hide-labels --hide-conf --project ./results/"

os.system(command)
