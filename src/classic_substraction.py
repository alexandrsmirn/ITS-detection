from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='../traffic.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
    #backSub.setBackgroundRatio(0.7)
else:
    backSub = cv.createBackgroundSubtractorKNN()

backSub.setHistory(3000)
backSub.setShadowValue(0)
#backSub.setShadowThreshold(0.7) #с 0.7 норм

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

width  = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
fps = capture.get(cv.CAP_PROP_FPS)
path = ('../substraction_results/substraction.avi')
fourcc = cv.VideoWriter_fourcc(*'MJPG')

#video_writer = cv.VideoWriter(path, fourcc, fps, (width, height))
mask_writer = cv.VideoWriter(path, fourcc, fps, (width, height), False)

max_elem = 2
max_kernel_size = 10
title_trackbar_kernel_size = 'Kernel size'
title_window = 'FG Mask'

cv.namedWindow(title_window)

def erosion(src):
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size+' erosion\t', title_window) + 1
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_size, erosion_size))
    
    erosion_dst = cv.erode(src, element, iterations=1)
    return erosion_dst

def dilatation(src):
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size+' dilatation\t', title_window) + 1
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilatation_size, dilatation_size))

    dilatation_dst = cv.dilate(src, element, iterations=10)
    return dilatation_dst

def process(src):
    return dilatation(erosion(src))

#def changeBackgroundRatio(src):
#    ratio = cv.getTrackbarPos('Backround ratio\t', title_window)
#    backSub.setBackgroundRatio(ratio/10)

cv.createTrackbar(title_trackbar_kernel_size+' erosion\t', title_window, 0, max_kernel_size, lambda *args: None)
cv.createTrackbar(title_trackbar_kernel_size+' dilatation\t', title_window, 0, max_kernel_size, lambda *args: None)
#cv.createTrackbar('Backround ratio\t', title_window, 7, 10, changeBackgroundRatio)

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    
    #cv.rectangle(fgMask, (10, 2), (100,20), (255,255,255), -1)
    #cv.putText(fgMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    mask_writer.write(process(fgMask))
    cv.imshow('FG Mask', process(fgMask))
   
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break