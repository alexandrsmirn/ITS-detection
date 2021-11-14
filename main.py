from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
    backSub.setDetectShadows(False)
    backSub.setHistory(1000)
    #backSub.setBackgroundRatio(0.7)
else:
    backSub = cv.createBackgroundSubtractorKNN()
    backSub.setDetectShadows(False)
    backSub.setHistory(2000)

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

width  = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
fps = capture.get(cv.CAP_PROP_FPS)
path = ('./output2.avi')
fourcc = cv.VideoWriter_fourcc(*'MJPG')

#video_writer = cv.VideoWriter(path, fourcc, fps, (width, height))
#mask_writer = cv.VideoWriter(path, fourcc, fps, (width, height), False)

def erosion(src):
    erosion_size = 2;    
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_size, erosion_size))
    
    erosion_dst = cv.erode(src, element)
    return erosion_dst

def dilatation(src):
    dilatation_size = 2
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilatation_size, dilatation_size))

    dilatation_dst = cv.dilate(src, element)
    return dilatation_dst


while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)   
    
    
    cv.rectangle(fgMask, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(fgMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    #mask_writer.write(fgMask)
    #cv.imshow('Frame', frame)
    cv.imshow('FG Mask', dilatation(erosion(fgMask)))
    #cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break