import re
import cv2
import torch
import pandas
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='traffic.mp4')
parser.add_argument('--demo', type=bool, default=False)
args = parser.parse_args()

save_dir = '../segmentation_results/yolo/new/'
frame_path = save_dir + 'frames/'
txt_path  = save_dir + 'labels/'
mask_path = save_dir + 'masks/'

TRANSPORT = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']

def erosion(src):
    erosion_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    
    erosion_dst = cv2.erode(src, element, iterations=1)
    return erosion_dst


def dilatation(src):
    dilatation_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatation_size, dilatation_size))

    dilatation_dst = cv2.dilate(src, element, iterations=10)
    return dilatation_dst

def opening(src):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(src, cv2.MORPH_OPEN, element)

def closing(src):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(src, cv2.MORPH_CLOSE, element, iterations=5)

def process(src):
    #return dilatation(erosion(src))
    return closing(opening(src))

def fill_holes(mask):
    mask_floodfill = mask.copy()
    h, w = mask.shape[:2]
    mask2 = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask_floodfill, mask2, (0,0), 255)
    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)

    return mask | mask_floodfill_inv

def box_filter(results, iou_treshold=0.3, iosa_treshold=0.4): #iou_treshold=0.5, iosa_treshold=0.8, iou_frist=True, select_bigger=False
    def IoU(box1, box2):
        a1, b1 = box1
        a2, b2 = box2

        a1x, a1y = a1
        b1x, b1y = b1

        a2x, a2y = a2
        b2x, b2y = b2

        area1 = (b1x - a1x) * (b1y - a1y)
        area2 = (b2x - a2x) * (b2y - a2y)

        xleft = max(a1x, a2x)
        ytop = max(a1y, a2y)
        xright = min(b1x, b2x)
        ybot = min(b1y, b2y)

        w = max(0, xright - xleft)
        h = max(0, ybot - ytop)

        intersection_area = w * h
        union_area = area1 + area2 - intersection_area
        IoU = intersection_area / union_area
        return IoU

    def IoSA(box1, box2):
        a1, b1 = box1
        a2, b2 = box2

        a1x, a1y = a1
        b1x, b1y = b1

        a2x, a2y = a2
        b2x, b2y = b2

        area1 = (b1x - a1x) * (b1y - a1y)
        area2 = (b2x - a2x) * (b2y - a2y)

        xleft = max(a1x, a2x)
        ytop = max(a1y, a2y)
        xright = min(b1x, b2x)
        ybot = min(b1y, b2y)

        w = max(0, xright - xleft)
        h = max(0, ybot - ytop)

        intersection_area = w * h
        smallest_area = min(area1, area2)

        return area1 < area2, intersection_area / smallest_area

    def is_on_border(box, box_crop, eps):
        if box_crop == 1:
            return abs(box[0][0] - top_left1[0]) < eps or \
                abs(box[0][1] - top_left1[1]) < eps or \
                abs(box[1][0] - bot_right1[0]) < eps or \
                abs(box[1][1] - bot_right1[1]) < eps
        elif box_crop == 2:
            return abs(box[0][0] - top_left2[0]) < eps or \
                abs(box[1][0] - bot_right2[0]) < eps or \
                abs(box[1][1] - bot_right2[1]) < eps

    keep = []
    while (not results.empty):
        #choose the box with the highest confidence
        max_conf = 0
        max_conf_idx = 0
        for row in results.itertuples(index = True): #index?????
            idx = row[0]   
            conf  = row[5] 
            if (conf > max_conf):
                box = [(round(row[1]), round(row[2])), (round(row[3]), round(row[4]))]
                box_crop = row[8]
                max_conf_idx = idx
        results.drop(labels = max_conf_idx, axis = 0, inplace=True)

        #remove boxes that intersects with the choosen box
        drop_indices = []
        for row in results.itertuples(index = True):
            box_to_check = [(round(row[1]), round(row[2])), (round(row[3]), round(row[4]))]
            box_to_check_crop = row[8]

            iou = IoU(box, box_to_check)
            is_first_smaller, iosa = IoSA(box, box_to_check)
            if iosa > iosa_treshold:
                if box_crop == 2 and box_to_check_crop == 3:
                    if is_on_border(box, box_crop, 10):
                        box = box_to_check
                elif box_crop == 3 and box_to_check_crop == 2:
                    if not is_on_border(box_to_check, box_to_check_crop, 10):
                        box = box_to_check
                elif box_crop == 1 and box_to_check_crop == 2:
                    if is_on_border(box, box_crop, 2):
                        box = box_to_check
                elif box_crop == 2 and box_to_check_crop == 1:
                    if not is_on_border(box_to_check, box_to_check_crop, 2):
                        box = box_to_check
                elif box_crop == 3 and box_to_check_crop == 1:
                    box = box_to_check
                elif box_crop == box_to_check_crop:
                    if iou < iou_treshold: #or iosa < 0.7:
                        continue
                    elif not is_first_smaller:
                        box = box_to_check

                drop_indices.append(row[0])

        keep.append(box)
        results.drop(labels = drop_indices, axis = 0, inplace=True)

    return keep


def write_boxes(results, frame, frame_number):
    frame_name = 'frame_' + str(frame_number)

    boxes = box_filter(results)
    yaml_writer = cv2.FileStorage(txt_path + frame_name + '.yml', cv2.FileStorage_WRITE | cv2.FileStorage_FORMAT_YAML)
    yaml_writer.startWriteStruct("boxes", cv2.FileNode_SEQ)

    for box in boxes:
        xmin = box[0][0] #returns integer
        ymin = box[0][1] 
        xmax = box[1][0] 
        ymax = box[1][1]

        yaml_writer.startWriteStruct("", cv2.FileNode_MAP)
        yaml_writer.write("x_min", xmin)
        yaml_writer.write("y_min", ymin)
        yaml_writer.write("x_max", xmax)
        yaml_writer.write("y_max", ymax)
        yaml_writer.endWriteStruct()

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))

    yaml_writer.endWriteStruct()
    yaml_writer.release()

    cv2.imwrite(frame_path + frame_name + '.png', frame)
    return frame


def show_boxes(results, frame, frame_number):
    frame_name = 'frame_' + str(frame_number)

    #boxes = box_filter(results, 0.7, 0.4) !!!!!!!good
    boxes = box_filter(results)

    for box in boxes:
        xmin = box[0][0] #returns integer
        ymin = box[0][1] 
        xmax = box[1][0] 
        ymax = box[1][1]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))

    cv2.rectangle(frame, top_left1, bot_right1, (255, 0, 0))
    cv2.rectangle(frame, top_left2, bot_right2, (0, 0, 255))
    cv2.imwrite('/tmp/segmentation_results/yolo/new/frames/' + frame_name + '.png', frame)
    return frame


backSub = cv2.createBackgroundSubtractorMOG2()
backSub.setHistory(3000)
backSub.setShadowValue(0)


def write_mask(frame, frame_number):
    frame_name = 'frame_' + str(frame_number) + '.png'
    mask = backSub.apply(frame)
    processed_mask = fill_holes(process(mask))
    cv2.imwrite(mask_path + frame_name, processed_mask)
    return processed_mask


capture = cv2.VideoCapture(args.input)
width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
fps = capture.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

#video_writer = cv2.VideoWriter("../prepared_datasets/demo.mp4", fourcc, fps, (width, height))

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#model = torch.hub.load('ultralytics/yolov5', 'yolov5n6')

model.classes = [1, 2, 3, 5, 7]

top_left1 = (495, 182)
crop_size1 = (280, 125)
bot_right1 = (top_left1[0] + crop_size1[0], top_left1[1] + crop_size1[1])

top_left2 = (314, 261)
crop_size2 = (637, 324)
bot_right2 = (top_left2[0] + crop_size2[0], top_left2[1] + crop_size2[1])

frame_number = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    frame_cropped1 = frame[top_left1[1] : top_left1[1] + crop_size1[1], top_left1[0] : top_left1[0] + crop_size1[0]]
    frame_cropped2 = frame[top_left2[1] : top_left2[1] + crop_size2[1], top_left2[0] : top_left2[0] + crop_size2[0]]

    # OpenCV image (BGR to RGB)
    img = frame[..., ::-1]
    img_cropped1 = frame_cropped1[..., ::-1]
    img_cropped2 = frame_cropped2[..., ::-1]

    res = model([img, img_cropped1, img_cropped2], size=640)

    outputs = res.pandas().xyxy[0]
    outputs_cropped1 = res.pandas().xyxy[1]
    outputs_cropped2 = res.pandas().xyxy[2]

    outputs.insert(7, "crop", 3)
    outputs_cropped1.insert(7, "crop", 1)
    outputs_cropped2.insert(7, "crop", 2)

    outputs_cropped1[['xmin', 'xmax']] += top_left1[0]
    outputs_cropped1[['ymin', 'ymax']] += top_left1[1]

    outputs_cropped2[['xmin', 'xmax']] += top_left2[0]
    outputs_cropped2[['ymin', 'ymax']] += top_left2[1]

    outputs = outputs.append(outputs_cropped1, ignore_index=True)
    outputs = outputs.append(outputs_cropped2, ignore_index=True)


    mask = write_mask(frame, frame_number)
    frame = write_boxes(outputs, frame, frame_number)
    
    if (args.demo):
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output_frame = cv2.addWeighted(frame, 0.6, mask_bgr, 0.4, 0.0)
        cv2.imshow('qwe', output_frame)
        #video_writer.write(output_frame)
    else:
        cv2.imshow('qwe', frame)
    cv2.waitKey(10)
    frame_number += 1