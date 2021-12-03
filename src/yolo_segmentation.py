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
save_path = save_dir + 'frames/'
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

def box_filter(results, iou_treshold=0.45, iosa_treshold=0.5, iou_frist=True, select_bigger=True):
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

    keep = []
    while (not results.empty):
        max_conf = 0
        max_conf_idx = 0
        for row in results.itertuples(index = True): #index?????
            idx = row[0]   
            conf  = row[5] 
            if (conf > max_conf):
                box = [(round(row[1]), round(row[2])), (round(row[3]), round(row[4]))]
                max_conf_idx = idx
        results.drop(labels = max_conf_idx, axis = 0, inplace=True)

        drop_indices = []
        for row in results.itertuples(index = True):
            box_to_check = [(round(row[1]), round(row[2])), (round(row[3]), round(row[4]))]
            if iou_frist:
                if IoU(box, box_to_check) > iou_treshold:
                    drop_indices.append(row[0])
                else:
                    is_first_smaller, iosa = IoSA(box, box_to_check)
                    if iosa > iosa_treshold:
                        if is_first_smaller and select_bigger:
                            box = box_to_check
                        drop_indices.append(row[0])
            else:
                is_first_smaller, iosa = IoSA(box, box_to_check)
                if iosa > iosa_treshold:
                    if is_first_smaller and select_bigger:
                        box = box_to_check
                    drop_indices.append(row[0])
                elif IoU(box, box_to_check) > iou_treshold:
                    drop_indices.append(row[0])

        keep.append(box)
        results.drop(labels = drop_indices, axis = 0, inplace=True)

    return keep


def write_boxes(results, frame, frame_number):
    frame_name = 'frame_' + str(frame_number)
    #yaml_writer = cv2.FileStorage(txt_path + frame_name + '.yml', cv2.FileStorage_WRITE | cv2.FileStorage_FORMAT_YAML)
    #yaml_writer.startWriteStruct("boxes", cv2.FileNode_SEQ)
    #for row in results.itertuples(index = False):
    #    cls = row[6]   
    #    conf  = row[4] 
    #    if (cls in TRANSPORT and conf>0.35):
    #        xmin = round(row[0]) #returns integer
    #        ymin = round(row[1]) 
    #        xmax = round(row[2]) 
    #        ymax = round(row[3]) 
    #
    #        #yaml_writer.startWriteStruct("", cv2.FileNode_MAP)
    #        #yaml_writer.write("class", cls)
    #        #yaml_writer.write("x_min", xmin)
    #        #yaml_writer.write("y_min", ymin)
    #        #yaml_writer.write("x_max", xmax)
    #        #yaml_writer.write("y_max", ymax)
    #        #yaml_writer.endWriteStruct()
    #
    #        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))
    #        cv2.putText(frame, cls, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    #yaml_writer.endWriteStruct()
    #yaml_writer.release()

    #cv2.imwrite(save_path + frame_name + '.png', frame)
    #cv2.imshow('window_name', frame)
    #cv2.waitKey(10)

    boxes = box_filter(results, 0.7, 0.4)

    for box in boxes:
        xmin = box[0][0] #returns integer
        ymin = box[0][1] 
        xmax = box[1][0] 
        ymax = box[1][1]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))

    return frame


backSub = cv2.createBackgroundSubtractorMOG2()
backSub.setHistory(3000)
backSub.setShadowValue(0)


def write_mask(frame):
    frame_name = 'frame_' + str(frame_number) + '.png'
    mask = backSub.apply(frame)
    processed_mask = fill_holes(process(mask))
    #cv2.imwrite(mask_path + frame_name, processed_mask)
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
#model.multi_label = True
model.classes = [1, 2, 3, 5, 7]

frame_number = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    top_left = (495, 182)
    crop_size = (280, 125)
    bot_right = (top_left[0] + crop_size[0], top_left[1] + crop_size[1])
    frame_cropped = frame[top_left[1] : top_left[1] + crop_size[1], top_left[0] : top_left[0] + crop_size[0]]

    top_left2 = (314, 261)
    crop_size2 = (637, 324)
    bot_right2 = (top_left2[0] + crop_size2[0], top_left2[1] + crop_size2[1])
    frame_cropped2 = frame[top_left2[1] : top_left2[1] + crop_size2[1], top_left2[0] : top_left2[0] + crop_size2[0]]

    # OpenCV image (BGR to RGB)
    img = frame[..., ::-1]
    img_cropped = frame_cropped[..., ::-1]
    img_cropped2 = frame_cropped2[..., ::-1]

    res = model([img, img_cropped, img_cropped2], size=640)
    outputs = res.pandas().xyxy[0]
    outputs_cropped = res.pandas().xyxy[1]
    outputs_cropped2 = res.pandas().xyxy[2]

    outputs.insert(7, "crop", 0)
    outputs_cropped.insert(7, "crop", 1)
    outputs_cropped2.insert(7, "crop", 2)

    outputs_cropped[['xmin', 'xmax']] += top_left[0]
    outputs_cropped[['ymin', 'ymax']] += top_left[1]

    outputs_cropped2[['xmin', 'xmax']] += top_left2[0]
    outputs_cropped2[['ymin', 'ymax']] += top_left2[1]

    outputs = outputs.append(outputs_cropped, ignore_index=True)
    outputs = outputs.append(outputs_cropped2, ignore_index=True)

    #outputs = outputs_cropped.append(outputs_cropped2, ignore_index=True)
    #outputs = outputs_cropped2

    mask = write_mask(frame)
    frame = write_boxes(outputs, frame, frame_number)
    #frame = write_boxes(outputs, frame, frame_number)
    cv2.rectangle(frame, top_left, bot_right, (255, 0, 0))
    cv2.rectangle(frame, top_left2, bot_right2, (0, 0, 255))
    if (args.demo):
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output_frame = cv2.addWeighted(frame, 0.6, mask_bgr, 0.4, 0.0)
        cv2.imshow('qwe', output_frame)
        #video_writer.write(output_frame)
    else:
        cv2.imshow('qwe', frame)
    cv2.waitKey(10)
    frame_number += 1