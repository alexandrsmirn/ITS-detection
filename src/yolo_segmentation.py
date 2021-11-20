import re
import cv2
import torch

import argparse

from torchvision.transforms.functional import crop

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='traffic.mp4')
args = parser.parse_args()

save_dir = '../segmentation_results/yolo/new/'
save_path = save_dir + 'frames/'
txt_path  = save_dir + 'labels/'
mask_path = save_dir + 'masks/' 

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


def process(src):
    return dilatation(erosion(src))


TRANSPORT = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']


def write_boxes(results, frame, frame_number):
    frame_name = 'frame_' + str(frame_number)
    yaml_writer = cv2.FileStorage(txt_path + frame_name + '.yml', cv2.FileStorage_WRITE | cv2.FileStorage_FORMAT_YAML)
    yaml_writer.startWriteStruct("boxes", cv2.FileNode_SEQ)

    for row in results.itertuples(index = False):
        cls = row[6]   
        conf  = row[4] 
        if (cls in TRANSPORT):
            xmin = round(row[0]) #returns integer
            ymin = round(row[1]) 
            xmax = round(row[2]) 
            ymax = round(row[3]) 

            yaml_writer.startWriteStruct("", cv2.FileNode_MAP)
            yaml_writer.write("class", cls)
            yaml_writer.write("x_min", xmin)
            yaml_writer.write("y_min", ymin)
            yaml_writer.write("x_max", xmax)
            yaml_writer.write("y_max", ymax)
            yaml_writer.endWriteStruct()

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))
    yaml_writer.endWriteStruct()
    yaml_writer.release()

    cv2.imwrite(save_path + frame_name + '.png', frame)
    cv2.imshow('window_name', frame)
    cv2.waitKey(10)


backSub = cv2.createBackgroundSubtractorMOG2()
backSub.setHistory(3000)
backSub.setShadowValue(0)


def write_mask(frame):
    frame_name = 'frame_' + str(frame_number) + '.png'
    mask = backSub.apply(frame)
    cv2.imwrite(mask_path + frame_name, process(mask))


capture = cv2.VideoCapture(args.input)
width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

frame_number = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    #top_left = (495, 182)
    #crop_size = (280, 125)
    #frame_cropped = frame[top_left[1] : top_left[1] + crop_size[1], top_left[0] : top_left[0] + crop_size[0]]

    # OpenCV image (BGR to RGB)
    img = frame[..., ::-1]
    #img_cropped = frame_cropped[..., ::-1]

    outputs = model(img, size=640).pandas().xyxy[0]
    #outputs.print()
    #outputs_cropped = model(img_cropped, size=640).pandas().xyxy[0]

    #probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    #probas_cropped = outputs_cropped['pred_logits'].softmax(-1)[0, :, :-1]
    #keep = probas.max(-1).values > 0.9 #0.99 good
    #keep_cropped = probas_cropped.max(-1).values > 0.6

    #(c_x, c_y) = (width / crop_size[0], height / crop_size[1])
    #outputs_cropped[['xmin', 'ymin', 'xmax', 'ymax']] *= (c_x, c_y, c_x, c_y)
    #
    #print(outputs_cropped)
    #
    #outputs_cropped[['xmin', 'xmax']] += top_left[0]
    #outputs_cropped[['xmin', 'xmax']] += top_left[1]
    #
    #outputs.append(outputs_cropped, ignore_index=True)

    write_mask(frame)
    write_boxes(outputs, frame, frame_number)

    frame_number += 1