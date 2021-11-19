import math
import cv2

from PIL import Image
from numpy import dtype, number
import numpy
import requests
import matplotlib.pyplot as plt
import argparse
#%config InlineBackend.figure_format = 'retina'

#import ipywidgets as widgets
#from IPython.display import display, clear_output

import torch
from torch import nn
from torch._C import JITException
from torchvision.models import resnet50
import torchvision.transforms as T
from torchvision.transforms.functional import crop
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='traffic.mp4')
args = parser.parse_args()

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

TRANSPORT = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    #print(b)
    return b

def rescale_bboxes_cropped(out_bbox, size, bottom_left):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    b = b + torch.tensor([bottom_left[0], bottom_left[1], bottom_left[0], bottom_left[1]], dtype=torch.float32)
    #print(b)
    return b

backSub = cv2.createBackgroundSubtractorMOG2()
backSub.setHistory(3000)
backSub.setShadowValue(0)

capture = cv2.VideoCapture(args.input)
width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

def plot_results(frame, prob, boxes, frame_number):
    yaml_writer = cv2.FileStorage("../segmentation_results/detr/labels/frame"+str(frame_number)+".yml", cv2.FileStorage_WRITE | cv2.FileStorage_FORMAT_YAML)
    yaml_writer.startWriteStruct("boxes", cv2.FileNode_SEQ)

    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        if (CLASSES[cl] in TRANSPORT):
            p1 = (int(xmin), int(ymin))
            p2 = (int(xmax), int(ymax))
            cv2.rectangle(frame, p1, p2, (0, 255, 0))
            yaml_writer.startWriteStruct("", cv2.FileNode_MAP)
            yaml_writer.write("class", CLASSES[cl])
            yaml_writer.write("x_min", p1[0])
            yaml_writer.write("y_min", p1[1])
            yaml_writer.write("x_max", p2[0])
            yaml_writer.write("y_max", p2[1])
            yaml_writer.endWriteStruct()
            #cv2.putText(frame, CLASSES[cl], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        
        #text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        #ax.text(xmin, ymin, text, fontsize=15,
        #        bbox=dict(facecolor='yellow', alpha=0.5))
    yaml_writer.endWriteStruct()
    yaml_writer.release()
    cv2.imwrite("../segmentation_results/detr/frames/frame"+str(frame_number)+".png", frame)

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

frame_number = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    top_left = (495, 182)
    crop_size = (280, 125)
    frame_cropped = frame[top_left[1] : top_left[1] + crop_size[1], top_left[0] : top_left[0] + crop_size[0]]
    #cv2.imwrite("/tmp/cropped.png", frame_cropped)

    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    im_cropped = Image.fromarray(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))

    img = transform(im).unsqueeze(0)
    img_cropped = transform(im_cropped).unsqueeze(0)
    outputs = model(img)
    outputs_cropped = model(img_cropped)


    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    probas_cropped = outputs_cropped['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9 #0.99 good
    keep_cropped = probas_cropped.max(-1).values > 0.6

    #print(outputs['pred_boxes'][0, keep])
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], (width, height))
    bboxes_scaled_cropped = rescale_bboxes_cropped(outputs_cropped['pred_boxes'][0, keep_cropped], crop_size, top_left)

    #print(bboxes_scaled)
    #print(bboxes_scaled_cropped)

    bboxes_total = torch.cat((bboxes_scaled, bboxes_scaled_cropped), dim=0)
    probas_keep_total = torch.cat((probas[keep], probas_cropped[keep_cropped]))
    #print(total)
    #cv2.imwrite("../segmentation_results/detr/masks/frame"+str(frame_number)+".png", process(backSub.apply(frame)))
    plot_results(frame, probas_keep_total, bboxes_total, frame_number)


    frame_number += 1


# convert boxes from [0; 1] to image scales


