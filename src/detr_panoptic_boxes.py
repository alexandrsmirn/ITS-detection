import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

#import ipywidgets as widgets
#from IPython.display import display, clear_output
import cv2

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

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

from PIL import Image
import cv2
import requests
import io
import math
import matplotlib.pyplot as plt

import itertools
import seaborn as sns

import torch
from torch import nn
import torchvision.transforms as T
import numpy

#from yolov5.models.tf import parse_model
torch.set_grad_enabled(False)

import panopticapi
from panopticapi.utils import id2rgb, rgb2id

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

#def box_filter(results, confss, iou_treshold=0.3): #iou_treshold=0.5, iosa_treshold=0.8, iou_frist=True, select_bigger=False
#    def IoU(box1, box2):
#        a1x, a1y, b1x, b1y = box1.toList()
#        a2x, a2y, b2x, b2y = box2.toList()
#
#        area1 = (b1x - a1x) * (b1y - a1y)
#        area2 = (b2x - a2x) * (b2y - a2y)
#
#        xleft = max(a1x, a2x)
#        ytop = max(a1y, a2y)
#        xright = min(b1x, b2x)
#        ybot = min(b1y, b2y)
#
#        w = max(0, xright - xleft)
#        h = max(0, ybot - ytop)
#
#        intersection_area = w * h
#        union_area = area1 + area2 - intersection_area
#        IoU = intersection_area / union_area
#        return IoU
#
#    keep = []
#    confs = []
#    classes = []
#    while (not results.empty):
#        #choose the box with the highest confidence
#        max_conf = 0
#        max_conf_idx = 0
#        for row in results.itertuples(index = True): #index?????
#            idx = row[0]   
#            curr_conf = row[5]
#            if (curr_conf > max_conf):
#                max_conf = curr_conf
#                max_conf_idx = idx
#
#                box = [round(row[1]), round(row[2]), round(row[3]), round(row[4])]
#                box_conf  = curr_conf
#                box_cl = row[7]
#
#        results.drop(labels = max_conf_idx, axis = 0, inplace=True)
#
#        #remove boxes that intersects with the choosen box
#        drop_indices = []
#        for row in results:
#            row = row.toList()
#            box_to_check = [round(row[1]), round(row[2]), (round(row[3]), round(row[4]))]
#            box_to_check_crop = row[8]
#
#            
#
#            drop_indices.append(row[0])
#
#        keep.append(box)
#        confs.append(box_conf)
#        classes.append(box_cl)
#        results.drop(labels = drop_indices, axis = 0, inplace=True)
#
#    return keep, confs, classes

def plot_results(frame, prob, boxes):
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        conf = p[cl]
        if (cl < 30 and CLASSES[cl] in TRANSPORT):
            p1 = (int(xmin), int(ymin))
            p2 = (int(xmax), int(ymax))
            cv2.rectangle(frame, p1, p2, (0, 255, 0))
            #cv2.putText(frame, CLASSES[cl], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        
    #cv2.imwrite("results/detr/result.png", frame)
    #cv2.imshow("qwe", frame)
    #cv2.waitKey(0)

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250, threshold=0.4)
model.eval()

im = Image.open("/home/alex/prog/cv/crop_experiment/foo-21816.png")
frame_cv = cv2.imread("/home/alex/prog/cv/crop_experiment/foo-21816.png")
img = transform(im).unsqueeze(0)
out = model(img)

# keep only predictions with 0.7+ confidence
probas = out['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.5
confs = probas[keep].max(-1).values

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(out['pred_boxes'][0, keep], im.size)
plot_results(frame_cv, probas[keep], bboxes_scaled)

# the post-processor expects as input the target size of the predictions (which we set here to the image size)
result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

# The segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
#cv2.imshow('asd', panoptic_seg*125)
#cv2.waitKey(0)
print(panoptic_seg.shape)

# We retrieve the ids corresponding to each mask
panoptic_seg_id = rgb2id(panoptic_seg)
print(panoptic_seg_id.shape)

# Finally we color each mask individually
panoptic_seg[:, :, :] = 0
for id in range(panoptic_seg_id.max() + 1):
  segm_info = result['segments_info'][id]
  #print(segm_info)

  if segm_info['isthing'] is True and CLASSES[segm_info['category_id']] in TRANSPORT:
    panoptic_seg[panoptic_seg_id == id] = numpy.asarray((1, 1, 1)) * 255
  else:
    panoptic_seg[panoptic_seg_id == id] = numpy.asarray((0, 0, 0)) * 255


mask_cv = cv2.resize(cv2.cvtColor(panoptic_seg, cv2.COLOR_RGB2BGR), (1280, 720))
output_frame = cv2.addWeighted(frame_cv, 0.55, mask_cv, 0.45, 0.0)
cv2.imshow('qwe', output_frame)
cv2.imwrite('/tmp/detr_segm.png', output_frame)
cv2.waitKey(0)

#nearest neighbour, чтобы не придумывал новые значения!!!