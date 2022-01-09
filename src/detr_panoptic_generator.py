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


def box_filter(boxes, confs, classes, iou_treshold=0.3): #iou_treshold=0.5, iosa_treshold=0.8, iou_frist=True, select_bigger=False
    def IoU(box1, box2):
        a1x, a1y, b1x, b1y = box1
        a2x, a2y, b2x, b2y = box2

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

    keep = []
    confs_keep = []
    classes_keep = []
    while (not len(boxes) == 0):
        #choose the box with the highest confidence
        max_conf = 0
        max_conf_idx = 0
        for idx, (curr_box, curr_conf, curr_cl) in enumerate(zip(boxes, confs, classes)):
            if (curr_conf > max_conf):
                max_conf = curr_conf
                max_conf_idx = idx

                box = curr_box
                box_conf  = curr_conf
                box_cl = curr_cl

        del boxes[max_conf_idx]
        del confs[max_conf_idx]
        del classes[max_conf_idx]

        #remove boxes that intersects with the choosen box
        drop_indices = []
        for idx, curr_box in enumerate(boxes):
            if (IoU(box, curr_box) > iou_treshold):
                drop_indices.append(idx)

        keep.append(box)
        confs_keep.append(box_conf)
        classes_keep.append(box_cl)
        
        for index in sorted(drop_indices, reverse=True):
            del boxes[index]
            del confs[index]
            del classes[index]

    return keep, confs_keep, classes_keep


def cvt_results(prob, boxes):
    box_list = []
    conf_list = []
    class_list = []
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        if (cl < 30 and CLASSES[cl] in TRANSPORT):
            conf = p[cl].item()
            box = (int(xmin), int(ymin), int(xmax), int(ymax))
            box_list.append(box)
            conf_list.append(conf)
            class_list.append(cl)

    return box_list, conf_list, class_list


def plot_cvt_results_to_file(frame, prob, boxes, frame_number, panoptic_segm, mask):
    yaml_writer = cv2.FileStorage("/tmp/detr_segm/labels/frame-"+str(frame_number)+".yml", cv2.FileStorage_WRITE | cv2.FileStorage_FORMAT_YAML)
    yaml_writer.startWriteStruct("boxes", cv2.FileNode_SEQ)

    keep, confs, clas = box_filter(*cvt_results(prob, boxes))
    for box, conf, cl in zip(keep, confs, clas):        
        xmin, ymin, xmax, ymax = box
        p1 = (xmin, ymin)
        p2 = (xmax, ymax)
        cv2.rectangle(frame, p1, p2, (0, 255, 0))

        center_x = int(0.5*(xmax + xmin))
        center_y = int(0.5*(ymax + ymin))

        instance_id = panoptic_segm[center_y, center_x]

        yaml_writer.startWriteStruct("", cv2.FileNode_MAP)
        yaml_writer.write("instance_id", instance_id)
        yaml_writer.write("conf", conf)
        yaml_writer.write("class", CLASSES[cl])
        yaml_writer.write("x_min", xmin)
        yaml_writer.write("y_min", ymin)
        yaml_writer.write("x_max", xmax)
        yaml_writer.write("y_max", ymax)
        yaml_writer.endWriteStruct()

    yaml_writer.endWriteStruct()
    yaml_writer.release()
    cv2.imwrite("/tmp/detr_segm/frames/frame-"+str(frame_number)+".jpg", frame)
    cv2.imwrite("/tmp/detr_segm/masks/frame-"+str(frame_number)+".png", mask)
    cv2.waitKey(10)


model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250, threshold=0.6)
model.eval()

frame_number = 6250
while True:
    frame_cv = cv2.imread("/home/alex/prog/cv/prepared_datasets/Carla-final/from_0_camera/frame-" + str(frame_number) + ".png")
    #im = Image.open("/home/alex/prog/cv/prepared_datasets/Carla-final/from_0_camera/frame-" + str(frame_number) + ".png")
    im = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
    if frame_cv is None:
        frame_number += 1
        continue

    img = transform(im).unsqueeze(0)
    out = model(img)

    probas = out['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    confs = probas[keep].max(-1).values

    bboxes_scaled = rescale_bboxes(out['pred_boxes'][0, keep], im.size)
    #plot_results(frame_cv, probas[keep], bboxes_scaled)

    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()

    panoptic_seg_id = rgb2id(panoptic_seg)
    new_shape = panoptic_seg_id.shape

    panoptic_seg = numpy.zeros(new_shape, dtype=numpy.uint8)
    mask_list = []
    for id in range(panoptic_seg_id.max() + 1):
        segm_info = result['segments_info'][id]
        if segm_info['isthing'] is True and CLASSES[segm_info['category_id']] in TRANSPORT:
            mask_list.append(id)

    mask_id_max = len(mask_list)
    print(mask_id_max)
    if not mask_id_max == 0:
        mask_step = 255/(mask_id_max)
    mask_id = 0
    for id in mask_list:
        mask_id = mask_id + mask_step
        if mask_id > 255:
            mask_id = 255
        panoptic_seg[panoptic_seg_id == id] = mask_id


    panoptic_seg = cv2.resize(panoptic_seg, (1280, 720))
    mask_cv = cv2.cvtColor(panoptic_seg, cv2.COLOR_GRAY2BGR)

    plot_cvt_results_to_file(frame_cv, probas[keep], bboxes_scaled, frame_number, panoptic_seg, mask_cv)
    output_frame = cv2.addWeighted(frame_cv, 0.45, mask_cv, 0.55, 0.0)
    #output_frame = mask_cv

    frame_number += 1
    cv2.imshow('qwe', output_frame)
    #cv2.imwrite('/tmp/detr_segm.png', output_frame)
    cv2.waitKey(30)

#nearest neighbour, чтобы не придумывал новые значения!!!
#спросить про ситуацию, когда бокс не детектится, а маска генерится. интересен ли нам такой случай???