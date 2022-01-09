import math
from shutil import move

from PIL import Image
import requests
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

#import ipywidgets as widgets
#from IPython.display import display, clear_output
import cv2

import torch
from torch import nn
from torch._C import device
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
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    return b


def box_filter(boxes, confs, classes, crops, iou_treshold=0.3, iosa_treshold=0.4): #iou_treshold=0.5, iosa_treshold=0.8, iou_frist=True, select_bigger=False
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

    def IoSA(box1, box2):
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
        smallest_area = min(area1, area2)

        return area1 < area2, intersection_area / smallest_area

    def is_on_border(box, box_crop, eps):
        if box_crop == 1:
            return abs(box[0] - top_left1[0]) < eps or \
                abs(box[1] - top_left1[1]) < eps or \
                abs(box[2] - bot_right1[0]) < eps or \
                abs(box[3] - bot_right1[1]) < eps
        #elif box_crop == 2:
        #    return abs(box[0] - top_left2[0]) < eps or \
        #        abs(box[2] - bot_right2[0]) < eps or \
        #        abs(box[3] - bot_right2[1]) < eps

    keep = []
    confs_keep = []
    classes_keep = []
    while (not len(boxes) == 0):
        #choose the box with the highest confidence
        max_conf = 0
        max_conf_idx = 0
        for idx, (curr_box, curr_conf, curr_cl, curr_crop) in enumerate(zip(boxes, confs, classes, crops)):
            if (curr_conf > max_conf):
                max_conf = curr_conf
                max_conf_idx = idx

                box = curr_box
                box_conf  = curr_conf
                box_cl = curr_cl
                box_crop = curr_crop

        del boxes[max_conf_idx]
        del confs[max_conf_idx]
        del classes[max_conf_idx]
        del crops[max_conf_idx]

        #remove boxes that intersects with the choosen box
        drop_indices = []
        for curr_idx, (curr_box, curr_conf, curr_cl, curr_crop) in enumerate(zip(boxes, confs, classes, crops)):
            iou = IoU(box, curr_box)
            is_first_smaller, iosa = IoSA(box, curr_box)
            if iosa > iosa_treshold:
                if box_crop == 3 and curr_crop == 1:
                    box = curr_box
                    box_conf  = curr_conf
                    box_cl = curr_cl
                    box_crop = curr_crop
                elif box_crop == curr_crop:
                    if iou < iou_treshold and iosa < 0.99:
                        continue
                    elif not is_first_smaller:
                        box = curr_box
                        box_conf  = curr_conf
                        box_cl = curr_cl
                        box_crop = curr_crop
                
                drop_indices.append(curr_idx)

        keep.append(box)
        confs_keep.append(box_conf)
        classes_keep.append(box_cl)
        
        for index in sorted(drop_indices, reverse=True):
            del boxes[index]
            del confs[index]
            del classes[index]
            del crops[index]

    return keep, confs_keep, classes_keep


def create_mask(result):
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
    return panoptic_seg


def cvt_results(prob, boxes, crop_num):
    box_list = []
    conf_list = []
    class_list = []
    crop_list = []
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        if (cl < 30 and CLASSES[cl] in TRANSPORT):
            conf = p[cl].item()
            box = (int(xmin), int(ymin), int(xmax), int(ymax))
            box_list.append(box)
            conf_list.append(conf)
            class_list.append(cl)
            crop_list.append(crop_num)

    return box_list, conf_list, class_list, crop_list


def plot_cvt_results_to_file(frame, keep, confs, clas, frame_number, panoptic_segm, mask):
    yaml_writer = cv2.FileStorage("/tmp/detr_segm/labels/frame-"+str(frame_number)+".yml", cv2.FileStorage_WRITE | cv2.FileStorage_FORMAT_YAML)
    yaml_writer.startWriteStruct("boxes", cv2.FileNode_SEQ)

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

#torch.set_num_interop_threads()
device = torch.device("cpu") #or cpu
#model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250, threshold=0.7)
model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet50_panoptic', pretrained=True, return_postprocessor=True, num_classes=250, threshold=0.6)
model.to(device)
model.eval()

#top_left1 = (495, 0)
#crop_size1 = (280, 125)

top_left1 = (490, 20)
crop_size1 = (295, 150)
bot_right1 = (top_left1[0] + crop_size1[0], top_left1[1] + crop_size1[1])

frame_number = 6255
while True:
    frame_cv = cv2.imread("/home/alex/prog/cv/prepared_datasets/Carla-final/from_0_camera/frame-" + str(frame_number) + ".png")
    if frame_cv is None:
        frame_number += 1
        continue
    frame_cropped_1 = frame_cv[top_left1[1] : top_left1[1] + crop_size1[1], top_left1[0] : top_left1[0] + crop_size1[0]]
    #cv2.imshow('qwd', frame_cropped_1)

    im_3 = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
    im_1 = Image.fromarray(cv2.cvtColor(frame_cropped_1, cv2.COLOR_BGR2RGB))

    torch.cuda.empty_cache()

    img_1 = transform(im_1).unsqueeze(0).to(device)
    out_1 = model(img_1)
    probas_1 = out_1['pred_logits'].softmax(-1)[0, :, :-1]
    keep_1 = probas_1.max(-1).values > 0.9
    confs_1 = probas_1[keep_1].max(-1).values
    bboxes_scaled_1 = rescale_bboxes(out_1['pred_boxes'][0, keep_1], im_1.size) + torch.tensor([top_left1[0], top_left1[1], top_left1[0], top_left1[1]], dtype=torch.float32, device=device)
    box_list_1, conf_list_1, class_list_1, crop_list_1 = cvt_results(probas_1[keep_1], bboxes_scaled_1, 1)
    #print(box_list_1)

    del img_1
    del out_1
    del probas_1
    del bboxes_scaled_1
    torch.cuda.empty_cache()

    img_3 = transform(im_3).unsqueeze(0).to(device)
    out_3 = model(img_3)
    probas_3 = out_3['pred_logits'].softmax(-1)[0, :, :-1]
    keep_3 = probas_3.max(-1).values > 0.5
    confs_3 = probas_3[keep_3].max(-1).values
    bboxes_scaled_3 = rescale_bboxes(out_3['pred_boxes'][0, keep_3], im_3.size)

    result = postprocessor(out_3, torch.as_tensor(img_3.shape[-2:]).unsqueeze(0))[0]
    panoptic_seg = create_mask(result)

    box_list_3, conf_list_3, class_list_3, crop_list_3 = cvt_results(probas_3[keep_3], bboxes_scaled_3, 3)

    box_list = box_list_3 + box_list_1
    conf_list = conf_list_3 + conf_list_1
    class_list = class_list_3 + class_list_1
    crop_list = crop_list_3 + crop_list_1

    mask_cv = cv2.cvtColor(panoptic_seg, cv2.COLOR_GRAY2BGR)
    box_keep, confs_keep, class_keep = box_filter(box_list, conf_list, class_list, crop_list)
    plot_cvt_results_to_file(frame_cv, box_keep, confs_keep, class_keep, frame_number, panoptic_seg, mask_cv)

    output_frame = cv2.addWeighted(frame_cv, 0.45, mask_cv, 0.55, 0.0)
    #output_frame = mask_cv
    cv2.imshow('qwe', output_frame)
    #cv2.imwrite('/tmp/detr_segm.png', output_frame)

    del img_3
    del out_3
    del probas_3
    del bboxes_scaled_3
    torch.cuda.empty_cache()

    frame_number += 1
    cv2.waitKey(30)

#nearest neighbour, чтобы не придумывал новые значения!!!
#спросить про ситуацию, когда бокс не детектится, а маска генерится. интересен ли нам такой случай???