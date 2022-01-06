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

from detr_segmentation import TRANSPORT

#from yolov5.models.tf import parse_model
torch.set_grad_enabled(False)

import panopticapi
from panopticapi.utils import id2rgb, rgb2id

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

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    #T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250, threshold=0.2)
model.eval()

im = Image.open("/home/alex/prog/cv/crop_experiment/foo-21816.png")
img = transform(im).unsqueeze(0)
out = model(img)

# the post-processor expects as input the target size of the predictions (which we set here to the image size)
result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

palette = itertools.cycle(sns.color_palette())

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
  print(segm_info)

  if segm_info['isthing'] is True and CLASSES[segm_info['category_id']] in TRANSPORT:
    panoptic_seg[panoptic_seg_id == id] = numpy.asarray((1, 1, 1)) * 255
  else:
    panoptic_seg[panoptic_seg_id == id] = numpy.asarray((0, 0, 0)) * 255

frame_cv = cv2.imread("/home/alex/prog/cv/crop_experiment/foo-21816.png")
mask_cv = cv2.resize(cv2.cvtColor(panoptic_seg, cv2.COLOR_RGB2BGR), (1280, 720))
output_frame = cv2.addWeighted(frame_cv, 0.3, mask_cv, 0.7, 0.0)
cv2.imshow('qwe', output_frame)
#cv2.imwrite('/tmp/detr_segm.png', output_frame)
cv2.waitKey(0)