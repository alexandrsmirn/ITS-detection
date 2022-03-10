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

# for output bounding box post-processing
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

#def plot_results(pil_img, prob, boxes):
#    plt.figure(figsize=(16,10))
#    plt.imshow(pil_img)
#    ax = plt.gca()
#    colors = COLORS * 100
#    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
#        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                   fill=False, color=c, linewidth=1))
#        cl = p.argmax()
#        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#        ax.text(xmin, ymin, text, fontsize=5,
#                bbox=dict(facecolor='yellow', alpha=0.5))
#    plt.axis('off')
#    plt.savefig("mygraph.png")

def plot_results(frame, prob, boxes):
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        if (CLASSES[cl] in TRANSPORT):
            p1 = (int(xmin), int(ymin))
            p2 = (int(xmax), int(ymax))
            cv2.rectangle(frame, p1, p2, (0, 255, 0))
            #cv2.putText(frame, CLASSES[cl], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        
    #cv2.imwrite("results/detr/result.png", frame)
    cv2.imshow('qwe', frame)
    cv2.waitKey(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.to(device)
model.eval()

#url = "http://images.cocodataset.org/val2017/000000281759.jpg"
frame = cv2.imread("/home/alex/prog/cv/crop_experiment/frame-6245.png")
im = Image.open("/home/alex/prog/cv/crop_experiment/frame-6245.png")

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0).to(device)

# propagate through the model
outputs = model(img)
#print(outputs)

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.6

# convert boxes from [0; 1] to image scales
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
print(bboxes_scaled)

plot_results(frame, probas[keep], bboxes_scaled)