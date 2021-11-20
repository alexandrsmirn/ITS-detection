import re
import cv2
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
frame = cv2.imread('../crop_experiment/frame.png')
img = frame[..., ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model(img, size=640)  # includes NMS

# Results
results.print()  
#results.show()  # or .show()

#print(results.xyxy[0])  # img1 predictions (tensor)
#print(results.pandas().xyxy[0])  # img1 predictions (pandas)

for row in results.pandas().xyxy[0].itertuples(index = False):
    xmin = round(row[0]) #returns integer
    ymin = round(row[1]) 
    xmax = round(row[2]) 
    ymax = round(row[3]) 
    conf  = row[4]
    cls = row[6]
    print(conf, cls, sep=', ')
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))

cv2.namedWindow('window_name', cv2.WINDOW_NORMAL)
cv2.imshow('window_name', frame)
cv2.waitKey() 