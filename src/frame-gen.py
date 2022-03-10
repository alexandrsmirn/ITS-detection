import cv2

#capture = cv2.VideoCapture("../traffic_long.mp4")
width  = 1280
height = 720
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

video_writer = cv2.VideoWriter("../presentation/demo.avi", fourcc, fps, (width, height))

frame_number = 6960
while True:
    #frame = cv2.imread("/home/alex/prog/cv/generated_datasets/carla-new/from_0_camera/frames/frame_" + str(frame_number) + ".jpg")
    frame = cv2.imread("/home/alex/prog/cv/generated_datasets/detr_segm/frames/frame-" + str(frame_number) + ".jpg")
    if frame is None:
        break
    mask = cv2.imread("/home/alex/prog/cv/generated_datasets/detr_segm/masks/frame-" + str(frame_number) + ".png")
    #mask = cv2.imread("/home/alex/prog/cv/generated_datasets/carla-new/from_0_camera/masks/frame_" + str(frame_number) + ".png")
    output_frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0.0)

    video_writer.write(output_frame)
    frame_number += 1