from ultralytics import YOLO
import cv2

model = YOLO('model/best.pt')

results = model(source = 'video/video1.mp4', conf = 0.2, save = True)
