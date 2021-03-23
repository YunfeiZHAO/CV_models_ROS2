import cv2
import time

cap = cv2.VideoCapture(0)
_, img = cap.read()
while True:
    start = time.time()
    for i in range(120):
        if cap.isOpened():
            cap.grab()
            success, im = cap.retrieve()
            if success:
                cv2.imshow('webcam', im)
                cv2.waitKey(1)
    duration = time.time() - start
    fps = 120/duration
    print(f'fps: {fps}')