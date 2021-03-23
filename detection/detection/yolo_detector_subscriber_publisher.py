# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

import os
import sys

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from interfaces.msg import CameraStream

# change excuting dir and add current directory in path so as to import modules
yolo_path = '/home/yunfei/Desktop/detection_ws/src/detection/detection/yolov5'
os.chdir(yolo_path)
print('Path:', os.getcwd())
sys.path.append(os.path.join(yolo_path))

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

check_requirements()

"""
This is a yolov5 detector node, it takes in the camera data from camera_publisher node and proceed the detection 
and then, it publish the detection.
"""


def webcam_detect(img, imgsz, model, device, stride,
                  augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False):
    """
    A real time webcom dectector based on yolov5 and ros2
    :param img: RGB image
    :param imgsz: size of output image
    :param model: model for prediction
    :param device: device, if not given, it will be  configured by select_device which prioritize GPU
    :param augment: augmented inference
    :param conf_thres: object confidence threshold
    :param iou_thres: IOU threshold for NMS
    :param classes: filter by class: eg.[0,1,2]
    :param agnostic_nms: class-agnostic NMS
    :param save_conf: save confidences in --save-txt labels
    """

    im0 = img.copy()
    # Letterbox
    img = letterbox(img, imgsz, auto=True, stride=stride)[0]
    img = np.expand_dims(img, axis=0)

    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()

    # Process detections
    det = pred[0]  # detections per image

    s = '%gx%g ' % img.shape[2:]  # print string
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            # if view_img:
            # for each detection
            label = f'{names[int(cls)]} {conf:.2f}'
            # plot bounding box on image
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        cv2.imshow('webcam', im0)
        cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')


class YoloDetector(Node):
    def __init__(self, imgsz, model, device, stride):
        super().__init__('yolo_detector')
        self.imgsz = imgsz
        self.model = model
        self.device = device
        self.stride = stride
        self.subscription = self.create_subscription(
            CameraStream,
            'camera_stream',
            self.yolo_callback,
            10)
        self.subscription  # prevent unused variable warning

    def yolo_callback(self, msg):
        # deserialize image
        im_shape = (msg.height, msg.width, msg.channel)
        im = np.frombuffer(msg.img, dtype=msg.pixel_type).reshape(im_shape)
        webcam_detect(im, self.imgsz, self.model, self.device, self.stride)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def main(args=None):
    weights = 'yolov5s.pt'
    imgsz = 640
    device = ''

    # Initialize
    set_logging()
    device = select_device(device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # run in ros
    rclpy.init(args=args)
    yolo_detector = YoloDetector(imgsz, model, device, stride)
    rclpy.spin(yolo_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    yolo_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
