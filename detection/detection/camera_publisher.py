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

import cv2

from interfaces.msg import CameraStream


class CameraPublisher(Node):

    def __init__(self, source='0'):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(CameraStream, 'camera_stream', 10)

        self.cap = cv2.VideoCapture(eval(source) if source.isnumeric() else source)
        assert self.cap.isOpened(), f'Failed to open {source}'
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) % 100
        _, self.img = self.cap.read()  # guarantee first frame

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.frame_update)
        self.i = 0

    def frame_update(self):
        # get frame from camera
        if self.cap.isOpened():
            self.cap.grab()
            success, im = self.cap.retrieve()
            self.img = im if success else self.img * 0
            # serialize image
            msg = CameraStream()
            msg.height, msg.width, msg.channel = im.shape
            msg.img = self.img.reshape(-1).tolist()
            msg.pixel_type = 'uint8'
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing frame: {self.i}')
            self.i += 1
        else:
            self.get_logger().info('The camera is not open')


def main(args=None):
    rclpy.init(args=args)

    camera_publisher = CameraPublisher()

    rclpy.spin(camera_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
