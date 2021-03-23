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

from std_msgs.msg import String

class CameraPublisher(Node):

    def __init__(self, source='0'):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(CameraStream, 'camera_stream', 10)
        # serialize image
        self.msg = CameraStream()

        self.cap = cv2.VideoCapture(eval(source) if source.isnumeric() else source)
        assert self.cap.isOpened(), f'Failed to open {source}'
        _, self.img = self.cap.read()  # guarantee first frame

        timer_period = 0  # seconds
        self.timer = self.create_timer(timer_period, self.frame_update)
        self.i = 0

    def frame_update(self):
        # get frame from camera
        if self.cap.isOpened():
            self.cap.grab()
            success, img = self.cap.retrieve()
            if not success:
                self.msg.img = (self.img * 0).ravel().tolist()
            self.msg.height, self.msg.width, self.msg.channel = img.shape
            # ravel() do not need to copy array and it is faster 30.1ms for size(3,640,640)
            self.msg.img = img.reshape(-1).tolist()
            self.msg.pixel_type = 'uint8'
            self.publisher_.publish(self.msg)
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
