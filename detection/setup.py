from setuptools import setup

package_name = 'detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yunfei',
    maintainer_email='moizhaoyunfei@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = detection.publisher_member_function:main',
            'listener = detection.subscriber_member_function:main',
            'camera_publisher = detection.camera_publisher:main',
            'yolo = detection.yolo_detector_subscriber:main',
            'yolo_detector = detection.yolo_detector_subscriber_publisher:main',
            'show = detection.result_subscriber:main'
        ],
    },
)
