from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'indian_poker_ws'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ───── 추가: launch 파일 설치 ─────
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # ───── 추가: 모델/리소스 설치(예: tflite) ─────
        ('share/' + package_name + '/models', glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='sktlgud0117@gmial.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_card_ros = indian_poker_ws.gesture_card_ros:main',
            'green_detect_depth = indian_poker_ws.green_detect_depth:main',
            'g_c_ros_action = indian_poker_ws.g_c_ros_action:main',
            'chips_action = indian_poker_ws.chips_action:main',
            
        ],
    },
)
