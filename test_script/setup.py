from setuptools import setup
import os
from glob import glob

package_name = 'turtlebot_movement_test'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Basic movement test for TurtleBot 2',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'movement_controller = turtlebot_movement_test.movement_controller:main',
        ],
    },
)