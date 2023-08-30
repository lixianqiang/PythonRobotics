from setuptools import setup
from glob import glob
import os

package_name = 'obca_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stephen_li',
    maintainer_email='lxq243808918@gmail.com',
    description='OBCA Planner',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obca_planner_node = obca_planner.obca_planner_node:main'
        ],
    },
)
