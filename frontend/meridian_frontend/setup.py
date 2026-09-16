from glob import glob

from setuptools import find_packages, setup

package_name = 'meridian_frontend'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/frontend.launch.py']),
        ('share/' + package_name + '/engines', glob('engines/*.plan')),   # TensorRT 엔진이 패키지와 함께 이동
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jongyoon',
    maintainer_email='qkrwhddbs26@gmail.com',
    description='MERIDIAN Perception Frontend (single-process assembly)',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'frontend = meridian_frontend.frontend:main',
        ],
    },
)
