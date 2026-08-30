from glob import glob

from setuptools import setup

package_name = 'meridian_benchmark'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='meridian',
    maintainer_email='meridian@neoul.dev',
    description='Meridian benchmark harness: uHumans2 reader, GT builder, '
                'player/recorder nodes, scorer',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'bench-gt-build = meridian_benchmark.gt_build:main',
            'bench-gt-verify = meridian_benchmark.gt_verify:main',
            'bench-gt-tracklets = meridian_benchmark.gt_tracklets:main',
            'player = meridian_benchmark.player:main',
            'recorder = meridian_benchmark.recorder:main',
            'bench-score = meridian_benchmark.score:main',
            'bench-run = meridian_benchmark.runner:main',
        ],
    },
)
