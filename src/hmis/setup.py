from glob import glob
from setuptools import find_packages, setup


package_name = 'hmis'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/docs', glob('docs/*.md')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Sanjay',
    maintainer_email='sanjay@lsu.edu',
    description='Human-Machine Input Shaping for low-sway manual gantry control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hmis_node = hmis.hmis_node:main',
            'simulate_hmis = hmis.simulation:main',
            'simulate_precision_stop = hmis.precision_stop_simulation:main',
        ],
    },
)
