from setuptools import setup

package_name = 'payload_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sanjay',
    maintainer_email='sanjay@lsu.edu',
    description='OAK-D payload vision and GPIO encoders',
    license='MIT',
        entry_points={
        'console_scripts': [
            'payload_tracker = payload_perception.payload_tracker:main',
            'encoder_node = payload_perception.encoder_node:main',
            'encoder_serial_node = payload_perception.encoder_serial_node:main',
            'payload_relative_node = payload_perception.payload_relative_node:main',
            'test_publish_payload_state = payload_perception.test_publish_payload_state:main',
            'test_publish_imu_raw = payload_perception.test_publish_imu_raw:main',
            'payload_gantry_frame = payload_perception.payload_gantry_frame:main',
            'phase1_tracker_node = payload_perception.phase1_tracker_node:main',
            'encoder_diagnostics_logger = payload_perception.encoder_diagnostics_logger:main',
        ],
    },
)
