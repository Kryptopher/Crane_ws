from setuptools import setup

package_name = 'experiment_logger'

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
    description='Experiment CSV logging for crane stack',
    license='MIT',
    entry_points={
        'console_scripts': [
            'logger_node = experiment_logger.logger_node:main',
        ],
    },
)
