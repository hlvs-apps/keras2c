# setup.py

from setuptools import setup, find_packages

setup(
    name='keras2c',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'keras2c': ['include/*']},
    include_package_data=True,
    install_requires=[
        # Add dependencies here
    ],
)