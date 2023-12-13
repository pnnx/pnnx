from setuptools import setup, find_packages

import sys

sys.argv.extend(['--plat-name', 'x86_64'])

setup(
    name='pnnx-deps',
    version='0.1',
    packages=find_packages()
)
