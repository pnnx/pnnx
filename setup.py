from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True

setup(
    name='pnnx-deps',
    version='0.1',
    packages=find_packages(),
    distclass=BinaryDistribution
)
