from setuptools import setup, find_packages

setup(
    name='pnnx-deps',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        setuptools.Extension(
            name='deps',
            sources=[]
        )
    ]
)
