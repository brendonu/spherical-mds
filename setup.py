"""
Program structure modeled from https://github.com/jxz12/s_gd2
"""

from platform import python_version
from setuptools import setup, find_packages 
from setuptools.extension import Extension 

import sys 
import os

numpy_version = "numpy>=1.16"

setup_requires = [
    numpy_version,
    "Cython"
]

install_requires = [
    numpy_version
]

cpp_mods = Extension(
    name="cpp_mod",
    headers=["./smds/sgd.hpp"],
    sources=["./smds/sgd.cpp"],
    include_dirs=[get_numpy_include()]
)


setup(
    name="smds",
    version="0.6",
    author="Jacob Miller",
    author_email="jacobmiller1@arizona.edu",
    url="",
    description="Performing MDS onto the sphere via SGD",
    setup_requires=setup_requires,
    install_requires=install_requires,
    packages=find_packages(),
    ext_modules=[cpp_mods],
)