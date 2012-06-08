#!/usr/bin/env python
# encoding: utf-8

import sys
import os

try:
    from setuptools import setup, Extension
    setup, Extension
except ImportError:
    from distutils.core import setup, Extension
    setup, Extension

import numpy.distutils.misc_util


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


desc = open("README.rst").read()
required = ["numpy"]


setup(
    name="acor",
    version="1.0.2",
    author="Daniel Foreman-Mackey and Jonathan Goodman",
    author_email="danfm@nyu.edu",
    packages=["acor"],
    url="http://github.com/dfm/acor/",
    license="MIT",
    description="Estimate the autocorrelation time of a time series quickly.",
    long_description=desc,
    install_requires=required,
    ext_modules=[Extension("acor._acor", ["acor/_acor.c", "acor/acor.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
