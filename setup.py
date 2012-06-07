#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, Extension
import numpy.distutils.misc_util


desc = open("README.md").read()
with open("requirements.txt") as f:
    required = f.readlines()


setup(
    name="acor",
    version="1.0.1",
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
