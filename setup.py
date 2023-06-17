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

import numpy


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


include_dirs = [
    "acor",
    numpy.get_include(),
]
acor = Extension("acor._acor", ["acor/_acor.c", "acor/acor.c"],
                 include_dirs=include_dirs)


setup(
    ext_modules=[acor],
)
