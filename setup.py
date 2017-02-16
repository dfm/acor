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



if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


desc = open("README.rst").read()
required = ["numpy"]


def get_extensions():
    # Numpy should not be imported with some of the setup.py commands,
    # especially egg_info. See
    # https://github.com/pypa/pip/issues/25
    # for more information.
    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:]
                               or sys.argv[1] in ('--help-commands', 'egg_info',
                                                  '--version', '--clean')):
        return []
    else:
        import numpy
        include_dirs = [
            "acor",
            numpy.get_include(),
        ]
        acor = Extension("acor._acor", ["acor/_acor.c", "acor/acor.c"],
                         include_dirs=include_dirs)
        return [acor]

setup(
    name="acor",
    version="1.1.1",
    author="Daniel Foreman-Mackey and Jonathan Goodman",
    author_email="danfm@nyu.edu",
    packages=["acor"],
    url="http://github.com/dfm/acor",
    license="MIT",
    description="Estimate the autocorrelation time of a time series quickly.",
    long_description=desc,
    install_requires=required,
    ext_modules=get_extensions(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
