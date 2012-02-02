#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
import numpy.distutils.misc_util

long_description = """
ACOR
====

This is a direct port of a C++ routine by
`Jonathan Goodman <http://www.math.nyu.edu/faculty/goodman/index.html>`_
(NYU) called
`ACOR <http://www.math.nyu.edu/faculty/goodman/software/acor/index.html>`_
that estimates the autocorrelation time of time series data very quickly.

`Dan Foreman-Mackey <http://danfm.ca>`_ (NYU) made a few surface changes to the
interface in order to write a Python wrapper (with the permission of the original
author).

Installation
------------

Just run ``pip install acor`` with the optional ``sudo`` if you need it. NumPy
and the associated ``dev`` headers are needed.

Usage
-----

Given some time series ``x``, you can estimate the autocorrelation time
(``tau``) using::

import acor
tau, mean, sigma = acor.acor(x)

References
----------

* http://www.math.nyu.edu/faculty/goodman/software/acor/index.html
* http://www.stat.unc.edu/faculty/cji/Sokal.pdf

"""

setup(
    name="acor",
    version="1.0.0",
    author="Daniel Foreman-Mackey and Jonathan Goodman",
    author_email="danfm@nyu.edu",
    packages=["acor"],
    url="http://github.com/dfm/acor/",
    license="MIT",
    description="Estimate the autocorrelation time of a time series very quickly.",
    long_description=long_description,
    install_requires=["numpy"],
    ext_modules = [Extension('acor._acor', ['acor/_acor.c', 'acor/acor.c'])],
    include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)

