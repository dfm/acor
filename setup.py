#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
import numpy.distutils.misc_util

setup(name='acor',
        version='1',
        description='Estimate your autocorrelation times very quickly.',
        author='Daniel Foreman-Mackey',
        author_email='danfm@nyu.edu',
        packages=['acor'],
        ext_modules = [Extension('acor._acor', ['acor/_acor.c', 'acor/acor.c'])],
        include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
    )


