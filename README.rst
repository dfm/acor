ACOR
====
This is an updated, maintained fork of the `original acor <https://pypi.org/project/acor/>`_ package. Below is the original README with the appropriate changes to the name and source locations.

This is a direct port of a C++ routine by
`Jonathan Goodman <http://www.math.nyu.edu/faculty/goodman/index.html>`_ (NYU)
called `ACOR <http://www.math.nyu.edu/faculty/goodman/software/acor/>`_ that
estimates the autocorrelation time of time series data very quickly.

`Dan Foreman-Mackey <http://danfm.ca>`_ (NYU) made a few surface changes to
the interface in order to write a Python wrapper (with the permission of the
original author).

Installation
------------

Just run ::

    pip install encor

with ``sudo`` if you really need it.

Otherwise, download the source code
`as a tarball <https://github.com/dfm/acor/tarball/master>`_
or clone the git repository from `GitHub <https://github.com/dfm/acor>`_: ::

    git clone https://github.com/davecwright3/acor.git

Then run ::

    cd acor
    python -m pip install .

to compile and install the module ``acor`` in your Python path. The only
dependency is `NumPy <http://numpy.scipy.org/>`_ (including the
``python-dev`` and ``python-numpy-dev`` packages which you might have to
install separately on some systems).

Usage
-----

Given some time series ``x``, you can estimate the autocorrelation time
(``tau``) using: ::

    import acor
    tau, mean, sigma = acor.acor(x)

References
----------

* http://www.math.nyu.edu/faculty/goodman/software/acor/index.html
* http://www.stat.unc.edu/faculty/cji/Sokal.pdf

