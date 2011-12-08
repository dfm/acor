#ACOR

This is a direct port of a C++ routine by
[Jonathan Goodman](http://www.math.nyu.edu/faculty/goodman/index.html) (NYU) called
[ACOR](http://www.math.nyu.edu/faculty/goodman/software/acor/index.html) that
estimates the autocorrelation time of time series data very quickly.

[Dan Foreman-Mackey](http://danfm.ca) (NYU) made a few surface changes to the
interface in order to write a Python wrapper (with the permission of the original
author).

##Installation

To obtain the software, download the source code and documentation [as a
tarball](https://github.com/dfm/acor/tarball/master) or clone the git
repository from [GitHub](https://github.com/dfm/acor):

    git clone https://github.com/dfm/acor.git

Then run

    cd acor
    python setup.py install

to compile and install the module `acor` in your Python path. The only dependency
is [NumPy](http://numpy.scipy.org/) (including the `python-dev` and `python-numpy-dev`
packages which you might have to install separately on some systems).

##Usage

Given some time series `x`, you can estimate the autocorrelation time (`tau`) using:

```python
import acor
tau, mean, sigma = acor.acor(x)
```

##References

* http://www.math.nyu.edu/faculty/goodman/software/acor/index.html
* http://www.stat.unc.edu/faculty/cji/Sokal.pdf

