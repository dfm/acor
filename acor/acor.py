__all__ = ['acor']

import numpy as np

import _acor

def acor(data, maxlag=10):
    """
    Estimate the autocorrelation time of a time series

    Parameters
    ----------
    data : numpy.ndarray (N,) or (M, N)
        The time series.

    maxlag : int, optional
        N must be greater than maxlag times the estimated autocorrelation
        time.

    Returns
    -------
    tau : numpy.ndarray (1,) or (M,)
        An estimate of the autocorrelation time(s).

    mean : numpy.ndarray (1,) or (M,)
        The sample mean(s) of data.

    sigma : numpy.ndarray (1,) or (M,)
        An estimate of the standard deviation(s) of the sample mean(s).

    """
    return _acor.acor(np.array(data), maxlag)

