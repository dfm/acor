/*
 * Written by: Dan Foreman-Mackey - danfm@nyu.edu
 */

#include <Python.h>
#include <numpy/arrayobject.h>

#include "acor.h"

/* Docstrings */
static char doc[] = "A module to estimate the autocorrelation time of time-series "\
                    "data extremely quickly.\n";
static char acor_doc[] =
"Estimate the autocorrelation time of a time series\n\n"\
"Parameters\n"\
"----------\n"\
"data : numpy.ndarray (N,) or (M, N)\n"\
"    The time series.\n\n"\
"maxlag : int\n"\
"    N must be greater than maxlag times the estimated autocorrelation\n"\
"    time.\n\n"\
"Returns\n"\
"-------\n"\
"tau : numpy.ndarray (1,) or (M,)\n"\
"    An estimate of the autocorrelation time(s).\n\n"\
"mean : numpy.ndarray (1,) or (M,)\n"\
"    The sample mean(s) of data.\n\n"\
"sigma : numpy.ndarray (1,) or (M,)\n"\
"    An estimate of the standard deviation(s) of the sample mean(s).\n\n";

PyMODINIT_FUNC init_acor(void);
static PyObject *acor_acor(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"acor", acor_acor, METH_VARARGS, acor_doc},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_acor(void)
{
    PyObject *m = Py_InitModule3("_acor", module_methods, doc);
    if (m == NULL)
        return;

    import_array();
}

static PyObject *acor_acor(PyObject *self, PyObject *args)
{
    int i, N, ndim, info, maxlag;
    double *data;

    /* Return value */
    PyObject *ret;
    PyObject *tau_vec = NULL, *mean_vec = NULL, *sigma_vec = NULL;
    npy_intp M[1];
    double *tau, *mean, *sigma;

    /* Parse the input tuple */
    PyObject *data_obj;
    if (!PyArg_ParseTuple(args, "Oi", &data_obj, &maxlag))
        return NULL;

    /* Get the data as a numpy array object */
    PyObject *data_array  = PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    if (data_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "The input data must be a numpy.ndarray.");
        Py_XDECREF(data_array);
        return NULL;
    }

    /* Check the number of dimensions in the input data (must be 1 or 2 only) */
    ndim = (int)PyArray_NDIM(data_array);
    if (ndim > 2 || ndim < 1) {
        PyErr_SetString(PyExc_TypeError, "The input data must be a 1- or 2-D numpy.ndarray.");
        Py_DECREF(data_array);
        return NULL;
    }

    /* Get a pointer to the input data */
    data = (double*)PyArray_DATA(data_array);

    /* N gives the length of the time series */
    N = (int)PyArray_DIM(data_array, ndim-1);

    /* The zeroth (and only) element of M gives the number of series (default to 1) */
    M[0] = 1;
    if (ndim == 2)
        M[0] = (int)PyArray_DIM(data_array, 0);

    /* allocate memory for the output */
    tau_vec   = PyArray_SimpleNew(1, M, PyArray_DOUBLE);
    mean_vec  = PyArray_SimpleNew(1, M, PyArray_DOUBLE);
    sigma_vec = PyArray_SimpleNew(1, M, PyArray_DOUBLE);
    if (tau_vec == NULL || mean_vec == NULL || sigma_vec == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't allocate memory for output.");
        Py_XDECREF(tau_vec);
        Py_XDECREF(mean_vec);
        Py_XDECREF(sigma_vec);
        Py_DECREF(data_array);
        return NULL;
    }

    /* Get a pointers to the output data */
    tau   = (double*)PyArray_DATA(tau_vec);
    mean  = (double*)PyArray_DATA(mean_vec);
    sigma = (double*)PyArray_DATA(sigma_vec);

    for (i = 0; i < M[0]; i++) {
        info = acor(&(mean[i]), &(sigma[i]), &(tau[i]), &(data[i*N]), N, maxlag);
        if (info != 0) {
            if (info == 1)
                PyErr_Format(PyExc_RuntimeError, "The autocorrelation time is too "\
                        "long relative to the variance in dimension %d.", i+1);
            else
                PyErr_SetString(PyExc_RuntimeError, "Couldn't allocate memory for "\
                        "autocovariance vector.");

            Py_DECREF(tau_vec);
            Py_DECREF(mean_vec);
            Py_DECREF(sigma_vec);
            Py_DECREF(data_array);
            return NULL;
        }
    }

    /* clean up */
    Py_DECREF(data_array);

    /* Build the output tuple */
    ret = Py_BuildValue("NNN", tau_vec, mean_vec, sigma_vec);
    if (ret == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't build output tuple.");

        Py_DECREF(tau_vec);
        Py_DECREF(mean_vec);
        Py_DECREF(sigma_vec);
        Py_DECREF(data_array);
        return NULL;
    }

    return ret;
}


