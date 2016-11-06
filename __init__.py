# from ctypes import cdll, c_void_p, c_int, c_bool, c_double, c_float
import ctypes as _ctypes
import numpy as _numpy, os as _os

_libpath = _os.path.dirname(__file__) + '/out/libtsne.so'

_lib=_ctypes.cdll.LoadLibrary(_libpath)

_func=getattr(_lib, 'run_tSNE_float64')
_func.argtypes = [_ctypes.c_void_p, #inputData
                 _ctypes.c_void_p, #outputData
                 _ctypes.c_int, #Nsamples
                 _ctypes.c_int, #in_dims
                 _ctypes.c_int, #out_dims
                 _ctypes.c_double, #theta
                 _ctypes.c_double, #perplexity
                 _ctypes.c_int, #rand_seed
                 _ctypes.c_bool #verbose
                 ]


def run(input, dims=2, theta=0.5, perplexity=50, seed=0, verbose=True):

    input = _numpy.ascontiguousarray(input)
    output = _numpy.zeros((input.shape[0], dims), dtype=input.dtype)
    output = _numpy.ascontiguousarray(output)

    result = _func(input.ctypes.data, output.ctypes.data,
                 input.shape[0], input.shape[1],
                 dims, theta, perplexity,
                 seed, verbose)
    if(result > 0):
        raise RuntimeError("bhtsne library error")

    return output
