#cython: language_level=3
cimport cython
cimport numpy as np
import numpy as np
from scipy.stats import skew, kurtosis

""""
Cython:
=======

Types:
https://stackoverflow.com/a/46416257

NumPy dtype          Numpy Cython type         C Cython type identifier

np.bool_             None                      None
np.int_              cnp.int_t                 long
np.intc              None                      int       
np.intp              cnp.intp_t                ssize_t
np.int8              cnp.int8_t                signed char
np.int16             cnp.int16_t               signed short
np.int32             cnp.int32_t               signed int
np.int64             cnp.int64_t               signed long long
np.uint8             cnp.uint8_t               unsigned char
np.uint16            cnp.uint16_t              unsigned short
np.uint32            cnp.uint32_t              unsigned int
np.uint64            cnp.uint64_t              unsigned long
np.float_            cnp.float64_t             double
np.float32           cnp.float32_t             float
np.float64           cnp.float64_t             double
np.complex_          cnp.complex128_t          double complex
np.complex64         cnp.complex64_t           float complex
np.complex128        cnp.complex128_t          double complex

"""


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] swfe(np.uint64_t windowsize,
                                             np.uint64_t total, 
                                             np.uint64_t step,
                                             np.ndarray[np.float64_t, ndim=2] inarray):
    """
    Sliding windows and simple feature extraction in a single function.

    NUMPY
    In single precision, std() can be inaccurate.
    https://numpy.org/doc/stable/reference/generated/numpy.std.html

    BUT
    Using float64 will overflow Scikit-Learn.

    SO
    Calculate on float64 and return float64, and cast type AFTER.

    Use ddof=1 so NUMPY calculates the same as Pandas (sample std and var).
    """

    cdef np.uint64_t nrow = inarray.shape[0]
    cdef np.uint64_t ocol = inarray.shape[1]
    cdef np.uint64_t nfeat = 6
    total = min(total, max((nrow - windowsize) // step, 1))
    cdef np.ndarray[np.float64_t, ndim=2] out = np.zeros(
       [total, ocol * nfeat], dtype=np.float64
    )
    cdef np.ndarray[np.float64_t, ndim=2] window
    cdef np.ndarray[np.float64_t, ndim=2] reduce = np.zeros([ocol, nfeat], dtype=np.float64)
    cdef Py_ssize_t a, b, c, i, j

    for i in range(total):
        a = i * step
        b = a + windowsize
        window = inarray[a:b, :]
        if window.shape[0] == 0:
            print('empty array', window.shape[0], inarray.shape[0])
            break
        
        try:
            reduce[:, 0] = np.max(window, axis=0)
            reduce[:, 3] = np.min(window, axis=0)
            reduce[:, 1] = np.mean(window, axis=0)
            reduce[:, 2] = np.median(window, axis=0)
            reduce[:, 5] = np.var(window, axis=0, ddof=1)
            reduce[:, 4] = np.std(window, axis=0, ddof=1)

        except Exception as e:
            print(e)

        out[i, :] = np.ravel(reduce, order='C')

    return out


cpdef np.ndarray[np.float64_t, ndim=2] sampleentropy2d(np.ndarray[np.float64_t, ndim=2] data, int m, np.ndarray[np.float64_t, ndim=1] r):
    """

    Input: 
    L: Time series
    m: Template length
    r: Tolerance level

    https://en.wikipedia.org/wiki/Sample_entropy

    """
    cdef Py_ssize_t N = data.shape[0]
    cdef Py_ssize_t M = data.shape[1]
    cdef Py_ssize_t c
    cdef np.float64_t A = 0.0
    cdef np.float64_t B = 0.0
    cdef np.ndarray[np.float64_t, ndim=2] array = np.zeros((1, M), dtype=np.float64)

    for c in range(M):

        # Split time series and save all templates of length m
        xmi = np.array([data[i:i + m, c] for i in range(N - m)])
        xmj = np.array([data[i:i + m, c] for i in range(N - m + 1)])

        # Save all matches minus the self-match, compute B
        B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r[c]) - 1 for xmii in xmi])

        # Similar for computing A
        m += 1
        xm = np.array([data[i:i + m, c] for i in range(N - m + 1)])

        A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r[c]) - 1 for xmi in xm])

        array[0, c] = -np.log(A / B)

    return array

