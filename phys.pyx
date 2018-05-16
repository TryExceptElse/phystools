# cython: infer_types=True, cdivision=True, cmath=True, initializedcheck=False, boundscheck=False, wraparound=False

cimport scipy.special.cython_special as special
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sin, pi
import itertools as itr


cpdef double airy(
        theta: double, lambda_: double, a: double, i0: double=1
) nogil except -1.0:
    """
    Finds intensity of airy disk at position in focal plane with
    passed theta.
    :param θ: Angle between the axis of the circular aperture and the
                line between aperture center and observation point.
    :param λ: Wavelength.
    :param a: Aperture radius.
    :param i0: Maximum intensity, as received at center of airy disk.
    :return: i(θ) Intensity at position with passed θ.
    """
    if not theta >= 0:
        return -1
    if theta == 0:
        return i0
    cdef double k, ka_sin_theta, i
    k = 2 * pi / lambda_  # Find wave-number
    ka_sin_theta = k * a * sin(theta)
    i = i0 * (2 * special.j1(ka_sin_theta) / ka_sin_theta) ** 2
    return i


cpdef np.ndarray airy_arr(
        lambda_: double,
        a: double,
        i0: double=1,
        arr: np.ndarray=None,
        view_theta: float=0.00002):
    if arr is None:
        arr = np.ndarray((512, 512))
    cdef int img_width = arr.shape[0]

    cdef int x, y
    cdef double [:, :] arr_view = arr

    for x in range(img_width):
        for y  in range(img_width):
            pos_theta: double = find_theta(x, y, img_width, view_theta)
            i: double = airy(pos_theta, lambda_, a) * i0
            arr_view[x][y] = i

    return arr


cdef inline double find_theta(
        double x,
        double y,
        int img_width,
        double view_theta) except -1.0:
    rel_x: double = x / <double>img_width - 0.5
    rel_y: double = y / <double>img_width - 0.5
    rel_theta_x: double = rel_x * view_theta
    rel_theta_y: double = rel_y * view_theta
    theta_: double = sqrt(rel_theta_x ** 2 + rel_theta_y ** 2)
    return theta_
