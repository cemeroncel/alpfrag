"""This module performs symplectic integrations."""
from typing import Callable, Optional
import numpy as np


def euler1_step(x0: float,
                v0: float,
                t0: float,
                h: float,
                F: Callable[[float, float], float]
                ) -> tuple[float, float]:
    """Perform a single symplectic euler step.

    Parameters
    ----------
    x0 : float
        Initial position
    v0 : float
        Initial velocity
    t0 : float
        Initial time
    h : float
        Time step
    F : Callable[[float, float], float]
        Force function. Should have the signature `F(t, x)`.

    Returns
    -------
    tuple[float, float]
        Final position and velocity
    """
    
    x1 = x0 + h*v0
    v1 = v0 + h*F(t0 + h, x1)
    return x1, v1


def verletp2_step(x0: float,
                  v0: float,
                  t0: float,
                  h: float,
                  F: Callable[[float, float], float]
                  ) -> tuple[float, float]:
    """Perform a single position verlet step.

    Parameters
    ----------
    x0 : float
        Initial position
    v0 : float
        Initial velocity
    t0 : float
        Initial time
    h : float
        Time step
    F : Callable[[float, float], float]
        Force function. Should have the signature `F(t, x)`.

    Returns
    -------
    tuple[float, float]
        Final position and velocity
    """
    x1 = x0 + 0.5*h*v0
    v2 = v0 + h*F(t0 + h, x1)
    x2 = x1 + 0.5*h*v2
    return x2, v2


def verletv2_step(x0: float,
                  v0: float,
                  t0: float,
                  h: float,
                  F: Callable[[float, float], float]
                  ) -> tuple[float, float]:
    """Perform a single velocity verlet step.

    Parameters
    ----------
    x0 : float
        Initial position
    v0 : float
        Initial velocity
    t0 : float
        Initial time
    h : float
        Time step
    F : Callable[[float, float], float]
        Force function. Should have the signature `F(t, x)`.

    Returns
    -------
    tuple[float, float]
        Final position and velocity
    """
    v1 = v0 + 0.5*h*F(t0, x0)
    x2 = x0 + h*v1
    v2 = v1 + 0.5*h*F(t0 + 2*h, x2)
    return x2, v2


def _yoshida_coefs_euler1():
    return (1., ), (1., )


def _yoshida_coefs_verletp2():
    return (0.5, 0.5), (1., 0.)


def _yoshida_coefs_verletv2():
    return (0., 1.), (0.5, 0.5)


def _yoshida_coefs_neri4():
    # Define the coefficients
    w0 = -1.7024143839193153    # -2^(1/3)/(2-2^(1/3))
    w1 = 1.3512071919596578     # 1/(2-2^(1/3))
    c1 = 0.5*w1
    c2 = 0.5*(w0 + w1)
    c3 = 0.5*(w0 + w1)
    c4 = 0.5*w1
    d1 = w1
    d2 = w0
    d3 = w1
    d4 = 0.
    return (c1, c2, c3, c4), (d1, d2, d3, d4)


def get_yoshida_coefs(method: str
                      ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if method == 'euler1':
        return _yoshida_coefs_euler1()
    elif method == 'verletp2':
        return _yoshida_coefs_verletp2()
    elif method == 'verletv2':
        return _yoshida_coefs_verletv2()
    elif method == 'neri4':
        return _yoshida_coefs_neri4()
    else:
        raise ValueError("""Unknown method!\n
        Method should be one of the `euler1`, `verletp2`,
        `verletv2` or `neri4`.""")


def _yoshida_step(x0: float,
                  v0: float,
                  t0: float,
                  h: float,
                  c: tuple[float, ...],
                  d: tuple[float, ...],
                  F: Callable[..., float],
                  F_args: tuple = (),
                  ) -> tuple[float, float]:

    # Initialize the position and velocity arrays
    k: int = len(c)
    x = np.zeros(k + 1)
    v = np.zeros(k + 1)
    x[0] = x0
    v[0] = v0

    # Run the loop
    for i in range(1, k + 1):
        x[i] = x[i - 1] + h*c[i - 1]*v[i - 1]

        # Don't perform the force calculation if the coefficient is zero
        if d[i - 1] == 0.:
            v[i] = v[i - 1]
        else:
            v[i] = v[i - 1] + h*d[i - 1]*F(t0 + h*c[i - 1], x[i], *F_args)

    # Return the position and velocity after the step
    return x[-1], v[-1]


def _yoshida_integrate_interval(xi: float,
                                vi: float,
                                ti: float,
                                tf: float,
                                h: float,
                                c: tuple[float, ...],
                                d: tuple[float, ...],
                                F: Callable[..., float],
                                F_args: tuple = (),
                                ) -> tuple[float, float]:
    x, v = xi, vi
    t = ti
    while t < tf:
        x, v = _yoshida_step(x, v, t, h, c, d, F, F_args)
        t = t + h
    return x, v


def yoshida_integrate_store(xi: float,
                            vi: float,
                            ti: float,
                            tf: float,
                            h: float,
                            F: Callable[..., float],
                            F_args: tuple = (),
                            method: str = 'verletv2'):

    # Get the coefficients corresponding to the method
    c, d = get_yoshida_coefs(method)

    # Create the arrays
    t_arr = np.arange(ti, tf, h)
    x_arr = np.zeros(len(t_arr))
    v_arr = np.zeros(len(t_arr))
    
    # Initial conditions
    x_arr[0] = xi
    v_arr[0] = vi
    
    # Main loop
    for i in range(len(t_arr) - 1):
        x_arr[i + 1], v_arr[i + 1] = _yoshida_step(x_arr[i],
                                                   v_arr[i],
                                                   t_arr[i],
                                                   h, c, d, F, F_args)
        
    return t_arr, x_arr, v_arr


def get_yoshida_coefs_vec(method: str):
    if method == 'vv2':
        w = np.ones(1)
    elif method == 'vv4':
        w = np.zeros(3)
        w[0] = 1.351207191959657771818
        w[1] = -1.702414403875838200264
        w[2] = w[0]
    elif method == 'vv6':
        w = np.zeros(7)
        w[0] = 0.78451361047755726382
        w[1] = 0.23557321335935813368
        w[2] = -1.1776799841788710069
        w[3] = 1.3151863206839112189
        w[4] = w[2]
        w[5] = w[1]
        w[6] = w[0]
    elif method == 'vv8':
        w = np.zeros(15)
        w[0] = 0.74167036435061295345
        w[1] = -0.40910082580003159400
        w[2] = 0.19075471029623837995
        w[3] = -0.57386247111608226666
        w[4] = 0.29906418130365592384
        w[5] = 0.33462491824529818378
        w[6] = 0.31529309239676659663
        w[7] = -0.79688793935291635402
        w[8] = w[6]
        w[9] = w[5]
        w[10] = w[4]
        w[11] = w[3]
        w[12] = w[2]
        w[13] = w[1]
        w[14] = w[0]
    elif method == 'vv10':
        w = np.zeros(31)
        w[0] = -0.48159895600253002870
        w[1] = 0.0036303931544595926879
        w[2] = 0.50180317558723140279
        w[3] = 0.28298402624506254868
        w[4] = 0.80702967895372223806
        w[5] = -0.026090580538592205447
        w[6] = -0.87286590146318071547
        w[7] = -0.52373568062510581643
        w[8] = 0.44521844299952789252
        w[9] = 0.18612289547097907887
        w[10] = 0.23137327866438360633
        w[11] = -0.52191036590418628905
        w[12] = 0.74866113714499296793
        w[13] = 0.066736511890604057532
        w[14] = -0.80360324375670830316
        w[15] = 0.91249037635867994571
        w[16] = w[14]
        w[17] = w[13]
        w[18] = w[12]
        w[19] = w[11]
        w[20] = w[10]
        w[21] = w[9]
        w[22] = w[8]
        w[23] = w[7]
        w[24] = w[6]
        w[25] = w[5]
        w[26] = w[4]
        w[27] = w[3]
        w[28] = w[2]
        w[29] = w[1]
        w[30] = w[0]
    else:
        raise ValueError("Available methods are 'vv2', 'vv4', 'vv6', 'vv8', and 'vv10'.")
    return w


