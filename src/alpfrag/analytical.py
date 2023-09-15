#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module containing analytical expressions obtained via approximations."""
import numpy as np
import sys
from scipy.special import jv, gamma


def standard_bg_before_osc(t: float, mu: complex, u_der_ini: float,
                           theta_ini: float):
    # BUG : This fails in periodic potential with the initial value pi/2.
    r = u_der_ini/(mu**2)
    res = theta_ini - r + r*(2.**0.25)*gamma(1.25)*jv(0.25, mu*t)/(mu*t)**0.25
    # if np.isclose(res.imag, 0., rtol=1e-12, atol=1e-12):
    if abs(res.imag) > 1e-12:
        raise ValueError("Result has non-zero imaginary part. {:e}"
                         .format(res.imag))
    return res.real


def standard_bg_before_osc_der(t:float, mu: complex, u_der_ini: float):
    r = u_der_ini/(mu**2)
    res = -r*(2.**0.25)*gamma(1.25)*mu*jv(1.25, mu*t)/(mu*t)**0.25
    # if np.isclose(res.imag, 0., rtol=1e-12, atol=1e-12):
    if abs(res.imag) > 1e-12:
        raise ValueError("Result has non-zero imaginary part. {:e}"
                         .format(res.imag))
    return res.real
