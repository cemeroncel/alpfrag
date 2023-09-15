"""Modules for the background evolution. DEPRECATED!"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable
import alpfrag.symplectic as symplectic

def field_evolve_w_scaling_expl(ti: float,
                                tf: float,
                                x0: float,
                                v0: float,
                                ut_der: Callable[..., float],
                                u_args: tuple = (),
                                chit: Callable[..., float] | None = None,
                                c: float = 1.,
                                method: str = "DOP853",
                                dense_output: bool = False,
                                events: Callable[..., float] | None = None,
                                rtol: float = 1e-12,
                                atol: float = 1e-12
                                ):
    if chit is None:
        def chit(t):
            return 1.

    # RHS of the differential equation system to solve
    def rhs(t, y, *u_args):
        return [y[1],
                -((c**2)*chit(t)*ut_der(t, y[0], *u_args)
                  + 0.1875*y[0]/(t**2))]

    # Initial conditions
    y0 = [x0, v0]

    # Solution
    return solve_ivp(rhs, [ti, tf], y0, method=method,
                     dense_output=dense_output, events=events,
                     rtol=rtol, atol=atol, args=u_args)


def field_evolve_w_scaling_symp(ti: float,
                                tf: float,
                                x0: float,
                                v0: float,
                                ut_der: Callable[..., float],
                                u_args: tuple = (),
                                chit: Callable[..., float] | None = None,
                                c: float = 1.,
                                method: str = 'verletp2',
                                h: float = 1e-4):
    if chit is None:
        def chit(t):
            return 1.

    # Force function
    def force_fun(t, x, *u_args):
        return -1.*((c**2)*chit(t)*ut_der(t, x, *u_args)
                    + (3./16.)*x/(t**2))

    # Call the symplectic integrator
    t, x, v = symplectic.yoshida_integrate_store(x0, v0, ti, tf, h, force_fun,
                                                 F_args=u_args,
                                                 method=method)

    # Return the results as a dictionary
    return {
        't': t,
        'x': x,
        'v': v
        }
