"""Modules for the background evolution."""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable
import alpfrag.symplectic as symplectic


def scaled_initial_conditions(theta_ini: float,
                              theta_der_ini: float,
                              ti: float) -> tuple[float, float]:
    """Calculate the scaled initial conditions.

    This function calculates the initial conditions `x0` and `v0` from the
    unscaled ones.

    Parameters
    ----------
    theta_ini : float
        Initial angle
    theta_der_ini : float
        Initial angle derivative
    ti : float
        Initial simulation time

    Returns
    -------
    tuple[float, float]
        Scaled initial conditions [`x0', `v0']
    """
    try:
        x0 = theta_ini*(ti**0.75)
        v0 = 0.75*x0/ti + (ti**0.75)*theta_der_ini
        return x0, v0
    except ZeroDivisionError:
        msg = 'Initial time needs to be larger than zero!'
        raise ValueError(msg)


def field_evolve_wo_scaling(ti: float,
                            tf: float,
                            theta_ini: float,
                            theta_der_ini: float,
                            u_der: Callable[..., float],
                            u_dder: Callable[..., float] | None = None,
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
                -1.5*y[1]/t - (c**2)*chit(t)*u_der(y[0], *u_args)]

    def jac(t, y, *u_args):
        return [[0., 1.],
                [-(c**2)*chit(t)*u_dder(y[0], *u_args), -1.5/t]]

    # Initial conditions
    y0 = [theta_ini, theta_der_ini]

    # Solution
    if method not in ['RK23', 'RK45', 'DOP853']:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args, jac=jac)
    else:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args)


def field_evolve_w_scaling(ti: float,
                           tf: float,
                           x0: float,
                           v0: float,
                           ut_der: Callable[..., float],
                           ut_dder: Callable[..., float] | None = None,
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

    # Define the Jacobian. It will be used only if the method is implicit.
    def jac(t, y):
        return [[0., 1.],
                [-((c**2)*chit(t)*ut_dder(t, y[0]) + 0.1875/(t**2)), 0.]]

    # Initial conditions
    y0 = [x0, v0]

    # Solution
    if method not in ['RK23', 'RK45', 'DOP853']:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args, jac=jac)
    else:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args)


def unscale_solution(t: np.ndarray, x: np.ndarray,
                     v: np.ndarray) -> dict[str, np.ndarray]:
    """Restore the redshift scaling of the solution.

    This function takes a scaled solution and restores the redshift
    dependence.

    Parameters
    ----------
    t : np.ndarray
        Simulation time data.
    x : np.ndarray
        `x` data.
    v : np.ndarray
        `v` data

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with the following key-value pairs.

    """

    return {
        't': t,
        'theta': x*(t**-0.75),
        'theta_der': (v - 0.75*x/t)*(t**-0.75)
    }


def get_energy_density(t: np.ndarray,
                       theta: np.ndarray,
                       theta_der: np.ndarray,
                       u: Callable[..., float],
                       u_args: tuple = (),
                       chit: Callable[..., float] | None = None,
                       c: float = 1.
                       ) -> np.ndarray:
    if chit is None:
        def chit(t):
            return 1.
    return 0.5*((theta_der/c)**2) + chit(t)*u(theta, *u_args)


def get_com_energy_density(t: np.ndarray,
                           theta: np.ndarray,
                           theta_der: np.ndarray,
                           u: Callable[..., float],
                           u_args: tuple = (),
                           chit: Callable[..., float] | None = None,
                           c: float = 1.
                           ) -> np.ndarray:
    if chit is None:
        def chit(t):
            return 1.
    return (0.5*((theta_der/c)**2) + chit(t)*u(theta, *u_args))*(t**1.5)


def get_pressure_density(t: np.ndarray,
                         theta: np.ndarray,
                         theta_der: np.ndarray,
                         u: Callable[..., float],
                         u_args: tuple = (),
                         chit: Callable[..., float] | None = None,
                         c: float = 1.
                         ) -> np.ndarray:
    if chit is None:
        def chit(t):
            return 1.
    return 0.5*((theta_der/c)**2) - chit(t)*u(theta, *u_args)


def get_com_pressure_density(t: np.ndarray,
                             theta: np.ndarray,
                             theta_der: np.ndarray,
                             u: Callable[..., float],
                             u_args: tuple = (),
                             chit: Callable[..., float] | None = None,
                             c: float = 1.
                             ) -> np.ndarray:
    if chit is None:
        def chit(t):
            return 1.
    return (0.5*((theta_der/c)**2) - chit(t)*u(theta, *u_args))*(t**1.5)


def get_eos(t: np.ndarray,
            theta: np.ndarray,
            theta_der: np.ndarray,
            u: Callable[..., float],
            u_args: tuple = (),
            chit: Callable[..., float] | None = None,
            c: float = 1.
            ) -> np.ndarray:
    if chit is None:
        def chit(t):
            return 1.
    pd = get_com_pressure_density(t, theta, theta_der, u, u_args, chit, c)
    ed = get_com_energy_density(t, theta, theta_der, u, u_args, chit, c)
    return pd/ed
