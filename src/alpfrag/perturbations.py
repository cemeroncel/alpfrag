#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modules for calculating the evolution of perturbations."""
# TODO: Use oldalpfrag.linearjit as a base
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable
import alpfrag.symplectic as symplectic
from scipy.integrate._ivp.ivp import OdeResult
from scipy.special import sici
import alpfrag.background as bg
from alpfrag.cosmology import Cosmology


def mode_time(tm: float, kt: float) -> float:
    """
    Calculate the mode time tk.

    Parameters
    ----------
    tm : float
    kt : float
        Dimensionless momentum ktilde.

    Returns
    -------
    float
        Mode time tk.

    """
    if tm <= 0:
        raise ValueError("`tm` needs to be larger than zero!")
    if kt <= 0:
        raise ValueError("`kt` needs to be larger than zero!")
    return np.sqrt(4.*tm/3.)*kt


def get_phitk_rad(tk: float) -> tuple[float, float]:
    r"""
    Return the normalized curvature perturbation, and its derivatives.

    This method evaluates the following function and its derivatives.

    .. math:: 3\frac{\sin(t_k) - t_k \cos(t_k)}{t_k^3}

    where :math:`t_k = (k/a)/(\sqrt{3}H)` is the time variable.

    Parameters
    ----------
    tk : float
        time variable

    Returns
    -------
    pert : float
        Normalized curvature perturbation.
    pert_der : float
        First Derivative of the normalized curvature perturbation.
    pert_dder : float
        First Derivative of the normalized curvature perturbation.

    """
    if tk <= 0:
        raise ValueError("`tk` needs to be larger than zero!")
    if tk < 5e-3:
        c0 = 1.0
        c2 = -0.1
        c4 = 280.**-1
        c6 = -15120.**-1
        c8 = 1330560.**-1
        pert = c0 + c2*(tk**2) + c4*(tk**4) + c6*(tk**6) + c8*(tk**8)
        pert_der = 2*c2*tk + 4*c4*(tk**3) + 6*c6*(tk**5) + 8*c8*(tk**7)
        # pert_dder = 2*c2 + 12*c4*(tk**2) + 30*c6*(tk**4) + 56*c8*(tk**6)
    else:
        pert = 3*(np.sin(tk) - tk*np.cos(tk))/(tk**3)
        pert_der = (9*np.cos(tk)/(tk**3)
                    - 9*np.sin(tk)/(tk**4)
                    + 3*np.sin(tk)/(tk**2))
        # pert_dder = (3*np.cos(tk)/(tk**2)
        #              - 15*np.sin(tk)/(tk**3)
        #              - 36*np.cos(tk)/(tk**4)
        #              + 36*np.sin(tk)/(tk**5))
    return pert, pert_der


def get_delta_cdm_rad(tk):
    """
    Return the normalized CDM density constrast in radiation era.

    The result is Newtonian gauge.

    Parameters
    ----------
    tk : float
        Mod time tk.

    Returns
    -------
    float
        Normalized CDM density contrast in radiation era and in
        Newtonian gauge.

    """
    if tk <= 0:
        raise ValueError("`tk` needs to be larger than zero!")
    if tk < 5e-3:
        c0 = 1.5
        c2 = 1.05
        c4 = -0.0294642857142857
        c6 = 0.000496031746031746
        c8 = -5.35488816738817e-6
        delta = c0 + c2*(tk**2) + c4*(tk**4) + c6*(tk**6) + c8*(tk**8)
        delta_der = 2*c2*tk + 4*c4*(tk**3) + 6*c6*(tk**5) + 8*c8*(tk**8)
    else:
        delta = 9.*(np.sin(tk)/tk
                    + np.cos(tk)/(tk**2)
                    - np.sin(tk)/(tk**3)
                    - sici(tk)[1]
                    + np.log(tk)
                    + np.euler_gamma
                    - 0.5)
        delta_der = (9./tk
                     - 18.*np.sin(tk)/(tk**2)
                     - 27.*np.cos(tk)/(tk**3)
                     + 27.*np.sin(tk)/(tk**4)
                     )
    return delta, delta_der


def _bg_sol_span_interval_check(ti: float,
                                tf: float,
                                bg_sol: OdeResult
                                ) -> None:
    bg_sol_span_interval = True
    if not bg_sol.t[0] <= ti <= bg_sol.t[-1]:
        bg_sol_span_interval = False
    if not bg_sol.t[0] <= tf <= bg_sol.t[-1]:
        bg_sol_span_interval = False
    if not bg_sol_span_interval:
        msg = "The background solution does not cover the interval [ti, tf]"
        raise ValueError(msg)


def mode_evolve_wo_scaling(kt: float,
                           mode_ini: float,
                           mode_der_ini: float,
                           bg_sol: OdeResult,
                           u_der: Callable[..., float],
                           u_dder: Callable[..., float],
                           ti: float | None = None,
                           tf: float | None = None,
                           u_args: tuple = (),
                           chit: Callable[..., float] | None = None,
                           c: float = 1.,
                           method: str = "DOP853",
                           dense_output: bool = False,
                           events: Callable[..., float] | None = None,
                           rtol: float = 1e-12,
                           atol: float = 1e-12,
                           t_eval: list[float] | None = None
                           ):
    if chit is not None:
        msg = "Temperature-dependent potential is not yet implemented!"
        raise NotImplementedError(msg)

    # If ti and tf are not given, get them from the background solution.
    if ti is None:
        ti = bg_sol.t[0]
    if tf is None:
        tf = bg_sol.t[-1]

    # If both ti and tf are given, make sure that the background solution
    # covers the requested time range
    if (ti is not None) and (tf is not None):
        _bg_sol_span_interval_check(ti, tf, bg_sol)

    def rhs(t, y, *u_args):
        # Convert tm to tk
        tk = mode_time(t, kt)

        # Curvature perturbations Phi(tk)
        phitk, phitk_der = get_phitk_rad(tk)

        # Background quantities for the ALP field
        try:
            theta, theta_der, = bg_sol.sol(t)
        except TypeError:
            msg = "`bg_sol` should be obtained with dense_output=True."
            raise AttributeError(msg)

        # Adiabatic sourcing term
        s = (2.*phitk*(c**2)*u_der(theta, *u_args)
             - 2.*(tk/t)*phitk_der*theta_der)

        return [y[1],
                (s
                 - 1.5*y[1]/t
                 - ((kt**2)/t + (c**2)*u_dder(theta, *u_args))*y[0])]

    def jac(t, y, *u_args):
        # Convert tm to tk
        tk = mode_time(t, kt)

        # Curvature perturbations Phi(tk)
        phitk, phitk_der = get_phitk_rad(tk)

        # Background quantities for the ALP field
        try:
            theta, theta_der, = bg_sol.sol(t)
        except TypeError:
            msg = "`bg_sol` should be obtained with dense_output=True."
            raise AttributeError(msg)

        return [[0., 1.],
                [-((kt**2)/t + (c**2)*u_dder(theta, *u_args)), -1.5/t]]

    y0 = [mode_ini, mode_der_ini]

    # Solution
    if method not in ['RK23', 'RK45', 'DOP853']:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args, jac=jac,
                         t_eval=t_eval)
    else:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args,
                         t_eval=t_eval)


def mode_evolve_w_scaling(kt: float,
                          x0: float,
                          v0: float,
                          bg_sol: OdeResult,
                          ut_der: Callable[..., float],
                          ut_dder: Callable[..., float],
                          ti: float | None = None,
                          tf: float | None = None,
                          u_args: tuple = (),
                          chit: Callable[..., float] | None = None,
                          c: float = 1.,
                          method: str = "DOP853",
                          dense_output: bool = False,
                          events: Callable[..., float] | None = None,
                          rtol: float = 1e-12,
                          atol: float = 1e-12,
                          t_eval: list[float] | None = None
                          ):
    if chit is not None:
        msg = "Temperature-dependent potential is not yet implemented!"
        raise NotImplementedError(msg)

    # If ti and tf are not given, get them from the background solution.
    if ti is None:
        ti = bg_sol.t[0]
    if tf is None:
        tf = bg_sol.t[-1]

    # If both ti and tf are given, make sure that the background solution
    # covers the requested time range
    if (ti is not None) and (tf is not None):
        _bg_sol_span_interval_check(ti, tf, bg_sol)

    def rhs(t, y, *u_args):
        # Convert tm to tk
        tk = mode_time(t, kt)

        # Curvature perturbations Phi(tk)
        phitk, phitk_der = get_phitk_rad(tk)

        # Background quantities for the ALP field
        try:
            x_bg, v_bg, = bg_sol.sol(t)
        except TypeError:
            msg = "`bg_sol` should be obtained with dense_output=True."
            raise AttributeError(msg)

        # Frequency term
        freq = (kt**2)/t + (c**2)*ut_dder(t, x_bg) + 0.1875/(t**2)

        return [y[1],
                (2*phitk*(c**2)*ut_der(t, x_bg)
                - 2*(tk/t)*phitk_der*(v_bg - 0.75*x_bg/t)
                - freq*y[0])]

    def jac(t, y, *u_args):
        # Convert tm to tk
        tk = mode_time(t, kt)

        # Curvature perturbations Phi(tk)
        phitk, phitk_der = get_phitk_rad(tk)

        # Background quantities for the ALP field
        try:
            x_bg, v_bg, = bg_sol.sol(t)
        except TypeError:
            msg = "`bg_sol` should be obtained with dense_output=True."
            raise AttributeError(msg)

        # Frequency term
        freq = (kt**2)/t + (c**2)*ut_dder(t, x_bg) + 0.1875/(t**2)

        return [[0., 1.],
                [-freq, 0.]]

    y0 = [x0, v0]

    # Solution
    if method not in ['RK23', 'RK45', 'DOP853']:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args, jac=jac,
                         t_eval=t_eval)
    else:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args,
                         t_eval=t_eval)


def dc_evolve_wo_scaling_expl(kt: float,
                              dc_ini: float,
                              dc_der_ini: float,
                              bg_sol: OdeResult,
                              u: Callable[..., float],
                              u_der: Callable[..., float],
                              u_dder: Callable[..., float],
                              ti: float | None = None,
                              tf: float | None = None,
                              u_args: tuple = (),
                              chit: Callable[..., float] | None = None,
                              c: float = 1.,
                              method: str = "DOP853",
                              dense_output: bool = False,
                              events: Callable[..., float] | None = None,
                              rtol: float = 1e-12,
                              atol: float = 1e-12
                              ):
    if chit is not None:
        msg = "Temperature-dependent potential is not yet implemented!"
        raise NotImplementedError(msg)

    # If ti and tf are not given, get them from the background solution.
    if ti is None:
        ti = bg_sol.t[0]
    if tf is None:
        tf = bg_sol.t[-1]

    # If both ti and tf are given, make sure that the background solution
    # covers the requested time range
    if (ti is not None) and (tf is not None):
        _bg_sol_span_interval_check(ti, tf, bg_sol)

    def rhs(t, y, *u_args):
        # Convert tm to tk
        tk = mode_time(t, kt)

        # Curvature perturbations Phi(tk)
        phitk, phitk_der = get_phitk_rad(tk)

        # Background quantities for the ALP field
        try:
            theta, theta_der, = bg_sol.sol(t)
        except TypeError:
            msg = "`bg_sol` should be obtained with dense_output=True."
            raise AttributeError(msg)

        # Get the equation of state, and calculate its derivative
        w = bg.get_eos(t, theta, theta_der, u, u_args=u_args, chit=chit, c=c)
        ed = bg.get_energy_density(t, theta, theta_der, u, u_args=u_args,
                                   chit=chit, c=c)
        w_der = -2.*u_der(theta, *u_args)*theta_der/ed - 1.5*(1. - w**2)/t

        # Calculate the adiabatic sound speed squared
        c_ad_sq = w - (2./3.)*w_der*t/(1. + w)

        # RHS of the differential equation
        return [
            (-kt*y[1]/np.sqrt(t)
             - 1.5*(1. + w)*(tk/t)*phitk_der
             - 1.5*(1. - w)*y[0]/t
             - 2.25*(1. - c_ad_sq)*y[1]*np.sqrt(t)/(kt*(t**2))),
            (y[1]/t
             + kt*y[0]/np.sqrt(t)
             + 1.5*(w - c_ad_sq)*y[1]/t
             - (1. + w)*phitk*kt/np.sqrt(t))
        ]

    def jac(t, y, *u_args):
        # Convert tm to tk
        tk = mode_time(t, kt)

        # Curvature perturbations Phi(tk)
        phitk, phitk_der = get_phitk_rad(tk)

        # Background quantities for the ALP field
        try:
            theta, theta_der, = bg_sol.sol(t)
        except TypeError:
            msg = "`bg_sol` should be obtained with dense_output=True."
            raise AttributeError(msg)

        # Get the equation of state, and calculate its derivative
        w = bg.get_eos(t, theta, theta_der, u, u_args=u_args, chit=chit, c=c)
        ed = bg.get_energy_density(t, theta, theta_der, u, u_args=u_args,
                                   chit=chit, c=c)
        w_der = -2.*u_der(theta, *u_args)*theta_der/ed - 1.5*(1. - w**2)/t

        # Calculate the adiabatic sound speed squared
        c_ad_sq = w - (2./3.)*w_der*t/(1. + w)

        return [
            [-1.5*(1. - w)/t,
             -kt/np.sqrt(t) - 2.25*(1. - c_ad_sq)*np.sqrt(t)/(kt*(t**2))],
            [kt/np.sqrt(t), 1./t + 1.5*(w - c_ad_sq)/t]
        ]

    y0 = [dc_ini, dc_der_ini]

    # Solution
    if method not in ['RK23', 'RK45', 'DOP853']:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args, jac=jac)
    else:
        return solve_ivp(rhs, [ti, tf], y0, method=method,
                         dense_output=dense_output, events=events,
                         rtol=rtol, atol=atol, args=u_args)


def mode_evolve_w_scaling_symp(kt: float,
                               ti: float,
                               tf: float,
                               x0: float,
                               v0: float,
                               ut_der: Callable[..., float],
                               ut_dder: Callable[..., float],
                               x_bg: Callable[float, float],
                               v_bg: Callable[float, float],
                               chit: Callable[..., float] | None = None,
                               c: float = 1.,
                               h: float = 1e-4,
                               method: str = 'verletv2'
                               ):
    if chit is not None:
        msg = "Temperature-dependent potential is not yet implemented!"
        raise NotImplementedError(msg)

    # Get scaled initial conditions
    x0, v0 = bg.scaled_initial_conditions(x0, v0, ti)

    def force_fun(t, x):
        # Convert tm to tk
        tk = mode_time(t, kt)

        # Curvature perturbations Phi(tk)
        phitk, phitk_der = get_phitk_rad(tk)

        # Frequency term
        freq = (kt**2)/t + (c**2)*ut_dder(t, x_bg(t)) + 0.1875/(t**2)

        return (2*phitk*(c**2)*ut_der(t, x_bg(t))
                - 2*(tk/t)*phitk_der*(v_bg(t) - 0.75*x_bg(t)/t)
                - freq*x)

    # Call the symplectic integrator
    t, x, v = symplectic.yoshida_integrate_store(x0, v0, ti, tf, h, force_fun,
                                                 method=method)

    # Return the results as a dictionary
    return {
        't': t,
        'x': x,
        'v': v
        }


def mode_density_perturbation(kt: float,
                              tm: float,
                              mode: float,
                              mode_der: float,
                              theta: float,
                              theta_der: float,
                              u_der: Callable[..., float],
                              u_args: tuple = (),
                              chit: Callable[..., float] | None = None,
                              c: float = 1.
                              ) -> float:
    if chit is not None:
        msg = "Temperature-dependent potential is not yet implemented!"
        raise NotImplementedError(msg)

    tk = mode_time(tm, kt)
    phik = get_phitk_rad(tk)[0]

    return ((theta_der/c)*(mode_der/c)
            + phik*(theta_der/c)**2
            + u_der(theta, *u_args)*mode)


def mode_pressure_perturbation(kt: float,
                               tm: float,
                               mode: float,
                               mode_der: float,
                               theta: float,
                               theta_der: float,
                               u_der: Callable[..., float],
                               u_args: tuple = (),
                               chit: Callable[..., float] | None = None,
                               c: float = 1.
                               ) -> float:
    if chit is not None:
        msg = "Temperature-dependent potential is not yet implemented!"
        raise NotImplementedError(msg)

    tk = mode_time(tm, kt)
    phik = get_phitk_rad(tk)[0]

    return ((theta_der/c)*(mode_der/c)
            + phik*(theta_der/c)**2
            - u_der(theta, *u_args)*mode)


def mode_dc_unscaled(kt: float,
                     tm: float,
                     mode: float,
                     mode_der: float,
                     theta: float,
                     theta_der: float,
                     u: Callable[..., float],
                     u_der: Callable[..., float],
                     u_args: tuple = (),
                     chit: Callable[..., float] | None = None,
                     c: float = 1.
                     ) -> float:
    dp = mode_density_perturbation(kt, tm, mode, mode_der, theta,
                                   theta_der, u_der, u_args=u_args,
                                   chit=chit, c=c)

    ed = bg.get_energy_density(tm, theta, theta_der, u,
                               u_args=u_args, chit=chit, c=c)
    return dp/ed


def mode_dc_scaled(kt: float,
                   t: float,
                   x_pt: float,
                   v_pt: float,
                   x_bg: float,
                   v_bg: float,
                   ut: Callable[..., float],
                   ut_x: Callable[..., float],
                   u_args: tuple = (),
                   chit: Callable[..., float] | None = None,
                   c: float = 1.
                   ) -> float:
    if chit is not None:
        msg = "Temperature-dependent potential is not yet implemented!"
        raise NotImplementedError(msg)

    tk = mode_time(t, kt)
    phik = get_phitk_rad(tk)[0]

    num = ((v_bg - 0.75*x_bg/t)*(v_pt - 0.75*x_pt/t)/(c**2)
           + phik*((v_bg - 0.75*x_bg/t)/c)**2
           + ut_x(t, x_bg, *u_args)*x_pt)

    den = 0.5*((v_bg - 0.75*x_bg/t)/c)**2 + ut(t, x_bg, *u_args)
    return num/den


def dc_eval_latetime(kt: float, delta_ini: float, delta_der_ini: float,
                     cosmo: Cosmology, z_start: float, z_end: float = 0.,
                     rtol: float = 1e-8, atol: float = 1e-8,
                     **solve_ivp_kwargs):
    """
    Solve the evolution of the linear density contrast using WKB approximation.

    Parameters
    ----------
    kt : float
        Dimensionless momentum kt.
    delta_ini : float
        Initial value of the density contrast.
    delta_der_ini : float
        Initial value of the derivative of the density contrast with respect
        to log(y) where y = a/a_eq = (1+z_eq)/(1+z)
    cosmo: alpfrag.cosmology.Cosmology
        Cosmology that will be used to calculate z_eq, etc...
    z_start : float
        Redshift at the start of the simulation.
    z_end : float, optional
        Redshift at the end of the simulation. The default is 0., i.e. today.

    Returns
    -------
    solve_ivp
        Solution.

    """
    def rhs(t, y):
        return [y[1],
                (-0.5*(np.exp(t)/(1. + np.exp(t)))*y[1]
                 - (kt**4 - 1.5*np.exp(t))*y[0]/(1. + np.exp(t)))]
    y0 = [delta_ini, delta_der_ini]
    t_span = [np.log(1. + cosmo.zeq)/np.log(1. + z_start),
              np.log(1. + cosmo.zeq)/np.log(1. + z_end)]
    return solve_ivp(rhs, t_span, y0, rtol=rtol, atol=atol, **solve_ivp_kwargs)



















