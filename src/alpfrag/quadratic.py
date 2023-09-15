#!/usr/bin/env python3
"""Modules for the quadratic potential with constant mass."""
from scipy.special import jv, jvp, gamma
import numpy as np


def normalized_field_amplitude(t: float) -> float:
    r"""Calculate the normalized field amplitude for the quadratic model.

    Calculate the exact result for the field amplitude :math:`\Theta`
    normalized with respect to the initial value :math:`\Theta_i`.

    Parameters
    ----------
    t : float
        Simulation time

    Returns
    -------
    float
        Exact normalized field amplitude for the quadratic model
    """
    return (2.**0.25)*gamma(1.25)*jv(0.25, t)/(t**0.25)


def normalized_field_velocity(t: float) -> float:
    r"""Calculate the normalized field velocity for the quadratic model.

    Calculate the exact result for the field velocity :math:`\Theta'`
    normalized with respect to the initial value :math:`\Theta_i`.

    Parameters
    ----------
    t : float
        Simulation time

    Returns
    -------
    float
        Exact normalized field velocity for the quadratic model
    """
    return (2.**0.25)*gamma(1.25)*(jvp(0.25, t) - 0.25*jv(0.25, t)/t)/(t**0.25)


def normalized_energy_density(t: float) -> float:
    r"""Calculate the normalized energy density for the quadratic model.

    Calculate the exact result for the energy density :math:`\rho/m_0^2 f^2`
    normalized with respect to the initial value :math:`\Theta_i^2`.

    Parameters
    ----------
    t : float
        Simulation time

    Returns
    -------
    float
        Exact normalized energy density for the quadratic model.
    """
    return (gamma(1.25)**2)*(1./np.sqrt(2*t))*(jv(0.25, t)**2 + jv(1.25, t)**2)


def normalized_comoving_energy_density(t: float) -> float:
    r"""Calculate normalized comoving energy density for the quadratic model.

    Calculate the exact result for the comoving energy density
    :math:`\rho/m_0^2 f^2` normalized with respect to the initial
    value :math:`\Theta_i^2`.

    Parameters
    ----------
    t : float
        Simulation time

    Returns
    -------
    float
        Exact normalized energy density for the quadratic model.
    """
    return (gamma(1.25)**2)*(t/np.sqrt(2.0))*(jv(0.25, t)**2 + jv(1.25, t)**2)


def normalized_pressure_density(t: float) -> float:
    r"""Calculate the normalized pressure density for the quadratic model.

    Calculate the exact result for the pressure density :math:`\rho/m_0^2 f^2`
    normalized with respect to the initial value :math:`\Theta_i^2`.

    Parameters
    ----------
    t : float
        Simulation time

    Returns
    -------
    float
        Exact normalized pressure density for the quadratic model.
    """
    return (gamma(1.25)**2)*(1./np.sqrt(2*t))*(jv(1.25, t)**2 - jv(0.25, t)**2)


def normalized_comoving_pressure_density(t: float) -> float:
    r"""Calculate normalized comoving pressure density for the quadratic model.

    Calculate the exact result for the comoving pressure density
    :math:`\rho/m_0^2 f^2` normalized with respect to the initial
    value :math:`\Theta_i^2`.

    Parameters
    ----------
    t : float
        Simulation time

    Returns
    -------
    float
        Exact normalized pressure density for the quadratic model.
    """
    return (gamma(1.25)**2)*(t/np.sqrt(2.0))*(jv(1.25, t)**2 - jv(0.25, t)**2)


def equation_of_state(t: float) -> float:
    return (jv(1.25, t)**2 - jv(0.25, t)**2)/(jv(1.25, t)**2 + jv(0.25, t)**2)
