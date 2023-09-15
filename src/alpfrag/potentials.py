#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module containing the potential definitions."""
from typing import Callable
import numpy as np


class Potential:
    def __init__(self,
                 u_of_theta: Callable[[float], float],
                 u_of_theta_der: Callable[[float], float],
                 u_of_theta_dder: Callable[[float], float],
                 ut_of_x: Callable[[float, float], float],
                 ut_of_x_der: Callable[[float, float], float],
                 ut_of_x_dder: Callable[[float, float], float],
                 ) -> None:
        self.u_of_theta = u_of_theta
        self.u_of_theta_der = u_of_theta_der
        self.u_of_theta_dder = u_of_theta_dder
        self.ut_of_x = ut_of_x
        self.ut_of_x_der = ut_of_x_der
        self.ut_of_x_dder = ut_of_x_dder

    def get_u_of_theta(self, theta: float) -> float:
        return self.u_of_theta(theta)

    def get_u_of_theta_der(self, theta: float) -> float:
        return self.u_of_theta_der(theta)

    def get_u_of_theta_dder(self, theta: float) -> float:
        return self.u_of_theta_dder(theta)

    def get_ut_of_x(self, t: float, x: float) -> float:
        return self.ut_of_x(t, x)

    def get_ut_of_x_der(self, t: float, x: float) -> float:
        return self.ut_of_x_der(t, x)

    def get_ut_of_x_dder(self, t: float, x: float) -> float:
        return self.ut_of_x_dder(t, x)


class Free(Potential):
    def __init__(self) -> None:
        def _u_of_theta(theta: float) -> float:
            return 0.5*(theta**2)

        def _u_of_theta_der(theta: float) -> float:
            return theta

        def _u_of_theta_dder(theta: float) -> float:
            return 1.

        def _ut_of_x(t: float, x: float) -> float:
            return 0.5*(x**2)

        def _ut_of_x_der(t: float, x: float) -> float:
            return x

        def _ut_of_x_dder(t: float, x: float) -> float:
            return 1.

        super().__init__(_u_of_theta, _u_of_theta_der, _u_of_theta_dder,
                         _ut_of_x, _ut_of_x_der, _ut_of_x_dder)


class Periodic(Potential):
    def __init__(self) -> None:
        def _u_of_theta(theta: float) -> float:
            return 1. - np.cos(theta)

        def _u_of_theta_der(theta: float) -> float:
            return np.sin(theta)

        def _u_of_theta_dder(theta: float) -> float:
            return np.cos(theta)

        def _ut_of_x(t: float, x: float) -> float:
            return (t**1.5)*(1. - np.cos(x*(t**-0.75)))

        def _ut_of_x_der(t: float, x: float) -> float:
            return (t**0.75)*np.sin(x*(t**-0.75))

        def _ut_of_x_dder(t: float, x: float):
            return np.cos(x*(t**-0.75))

        super().__init__(_u_of_theta, _u_of_theta_der, _u_of_theta_dder,
                         _ut_of_x, _ut_of_x_der, _ut_of_x_dder)


class NonPeriodic(Potential):
    def __init__(self, p: float) -> None:
        self.p = p

        def _u_of_theta(theta: float) -> float:
            return (0.5/p)*((1. + theta**2)**self.p - 1.)

        def _u_of_theta_der(theta: float) -> float:
            return theta*(1. + theta**2)**(self.p - 1.)

        def _u_of_theta_dder(theta: float) -> float:
            return ((1. + (2*self.p - 1.)*(theta**2))
                    * ((1. + theta**2)**(self.p - 2.)))

        def _ut_of_x(t: float, x: float) -> float:
            return (0.5*(t**1.5)/self.p)*((1. + (x**2)/(t**1.5))**self.p - 1.)

        def _ut_of_x_der(t: float, x: float) -> float:
            return x*(1. + (x**2)/(t**1.5))**(self.p - 1.)

        def _ut_of_x_dder(t: float, x: float) -> float:
            return ((1. + (2*self.p - 1.)*(x**2)/(t**1.5))
                    * (1. + (x**2)/(t**1.5))**(self.p - 2.))

        super().__init__(_u_of_theta, _u_of_theta_der, _u_of_theta_dder,
                         _ut_of_x, _ut_of_x_der, _ut_of_x_dder)
