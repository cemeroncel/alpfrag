#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes for various ALP models."""
from astropy.units import Quantity, UnitTypeError
from astropy.units.core import Unit, UnitConversionError
from astropy.cosmology import FlatLambdaCDM, Planck18
from alpfrag.cosmology import Cosmology
import natpy as nat
from typing import Callable
from scipy.optimize import root
from scipy.signal import argrelmax
import numpy as np
from functools import cached_property
import alpfrag.background as bg
from alpfrag.potentials import Potential
from scipy.integrate._ivp.ivp import OdeResult
import alpfrag.analytical as analytical
import alpfrag.perturbations as pt
from abc import ABC, abstractmethod


def _unit_consistent(x: Quantity, u: Unit) -> bool:
    try:
        nat.convert(x, u)
        return True
    except UnitConversionError:
        return False


class ALP(ABC):
    def __init__(self,
                 pot: Potential,
                 m0: Quantity | None = None,
                 cosmo: Cosmology = Cosmology(),
                 c: float = 1.
                 ) -> None:
        self.pot = pot
        if m0 is not None:
            if _unit_consistent(m0, nat.GeV):
                self.m0 = m0
            else:
                msg = f"The unit {m0.unit} for `m0` is inconsistent!"
                raise UnitTypeError(msg)
        else:
            self.m0 = None
        self.cosmo = cosmo
        self.c = c

    def convert_tm_to_redshift(self, tm):
        # Check whether m0 is defined
        if self.m0 is None:
            msg = "You need to set the ALP mass to use this method!"
            raise AttributeError(msg)

        # Convert m0 to GeV and work with its value
        _m0 = nat.convert(self.m0, nat.GeV).value

        # Get the desired Hubble
        _H = 0.5*_m0/(self.c*tm)

        # Get the temperature
        _T = self.cosmo.inv_H_at_T_in_rad(_H)

        # Get the redshift
        return self.cosmo.z_at_T(_T)

    @property
    def H1(self):
        return 0.5*self.m0/self.c

    @cached_property
    def T1(self):
        return self.cosmo.inv_H_at_T_in_rad(self.H1, self.H1.unit)

    @cached_property
    def z1(self):
        return self.cosmo.z_at_T(self.T1)

    def bg_field_evolve_wo_scaling(self,
                                   ti: float,
                                   tf: float,
                                   theta_ini: float,
                                   theta_der_ini: float,
                                   c: float | None = None,
                                   method: str = 'DOP853',
                                   dense_output: bool = False,
                                   events: Callable[..., float] | None = None,
                                   rtol: float = 1e-12,
                                   atol: float = 1e-12
                                   ):
        if c is None:
            c = self.c
        return bg.field_evolve_wo_scaling(ti, tf, theta_ini, theta_der_ini,
                                          self.pot.get_u_of_theta_der,
                                          self.pot.get_u_of_theta_dder,
                                          c=c, method=method,
                                          dense_output=dense_output,
                                          events=events,
                                          rtol=rtol, atol=atol)

    def bg_field_evolve_w_scaling(self,
                                  ti: float,
                                  tf: float,
                                  x0: float,
                                  v0: float,
                                  c: float | None = None,
                                  method: str = 'DOP853',
                                  dense_output: bool = False,
                                  events: Callable[..., float] | None = None,
                                  rtol: float = 1e-12,
                                  atol: float = 1e-12
                                  ):
        if c is None:
            c = self.c
        return bg.field_evolve_w_scaling(ti, tf, x0, v0,
                                         self.pot.ut_of_x_der,
                                         self.pot.get_ut_of_x_dder,
                                         c=c, method=method,
                                         dense_output=dense_output,
                                         events=events,
                                         rtol=rtol, atol=atol)

    def pt_mode_evolve_wo_scaling(self,
                                  kt: float,
                                  mode_ini: float,
                                  mode_der_ini: float,
                                  bg_sol: OdeResult,
                                  ti: float | None = None,
                                  tf: float | None = None,
                                  c: float | None = None,
                                  method: str = "DOP853",
                                  dense_output: bool = False,
                                  events: Callable[..., float] | None = None,
                                  rtol: float = 1e-12,
                                  atol: float = 1e-12,
                                  t_eval: list[float] | None = None
                                  ):
        if c is None:
            c = self.c
        return pt.mode_evolve_wo_scaling(kt, mode_ini,
                                         mode_der_ini, bg_sol,
                                         self.pot.get_u_of_theta_der,
                                         self.pot.u_of_theta_dder,
                                         ti, tf,
                                         method=method,
                                         rtol=rtol, atol=atol,
                                         t_eval=t_eval)

    def pt_mode_evolve_w_scaling(self,
                                 kt: float,
                                 x0: float,
                                 v0: float,
                                 bg_sol: OdeResult,
                                 ti: float | None = None,
                                 tf: float | None = None,
                                 c: float | None = None,
                                 method: str = "DOP853",
                                 dense_output: bool = False,
                                 events: Callable[..., float] | None = None,
                                 rtol: float = 1e-12,
                                 atol: float = 1e-12,
                                 t_eval: list[float] | None = None
                                 ):
        if c is None:
            c = self.c
        return pt.mode_evolve_w_scaling(kt, x0, v0, bg_sol,
                                        self.pot.ut_of_x_der,
                                        self.pot.ut_of_x_dder,
                                        ti=ti, tf=tf,
                                        method=method,
                                        dense_output=dense_output,
                                        events=events, rtol=rtol, atol=atol,
                                        t_eval=t_eval)

    @abstractmethod
    def bg_field_evolve(self):
        pass

    def _pt_mode_evolve(self, kt: float, precise_ics=False, t_eval=None):
        # Check whether the background solutions are computed.
        if len(self.bg_field) == 0:
            msg = "Run the method `bg_field_evolve` first!"
            raise AttributeError(msg)

        if precise_ics:
            msg = "Precise initial conditions for the modes has not been implemented yet."
            raise NotImplementedError(msg)
        else:
            mode_ini = 0.
            mode_der_ini = 0.

        sols = []

        for sol_dict in self.bg_field:
            # If the background solution is obtained without scaling
            if not sol_dict['scaled']:
                sol = self.pt_mode_evolve_wo_scaling(kt, mode_ini, mode_der_ini,
                                                     sol_dict['sol'])
                scaled = False
            else:
                x0, v0 = bg.scaled_initial_conditions(mode_ini, mode_der_ini,
                                                      sol_dict['sol'].t[0])
                sol = self.pt_mode_evolve_w_scaling(kt, x0, v0,
                                                    sol_dict['sol'],
                                                    t_eval=t_eval)
                scaled = True

            # Update the initial conditions to be used in the next step.
            mode_ini = sol.y[0][-1]
            mode_der_ini = sol.y[1][-1]

            # Add the solution to the list
            sols.append({'sol': sol, 'scaled': scaled, 'kt': kt})

        return sols

    def pt_modes_evolve(self,
                        kt_list: np.ndarray,
                        precise_ics: bool = False,
                        method: str = 'DOP853',
                        rtol: float = 1e-12,
                        atol: float = 1e-12,
                        t_eval: list[float] | None = None
                        ) -> None:
        self.kt_list = kt_list
        self.pt_modes = [self._pt_mode_evolve(kt,
                                              precise_ics=precise_ics,
                                              t_eval=t_eval)
                         for kt in kt_list]

    def get_dc_evolution(self,
                         mode_index: int,
                         sol_index: int = -1):
        try:
            mode_sol_dict = self.pt_modes[mode_index][sol_index]
            kt = mode_sol_dict['kt']
            tm = mode_sol_dict['sol'].t
        except IndexError:
            print("`sol_index` for the perturbations is out of bounds!")
        except AttributeError:
            print("You need to run the method `pt_modes_evolve` first.")

        try:
            bg_sol_dict = self.bg_field[sol_index]
        except IndexError:
            print("`sol_index` for the background is out of bounds!")
        except AttributeError:
            print("You need to run the method `bg_field_evolve` first")

        assert mode_sol_dict['scaled'] == bg_sol_dict['scaled']

        if mode_sol_dict['scaled']:
            mode = mode_sol_dict['sol'].y[0]
            mode_der = mode_sol_dict['sol'].y[1]
            bg_sol = bg_sol_dict['sol']
            dc = np.array([pt.mode_dc_unscaled(kt, tm[i], mode[i], mode_der[i],
                                               bg_sol.sol(tm[i])[0],
                                               bg_sol.sol(tm[i])[1],
                                               self.pot.u_of_theta,
                                               self.pot.u_of_theta_der)
                           for i in range(len(tm))])
        else:
            x_pt = mode_sol_dict['sol'].y[0]
            v_pt = mode_sol_dict['sol'].y[1]
            bg_sol = bg_sol_dict['sol']
            dc = np.array([pt.mode_dc_scaled(kt, tm[i], x_pt[i], v_pt[i],
                                             bg_sol.sol(tm[i])[0],
                                             bg_sol.sol(tm[i])[1],
                                             self.pot.ut_of_x,
                                             self.pot.ut_of_x_der)
                           for i in range(len(tm))])

        return tm, dc

    def get_avg_dc_evolution(self,
                             tm_start: float,
                             mode_index: int,
                             sol_index: int = -1,
                             step: int = 5):
        t, dc = self.get_dc_evolution(mode_index, sol_index)

        if tm_start < t[0]:
            raise ValueError("""`tm_start` is outside the time range of the
            mode solution!""")

        if step < 1:
            raise ValueError("`step` must be a positive non-zero integer.")

        t_res = t[t > tm_start]
        dc_res = dc[t > tm_start]
        max_indices = argrelmax(dc_res)[0][0:-1:step]
        t_avg = np.zeros(len(max_indices) - 1)
        dc_avg = np.zeros(len(max_indices) - 1)
        for i in range(len(max_indices) - 1):
            ind1 = max_indices[i]
            ind2 = max_indices[i + 1]
            t_avg[i] = np.average(t_res[ind1:ind2])
            dc_avg[i] = np.average(dc_res[ind1:ind2])

        # Calculate the derivative using backwards differences
        dc_der_avg = np.zeros(len(dc_avg))
        dc_der_avg[0] = 0.
        for i in range(1, len(dc_avg)):
            num = dc_avg[i] - dc_avg[i - 1]
            den = t_avg[i] - t_avg[i - 1]
            dc_der_avg[i] = num/den
        return t_avg, dc_avg, dc_der_avg

    def _pt_dc_eval_latetime_single(self, mode_index: int,
                                    tm_start: float | None = None,
                                    step: int = 5,
                                    index: int = -2):
        # Get the dimensionless momentum corresponding to the mode index.
        try:
            mode_sol_dict = self.pt_modes[mode_index][-1]
            kt = mode_sol_dict['kt']
            tm = mode_sol_dict['sol'].t
        except AttributeError:
            print("You need to run the method `pt_modes_evolve` first.")

        if tm_start is None:
            tm_start = 0.5*tm

        # Get the averaged density constrasty evolution
        t_avg, dc_avg, dc_der_avg = self.get_avg_dc_evolution(tm_start,
                                                              mode_index,
                                                              sol_index=-1,
                                                              step=step)
        
        sol = pt.dc_eval_latetime(kt, dc_avg[index], dc_der_avg[index],
                                  self.cosmo, z_start, z_end)


class StandardALP(ALP):
    def __init__(self,
                 theta_ini: float,
                 pot: Potential,
                 m0: Quantity | None = None,
                 cosmo: FlatLambdaCDM = Planck18,
                 c: float = 1.
                 ):
        super().__init__(pot, m0=m0, cosmo=cosmo, c=c)
        self.theta_ini = theta_ini
        self.bg_field = []
        self.t_zero_cross = None
        self.t_first_min = None

    def bg_get_precise_ics(self, ti: float) -> tuple[float, float]:
        musq = self.pot.u_of_theta_dder(self.theta_ini)
        if musq >= 0:
            mu = np.sqrt(musq) + 0j
        else:
            mu = 0. + np.sqrt(-musq)*1j
        u_der_ini = self.pot.u_of_theta_der(self.theta_ini)
        theta_ini_prc = analytical.standard_bg_before_osc(ti, mu,
                                                          u_der_ini,
                                                          self.theta_ini)
        theta_der_ini_prc = analytical.standard_bg_before_osc_der(ti, mu,
                                                                  u_der_ini)
        return theta_ini_prc, theta_der_ini_prc

    def bg_field_evolve(self, ti: float, tf: float,
                        precise_ics: bool = True,
                        method: str = 'DOP853',
                        dense_output: bool = True,
                        rtol: float = 1e-12,
                        atol: float = 1e-12,
                        stop_after_first_osc: bool = False,
                        c: float | None = None
                        ):

        if precise_ics:
            theta_ini, theta_der_ini = self.bg_get_precise_ics(ti)

        else:
            theta_ini = self.theta_ini
            theta_der_ini = 0.0

        def zero_crossing(t, y):
            return y[0]
        zero_crossing.terminal = False
        zero_crossing.direction = -1

        def local_minimum(t, y):
            return y[1]
        local_minimum.terminal = True
        local_minimum.direction = 1

        events = [zero_crossing, local_minimum]

        if c is None:
            c = self.c

        # Evolve the field w/o scaling in any case
        sol_early = self.bg_field_evolve_wo_scaling(ti, tf, theta_ini,
                                                    theta_der_ini,
                                                    c=c, method=method,
                                                    dense_output=dense_output,
                                                    events=events,
                                                    rtol=rtol, atol=atol)

        # Store the zero-crossing time and the time of first local_minimum
        # as properties
        if len(sol_early.t_events[0]) != 1:
            raise RuntimeError("Multiple zero crossings recorded!")
        if len(sol_early.t_events[1]) != 1:
            raise RuntimeError("Multiple local minima recorded!")
        self.t_zero_cross = sol_early.t_events[0][0]
        self.t_first_min = sol_early.t_events[1][0]

        self.bg_field.append({
            'sol': sol_early,
            'scaled': False
        })

        # Continue evolving the field w/ scaling if requested
        if not stop_after_first_osc:
            ts = sol_early.t[-1]
            x0, v0 = bg.scaled_initial_conditions(sol_early.y[0][-1],
                                                  sol_early.y[1][-1],
                                                  ts)
            sol_late = self.bg_field_evolve_w_scaling(ts, tf, x0, v0, c=c,
                                                      method=method,
                                                      dense_output=dense_output,
                                                      rtol=rtol, atol=atol)
            self.bg_field.append({
                'sol': sol_late,
                'scaled': True
            })

        



class KineticALP(ALP):
    pass


def mass_hubble_cross_z(m,
                        c: float,
                        cosmo: FlatLambdaCDM = Planck18,
                        z_guess: float | None = None):
    m0_eV = nat.convert(m, nat.eV).value
        
    def fun(lg_z):
        Hval = nat.convert(cosmo.H(np.exp(lg_z)), nat.eV).value
        return np.log(m0_eV/(Hval*c))
    
    if z_guess is None:
        # TODO: Calculate the z at equality via a function.
        z_guess = 3402.  # z at equality
        
    
    return np.exp(root(fun, np.log(z_guess)).x[0])
