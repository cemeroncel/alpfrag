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
from alpfrag.paths import RESULTS_PATH
import h5py
import alpfrag.saving as save
from pathlib import Path


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
                 c: float = 1.,
                 run_id: int | None = None
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
        self.run_id = run_id

    @abstractmethod
    def fname_base(self) -> str:
        pass

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

    def convert_kt_to_kMpc(self, kt: float) -> [float, float]:
        # Check whether m0 is defined
        if self.m0 is None:
            msg = "You need to set the ALP mass to use this method!"
            raise AttributeError(msg)

        # Convert the ALP mass to Mpc^-1
        m0_Mpc = nat.convert(self.m0, nat.Mpc**-1).value

        # Calculate k in Mpc^-1
        k_Mpc = np.sqrt(2./self.c)*kt*m0_Mpc/(1. + self.z1)

        # Return k in Mpc^-1 and h*Mpc^-1
        return k_Mpc, k_Mpc/self.cosmo.h

    def convert_kMpc_to_kt(self, k: Quantity) -> float:
        # Check whether m0 is defined
        if self.m0 is None:
            msg = "You need to set the ALP mass to use this method!"
            raise AttributeError(msg)

        # Convert the ALP mass to Mpc^-1
        m0_Mpc = nat.convert(self.m0, nat.Mpc**-1).value

        # Convert k to Mpc^-1
        kMpc = nat.convert(k, nat.Mpc**-1).value

        return np.sqrt(self.c*0.5)*(kMpc/m0_Mpc)*(1. + self.z1)

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
                         kt: float,
                         mode_sol: OdeResult,
                         bg_sol: OdeResult):

        err = "Both mode and bg solutions should be either scaled or not scaled!"
        assert mode_sol.scaled == bg_sol.scaled, err

        if mode_sol.scaled:
            t = mode_sol.t
            x_pt, v_pt = mode_sol.y
            x_bg, v_bg = bg_sol.sol(t)
            return pt.mode_dc_scaled(kt, t, x_pt, v_pt, x_bg, v_bg,
                                     self.pot.ut_of_x, self.pot.ut_of_x_der)
        else:
            raise NotImplementedError("Not implemented!")

    def get_dc_evolution_old(self,
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
            x_pt = mode_sol_dict['sol'].y[0]
            v_pt = mode_sol_dict['sol'].y[1]
            bg_sol = bg_sol_dict['sol']
            dc = np.array([pt.mode_dc_scaled(kt, tm[i], x_pt[i], v_pt[i],
                                             bg_sol.sol(tm[i])[0],
                                             bg_sol.sol(tm[i])[1],
                                             self.pot.ut_of_x,
                                             self.pot.ut_of_x_der)
                           for i in range(len(tm))])
            
        else:
            mode = mode_sol_dict['sol'].y[0]
            mode_der = mode_sol_dict['sol'].y[1]
            bg_sol = bg_sol_dict['sol']
            dc = np.array([pt.mode_dc_unscaled(kt, tm[i], mode[i], mode_der[i],
                                               bg_sol.sol(tm[i])[0],
                                               bg_sol.sol(tm[i])[1],
                                               self.pot.u_of_theta,
                                               self.pot.u_of_theta_der)
                           for i in range(len(tm))])

        return tm, dc

    def get_avg_dc_evolution(self,
                             mode_index: int,
                             tm_start: float | None = None,
                             sol_index: int = -1,
                             step: int = 5):
        t, dc = self.get_dc_evolution(mode_index, sol_index)

        if step < 1:
            raise ValueError("`step` must be a positive non-zero integer.")

        if tm_start is None:
            t_res = t
            dc_res = dc
        else:
            if tm_start < t[0]:
                raise ValueError("""`tm_start` is outside the time range of the
                mode solution!""")

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

    def pt_avg_dc_and_save(self,
                           tm_start: float | None = None,
                           step: int = 5,
                           index: int = -2,
                           fname: str | None = None,
                           results_path: Path | None = None):
        try:
            tm_match_arr = np.zeros(len(self.kt_list))
            delta_arr = np.zeros(len(self.kt_list))
            delta_der_arr = np.zeros(len(self.kt_list))

        except AttributeError:
            print("You need to run the method `pt_modes_evolve` first.")

        if tm_start is None:
            tm_start = 0.5*self.pt_modes[0][-1]['sol'].t[-1]

        for i in range(len(self.kt_list)):
            r = self.get_avg_dc_evolution(tm_start, i, step=step)
            tm_match_arr[i] = r[0][index]
            delta_arr[i] = r[1][index]
            delta_der_arr[i] = r[2][index]

        if fname is None:
            timestamp = save.get_timestamp()
            self.run_id = int(timestamp)
            fname = self.fname_base + "_int_" + timestamp + '.hdf5'

        if results_path is None:
            results_path = RESULTS_PATH

        with h5py.File(results_path / fname, 'a') as f:
            f.create_dataset('kt', data=self.kt_list)
            f.create_dataset('delta', data=delta_arr)
            f.create_dataset('delta_der_tm', data=delta_der_arr)
            f.create_dataset('tm_match', data=tm_match_arr)
            f.attrs['potential'] = str(self.pot)
            f.attrs['theta_initial'] = self.theta_ini
            f.attrs['tm_initial'] = self.bg_field[0]['sol'].t[0]
            f.attrs['tm_final'] = self.bg_field[-1]['sol'].t[-1]
            f.close

        print(f"Results saved in {results_path / fname}.")

        return self.kt_list, delta_arr, delta_der_arr, tm_match_arr

    def _pt_dc_eval_latetime_single(self, mode_index: int,
                                    z_end: float = 0.,
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
            tm_start = 0.5*tm[-1]

        # Get the averaged density contrast evolution
        t_avg, dc_avg, dc_der_avg = self.get_avg_dc_evolution(mode_index,
                                                              tm_start,
                                                              sol_index=-1,
                                                              step=step)

        # Get the simulation time at matching
        tm_match = t_avg[index]

        # Convert tm_match to redshift
        z_start = self.convert_tm_to_redshift(tm_match)

        # dc_der_avg constains derivatives with respect to tm. We need
        # to convert this to ln(y) derivatives where y = a/aeq
        dc = dc_avg[index]
        dc_der = 2*t_avg[index]*dc_der_avg[index]

        sol = pt.dc_eval_latetime(kt, dc, dc_der,
                                  self.cosmo, z_start, z_end)

        kMpc, khMpc = self.convert_kt_to_kMpc(kt)

        return {
            'kt': kt,
            'kMpc': kMpc,
            'khMpc': khMpc,
            'z_arr': (1. + self.cosmo.zeq)*np.exp(-sol.t) - 1.,
            'delta': sol.y[0],
            'sol': sol
        }

    def pt_dc_eval_latetime_direct(self, z_end: float = 0,
                                   tm_start: float | None = None,
                                   step: int = 5,
                                   index: int = -2):
        self.pt_dc_latetime = []
        for i in range(len(self.kt_list)):
            dc_dict = self._pt_dc_eval_latetime_single(i, z_end, tm_start,
                                                       step, index)
            self.pt_dc_latetime.append(dc_dict)

    def pt_dc_eval_latetime_savefile(self, savefile: str | Path,
                                     z_end: float = 0.):
        # fname = self.fname_base + "_int_" + str(self.run_id) + '.hdf5'
        f = h5py.File(savefile, 'r')
        # kt_list = f['kt_list']
        kt_list = f['kt'][...]
        delta_list = f['delta'][...]
        tm_match_list = f['tm_match'][...]
        delta_der_tm_list = f['delta_der_tm'][...]
        f.close()

        # Convert tm derivatives to convert to ln(y) derivatives where
        # y = a/aeq
        delta_der_lny_list = 2.*tm_match_list*delta_der_tm_list

        # Creating the list to store the solutions
        self.pt_dc_latetime = []

        # Loop to compute the solutions
        for i, kt in enumerate(kt_list):
            sol = pt.dc_eval_latetime(kt, delta_list[i], delta_der_lny_list[i],
                                      self.cosmo,
                                      self.convert_tm_to_redshift(tm_match_list[i]),
                                      z_end)
            if sol.success:
                kMpc, khMpc = self.convert_kt_to_kMpc(kt)
                self.pt_dc_latetime.append({
                    'kt': kt,
                    'kMpc': kMpc,
                    'khMpc': khMpc,
                    'z_arr': (1. + self.cosmo.zeq)*np.exp(-sol.t) - 1.,
                    'delta': sol.y[0],
                    'sol': sol
                })
            else:
                msg = "Integration failed for kt={:.2f}".format(kt)
                raise RuntimeError(msg)

    def get_latetime_powerspectrum(self):
        try:
            number_of_modes = len(self.pt_dc_latetime)
        except AttributeError:
            print("Run the method ")

        kt = np.zeros(number_of_modes)
        kMpc = np.zeros(number_of_modes)
        khMpc = np.zeros(number_of_modes)
        deltasq = np.zeros(number_of_modes)

        for i, sol in enumerate(self.pt_dc_latetime):
            kt[i] = sol['kt']
            kMpc[i] = sol['kMpc']
            khMpc[i] = sol['khMpc']
            deltasq[i] = sol['delta'][-1]**2

        return {
            'kt': kt,
            'kMpc': kMpc,
            'khMpc': khMpc,
            'deltasq': deltasq
        }



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
        self.pt_modes = []
        self.t_zero_cross = None
        self.t_first_min = None

    @property
    def fname_base(self) -> str:
        mech = 'smm'
        pot_str = str(self.pot)
        if self.m0 is None:
            m_str = "mNone"
        else:
            m_str = "m" + "{:.2e}".format(nat.convert(self.m0, nat.eV).value)
        i_str = "thi" + "{:.3f}".format(self.theta_ini)
        return mech + "_" + pot_str + "_" + m_str + "_" + i_str

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
                        c: float | None = None,
                        stop_after_convergence: bool = True,
                        convergence_tol: float = 1e-2,
                        verbose: bool = True,
                        ):
        # For safety, let us make sure that the list is empty
        assert len(self.bg_field) == 0

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
        sol_early.scaled = False

        # Store the zero-crossing time and the time of first local_minimum
        # as properties
        if len(sol_early.t_events[0]) != 1:
            raise RuntimeError("Multiple zero crossings recorded!")
        if len(sol_early.t_events[1]) != 1:
            raise RuntimeError("Multiple local minima recorded!")
        self.t_zero_cross = sol_early.t_events[0][0]
        self.t_first_min = sol_early.t_events[1][0]

        self.bg_field.append(sol_early)

        if verbose:
            print(f"""Initial run completed.
            Zero crossing at tm = {self.t_zero_cross}.
            First minimum at tm = {self.t_first_min}.
            ---""")

        # Continue evolving the field w/ scaling if requested
        if (not stop_after_first_osc) and (not stop_after_convergence):
            ts = sol_early.t[-1]
            x0, v0 = bg.scaled_initial_conditions(sol_early.y[0][-1],
                                                  sol_early.y[1][-1],
                                                  ts)
            sol_late = self.bg_field_evolve_w_scaling(ts, tf, x0, v0, c=c,
                                                      method=method,
                                                      dense_output=dense_output,
                                                      rtol=rtol, atol=atol)
            sol_late.scaled = True
            self.bg_field.append(sol_late)


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
