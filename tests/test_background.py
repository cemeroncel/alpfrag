#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for alpfrag.background.
"""
import alpfrag.background as bg
import alpfrag.quadratic as qdr
import numpy as np
import pytest


class TestScaledInitialConditions:
    def test_ti_unity(self):
        theta_ini = np.random.rand()
        theta_der_ini = np.random.rand()
        x0_get, v0_get = bg.scaled_initial_conditions(theta_ini,
                                                      theta_der_ini,
                                                      1.)
        x0_exp = theta_ini
        v0_exp = 0.75*theta_ini + theta_der_ini
        assert np.isclose(x0_exp, x0_get) and np.isclose(v0_exp, v0_get)

    def test_ti_zero(self):
        theta_ini = np.random.rand()
        theta_der_ini = np.random.rand()
        with pytest.raises(ValueError, match=r".*needs*."):
            bg.scaled_initial_conditions(theta_ini, theta_der_ini, 0.)


# Quadratic potential
def _u_quad(theta):
    return 0.5*(theta**2)


def _u_der_quad(theta):
    return theta


def _u_dder_quad(theta):
    return 1.


def _ut_quad(t, x):
    return 0.5*(x**2)


def _ut_der_quad(t, x):
    return x


def _ut_dder_quad(t, x):
    return 1.


# Power law potential
def _u_pl(theta, n):
    return (theta**n)/n


def _u_der_pl(theta, n):
    return theta**(n - 1)


def _u_dder_pl(theta, n):
    return (n - 1)*(theta**(n - 2))


def _ut_pl(t, x, n):
    return 0.5*(x**n)*(t**(1.5 - 0.75*n))


def _ut_der_pl(t, x, n):
    return 0.5*n*(x**(n - 1))*(t**(1.5 - 0.75*n))


def _ut_dder_pl(t, x, n):
    return 0.5*(n-1)*n*(x**(n - 2))*(t**(1.5 - 0.75*n))


class TestFieldEvolveWoScalingExpl:
    # Initial and final times
    ti = 1e-2
    tf = 2e3

    # Rescale the initial amplitude
    r = 4.2

    # Get precise initial conditions
    theta_ini = r*qdr.normalized_field_amplitude(ti)
    theta_der_ini = r*qdr.normalized_field_velocity(ti)

    # Get the solutions
    sol_quad = bg.field_evolve_wo_scaling(ti, tf, theta_ini,
                                          theta_der_ini, _u_der_quad)
    sol_pl = bg.field_evolve_wo_scaling(ti, tf, theta_ini, theta_der_ini,
                                        _u_der_pl, u_args=(2,))

    def test_quadratic_potential_pos(self):
        expected = self.r*qdr.normalized_field_amplitude(self.sol_quad.t)
        found = self.sol_quad.y[0]
        assert np.allclose(expected, found)

    def test_quadratic_potential_vel(self):
        expected = self.r*qdr.normalized_field_velocity(self.sol_quad.t)
        found = self.sol_quad.y[1]
        assert np.allclose(expected, found)

    def test_quadratic_potential_ed(self):
        expected = (self.r**2)*qdr.normalized_energy_density(self.sol_quad.t)
        found = bg.get_energy_density(self.sol_quad.t,
                                      self.sol_quad.y[0],
                                      self.sol_quad.y[1],
                                      _u_quad)
        assert np.allclose(expected, found)

    def test_quadratic_potential_com_ed(self):
        expected = (self.r**2)*qdr.normalized_comoving_energy_density(self.sol_quad.t)
        found = bg.get_com_energy_density(self.sol_quad.t,
                                          self.sol_quad.y[0],
                                          self.sol_quad.y[1],
                                          _u_quad)
        assert np.allclose(expected, found)

    def test_quadratic_potential_eos(self):
        expected = qdr.equation_of_state(self.sol_quad.t)
        found = bg.get_eos(self.sol_quad.t,
                           self.sol_quad.y[0],
                           self.sol_quad.y[1],
                           _u_quad)
        assert np.allclose(expected, found)

    def test_power_law_potential_pos(self):
        expected = self.r*qdr.normalized_field_amplitude(self.sol_pl.t)
        found = self.sol_pl.y[0]
        assert np.allclose(expected, found)

    def test_power_law_potential_vel(self):
        expected = self.r*qdr.normalized_field_velocity(self.sol_pl.t)
        found = self.sol_pl.y[1]
        assert np.allclose(expected, found)

    def test_power_law_potential_ed(self):
        expected = (self.r**2)*qdr.normalized_energy_density(self.sol_pl.t)
        found = bg.get_energy_density(self.sol_pl.t,
                                      self.sol_pl.y[0],
                                      self.sol_pl.y[1],
                                      _u_pl, u_args=(2, ))
        assert np.allclose(expected, found)

    def test_power_law_potential_com_ed(self):
        expected = (self.r**2)*qdr.normalized_comoving_energy_density(self.sol_pl.t)
        found = bg.get_com_energy_density(self.sol_pl.t,
                                          self.sol_pl.y[0],
                                          self.sol_pl.y[1],
                                          _u_pl, u_args=(2,))
        assert np.allclose(expected, found)

    def test_power_law_potential_eos(self):
        expected = qdr.equation_of_state(self.sol_quad.t)
        found = bg.get_eos(self.sol_pl.t,
                           self.sol_pl.y[0],
                           self.sol_pl.y[1],
                           _u_pl, u_args=(2, ))
        assert np.allclose(expected, found)

    def test_quadratic_potential_stopping_time(self):
        def local_maximum(t, y):
            return y[1]
        local_maximum.terminal = True
        local_maximum.direction = -1
        sol = bg.field_evolve_wo_scaling(self.ti, self.tf, self.theta_ini,
                                         self.theta_der_ini, _u_der_quad,
                                         events=local_maximum)
        expected = 7.3728964134327485746
        found = sol.t_events[0][0]
        assert np.isclose(expected, found)

    def test_power_law_potential_stopping_time(self):
        args = (2, )

        def local_maximum(t, y, *args):
            return y[1]
        local_maximum.terminal = True
        local_maximum.direction = -1
        sol = bg.field_evolve_wo_scaling(self.ti, self.tf, self.theta_ini,
                                         self.theta_der_ini, _u_der_pl,
                                         u_args=args,
                                         events=local_maximum)
        expected = 7.3728964134327485746
        found = sol.t_events[0][0]
        assert np.isclose(expected, found)


class TestFieldEvolveWoScalingImpl:
    # Initial and final times
    ti = 1e-2
    tf = 2e3

    # Rescale the initial amplitude
    r = 4.2

    # Get precise initial conditions
    theta_ini = r*qdr.normalized_field_amplitude(ti)
    theta_der_ini = r*qdr.normalized_field_velocity(ti)

    # Get the solutions
    sol_quad = bg.field_evolve_wo_scaling(ti, tf, theta_ini,
                                          theta_der_ini, _u_der_quad,
                                          _u_dder_quad, method='LSODA')
    sol_pl = bg.field_evolve_wo_scaling(ti, tf, theta_ini, theta_der_ini,
                                        _u_der_pl, _u_dder_pl, u_args=(2,),
                                        method='LSODA')

    def test_quadratic_potential_pos(self):
        expected = self.r*qdr.normalized_field_amplitude(self.sol_quad.t)
        found = self.sol_quad.y[0]
        assert np.allclose(expected, found)

    def test_quadratic_potential_vel(self):
        expected = self.r*qdr.normalized_field_velocity(self.sol_quad.t)
        found = self.sol_quad.y[1]
        assert np.allclose(expected, found)

    def test_quadratic_potential_ed(self):
        expected = (self.r**2)*qdr.normalized_energy_density(self.sol_quad.t)
        found = bg.get_energy_density(self.sol_quad.t,
                                      self.sol_quad.y[0],
                                      self.sol_quad.y[1],
                                      _u_quad)
        assert np.allclose(expected, found)

    def test_quadratic_potential_com_ed(self):
        expected = (self.r**2)*qdr.normalized_comoving_energy_density(self.sol_quad.t)
        found = bg.get_com_energy_density(self.sol_quad.t,
                                          self.sol_quad.y[0],
                                          self.sol_quad.y[1],
                                          _u_quad)
        assert np.allclose(expected, found)

    def test_quadratic_potential_eos(self):
        expected = qdr.equation_of_state(self.sol_quad.t)
        found = bg.get_eos(self.sol_quad.t,
                           self.sol_quad.y[0],
                           self.sol_quad.y[1],
                           _u_quad)
        assert np.allclose(expected, found)

    def test_power_law_potential_pos(self):
        expected = self.r*qdr.normalized_field_amplitude(self.sol_pl.t)
        found = self.sol_pl.y[0]
        assert np.allclose(expected, found)

    def test_power_law_potential_vel(self):
        expected = self.r*qdr.normalized_field_velocity(self.sol_pl.t)
        found = self.sol_pl.y[1]
        assert np.allclose(expected, found)

    def test_power_law_potential_ed(self):
        expected = (self.r**2)*qdr.normalized_energy_density(self.sol_pl.t)
        found = bg.get_energy_density(self.sol_pl.t,
                                      self.sol_pl.y[0],
                                      self.sol_pl.y[1],
                                      _u_pl, u_args=(2, ))
        assert np.allclose(expected, found)

    def test_power_law_potential_com_ed(self):
        expected = (self.r**2)*qdr.normalized_comoving_energy_density(self.sol_pl.t)
        found = bg.get_com_energy_density(self.sol_pl.t,
                                          self.sol_pl.y[0],
                                          self.sol_pl.y[1],
                                          _u_pl, u_args=(2,))
        assert np.allclose(expected, found)

    def test_power_law_potential_eos(self):
        expected = qdr.equation_of_state(self.sol_quad.t)
        found = bg.get_eos(self.sol_pl.t,
                           self.sol_pl.y[0],
                           self.sol_pl.y[1],
                           _u_pl, u_args=(2, ))
        assert np.allclose(expected, found)

    def test_quadratic_potential_stopping_time(self):
        def local_maximum(t, y):
            return y[1]
        local_maximum.terminal = True
        local_maximum.direction = -1
        sol = bg.field_evolve_wo_scaling(self.ti, self.tf, self.theta_ini,
                                         self.theta_der_ini, _u_der_quad,
                                         _u_dder_quad, method='LSODA',
                                         events=local_maximum)
        expected = 7.3728964134327485746
        found = sol.t_events[0][0]
        assert np.isclose(expected, found)

    def test_power_law_potential_stopping_time(self):
        args = (2, )

        def local_maximum(t, y, *args):
            return y[1]
        local_maximum.terminal = True
        local_maximum.direction = -1
        sol = bg.field_evolve_wo_scaling(self.ti, self.tf, self.theta_ini,
                                         self.theta_der_ini, _u_der_pl,
                                         _u_dder_pl, u_args=args,
                                         events=local_maximum, method='LSODA')
        expected = 7.3728964134327485746
        found = sol.t_events[0][0]
        assert np.isclose(expected, found)


class TestFieldEvolveWScalingExpl:
    # Initial and final times
    ti = 1e0
    tf = 2e3

    # Rescale the initial amplitude
    r = 4.2

    # Get precise initial conditions
    theta_ini = r*qdr.normalized_field_amplitude(ti)
    theta_der_ini = r*qdr.normalized_field_velocity(ti)

    # Scale the initial conditions
    x0, v0 = bg.scaled_initial_conditions(theta_ini, theta_der_ini, ti)

    # Get the solutions
    sol_quad = bg.field_evolve_w_scaling(ti, tf, x0, v0, _ut_der_quad)
    sol_pl = bg.field_evolve_w_scaling(ti, tf, x0, v0,
                                       _ut_der_pl, u_args=(2,))

    # Unscale the solutions
    quad_unscl = bg.unscale_solution(sol_quad.t, sol_quad.y[0], sol_quad.y[1])
    pl_unscl = bg.unscale_solution(sol_pl.t, sol_pl.y[0], sol_pl.y[1])

    def test_quadratic_potential_pos(self):
        expected = self.r*qdr.normalized_field_amplitude(self.sol_quad.t)
        found = self.quad_unscl['theta']
        assert np.allclose(expected, found)

    def test_quadratic_potential_vel(self):
        expected = self.r*qdr.normalized_field_velocity(self.sol_quad.t)
        found = self.quad_unscl['theta_der']
        assert np.allclose(expected, found)

    def test_quadratic_potential_eos(self):
        expected = qdr.equation_of_state(self.sol_quad.t)
        found = bg.get_eos(self.quad_unscl['t'],
                           self.quad_unscl['theta'],
                           self.quad_unscl['theta_der'],
                           _u_quad)
        assert np.allclose(expected, found)

    def test_power_law_potential_pos(self):
        expected = self.r*qdr.normalized_field_amplitude(self.sol_pl.t)
        found = self.pl_unscl['theta']
        assert np.allclose(expected, found)

    def test_power_law_potential_vel(self):
        expected = self.r*qdr.normalized_field_velocity(self.sol_pl.t)
        found = self.pl_unscl['theta_der']
        assert np.allclose(expected, found)

    def test_power_law_potential_eos(self):
        expected = qdr.equation_of_state(self.sol_pl.t)
        found = bg.get_eos(self.pl_unscl['t'],
                           self.pl_unscl['theta'],
                           self.pl_unscl['theta_der'],
                           _u_pl, u_args=(2,))
        assert np.allclose(expected, found)


class TestFieldEvolveWScalingImpl:
    # Initial and final times
    ti = 1e0
    tf = 2e3

    # Rescale the initial amplitude
    r = 4.2

    # Get precise initial conditions
    theta_ini = r*qdr.normalized_field_amplitude(ti)
    theta_der_ini = r*qdr.normalized_field_velocity(ti)

    # Scale the initial conditions
    x0, v0 = bg.scaled_initial_conditions(theta_ini, theta_der_ini, ti)

    # Get the solutions
    sol_quad = bg.field_evolve_w_scaling(ti, tf, x0, v0, _ut_der_quad,
                                         _ut_dder_quad, method='LSODA')
    sol_pl = bg.field_evolve_w_scaling(ti, tf, x0, v0,
                                       _ut_der_pl, _ut_dder_pl,
                                       u_args=(2,), method='LSODA')

    # Unscale the solutions
    quad_unscl = bg.unscale_solution(sol_quad.t, sol_quad.y[0], sol_quad.y[1])
    pl_unscl = bg.unscale_solution(sol_pl.t, sol_pl.y[0], sol_pl.y[1])

    def test_quadratic_potential_pos(self):
        expected = self.r*qdr.normalized_field_amplitude(self.sol_quad.t)
        found = self.quad_unscl['theta']
        assert np.allclose(expected, found)

    def test_quadratic_potential_vel(self):
        expected = self.r*qdr.normalized_field_velocity(self.sol_quad.t)
        found = self.quad_unscl['theta_der']
        assert np.allclose(expected, found)

    def test_quadratic_potential_eos(self):
        expected = qdr.equation_of_state(self.sol_quad.t)
        found = bg.get_eos(self.quad_unscl['t'],
                           self.quad_unscl['theta'],
                           self.quad_unscl['theta_der'],
                           _u_quad)
        assert np.allclose(expected, found)

    def test_power_law_potential_pos(self):
        expected = self.r*qdr.normalized_field_amplitude(self.sol_pl.t)
        found = self.pl_unscl['theta']
        assert np.allclose(expected, found)

    def test_power_law_potential_vel(self):
        expected = self.r*qdr.normalized_field_velocity(self.sol_pl.t)
        found = self.pl_unscl['theta_der']
        assert np.allclose(expected, found)

    def test_power_law_potential_eos(self):
        expected = qdr.equation_of_state(self.sol_pl.t)
        found = bg.get_eos(self.pl_unscl['t'],
                           self.pl_unscl['theta'],
                           self.pl_unscl['theta_der'],
                           _u_pl, u_args=(2,))
        assert np.allclose(expected, found)


@pytest.mark.skip(reason='The function is deprecated!')
class TestFieldEvolveWScalingSymp:
    # Tolerance for the maximum error
    tol = 1e-6

    # Initial and final times
    ti = 1e0
    tf = 2e3

    # Rescale the initial amplitude
    r = 4.2

    # Get precise initial conditions
    theta_ini = r*qdr.normalized_field_amplitude(ti)
    theta_der_ini = r*qdr.normalized_field_velocity(ti)

    # Scale the initial conditions
    x0, v0 = bg.scaled_initial_conditions(theta_ini, theta_der_ini, ti)

    @pytest.mark.slow
    def test_quadratic_potential(self):
        sol_quad = bg.field_evolve_w_scaling_symp(self.ti, self.tf, self.x0,
                                                  self.v0, _ut_der_quad)
        quad_unscl = bg.unscale_solution(sol_quad['t'], sol_quad['x'],
                                         sol_quad['v'])
        expected_pos = self.r*qdr.normalized_field_amplitude(sol_quad['t'])
        found_pos = quad_unscl['theta']
        expected_vel = self.r*qdr.normalized_field_velocity(sol_quad['t'])
        found_vel = quad_unscl['theta_der']
        max_pos_err = np.max(np.abs(expected_pos - found_pos))
        max_vel_err = np.max(np.abs(expected_vel - found_vel))
        assert max(max_pos_err, max_vel_err) < self.tol

    @pytest.mark.slow
    def test_power_law_potential(self):
        sol_pl = bg.field_evolve_w_scaling_symp(self.ti, self.tf, self.x0,
                                                self.v0, _ut_der_pl,
                                                u_args=(2, ))
        pl_unscl = bg.unscale_solution(sol_pl['t'], sol_pl['x'],
                                       sol_pl['v'])
        expected_pos = self.r*qdr.normalized_field_amplitude(sol_pl['t'])
        found_pos = pl_unscl['theta']
        expected_vel = self.r*qdr.normalized_field_velocity(sol_pl['t'])
        found_vel = pl_unscl['theta_der']
        max_pos_err = np.max(np.abs(expected_pos - found_pos))
        max_vel_err = np.max(np.abs(expected_vel - found_vel))
        assert max(max_pos_err, max_vel_err) < self.tol
