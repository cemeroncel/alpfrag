#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for alpfrag.models."""
import alpfrag.models as models
import alpfrag.quadratic as qdr
import alpfrag.background as bg
import alpfrag.potentials as pot
import numpy as np
import pytest


class TestFreeStandardALP:
    model = models.StandardALP(1., pot.Free())
    ti = 1e-2
    tf = 2e3
    model.bg_field_evolve(ti, tf)

    def test_bg_get_precise_ics(self):
        expected = [qdr.normalized_field_amplitude(self.ti),
                    qdr.normalized_field_velocity(self.ti)]
        obtained = self.model.bg_get_precise_ics(self.ti)
        assert np.allclose(expected, obtained)

    def test_zero_crossing(self):
        expected = 2.78088772399498
        obtained = self.model.t_zero_cross
        assert np.isclose(expected, obtained)

    def test_local_minimum(self):
        expected = 4.16542628447191
        obtained = self.model.t_first_min
        assert np.isclose(expected, obtained)

    def test_bg_field_early(self):
        t = self.model.bg_field[0]['sol'].t
        expected = qdr.normalized_field_amplitude(t)
        obtained = self.model.bg_field[0]['sol'].y[0]
        assert np.allclose(expected, obtained)

    def test_bg_field_vel_early(self):
        t = self.model.bg_field[0]['sol'].t
        expected = qdr.normalized_field_velocity(t)
        obtained = self.model.bg_field[0]['sol'].y[1]
        assert np.allclose(expected, obtained)

    def test_bg_field_late(self):
        t = self.model.bg_field[1]['sol'].t
        x = self.model.bg_field[1]['sol'].y[0]
        v = self.model.bg_field[1]['sol'].y[1]
        expected = qdr.normalized_field_amplitude(t)
        sol_unscl = bg.unscale_solution(t, x, v)
        obtained = sol_unscl['theta']
        assert np.allclose(expected, obtained)

    def test_bg_field_vel_late(self):
        t = self.model.bg_field[1]['sol'].t
        x = self.model.bg_field[1]['sol'].y[0]
        v = self.model.bg_field[1]['sol'].y[1]
        expected = qdr.normalized_field_velocity(t)
        sol_unscl = bg.unscale_solution(t, x, v)
        obtained = sol_unscl['theta_der']
        assert np.allclose(expected, obtained)
