#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for alpfrag.potentials.
"""
import alpfrag.potentials as potentials
import numpy as np

# Check whether non-periodic potential reduces to quadratic one when
# p=1.


class TestNonPeriodicPotential:
    p = 1.
    quad = potentials.Free()
    nonper = potentials.NonPeriodic(p)
    theta_range = np.geomspace(1e-8, 1e2)
    x = 2.42
    t_range = np.geomspace(1e-2, 2e3)

    def test_u_of_theta(self):
        from_quad = self.quad.get_u_of_theta(self.theta_range)
        from_nonper = self.nonper.get_u_of_theta(self.theta_range)
        assert np.allclose(from_quad, from_nonper)

    def test_u_of_theta_der(self):
        from_quad = self.quad.get_u_of_theta_der(self.theta_range)
        from_nonper = self.nonper.get_u_of_theta_der(self.theta_range)
        assert np.allclose(from_quad, from_nonper)

    def test_u_of_theta_dder(self):
        from_quad = self.quad.get_u_of_theta_dder(self.theta_range)
        from_nonper = self.nonper.get_u_of_theta_dder(self.theta_range)
        assert np.allclose(from_quad, from_nonper)

    def test_ut_of_x(self):
        from_quad = self.quad.get_ut_of_x(self.x, self.t_range)
        from_nonper = self.nonper.get_ut_of_x(self.x, self.t_range)
        assert np.allclose(from_quad, from_nonper)

    def test_ut_of_x_der(self):
        from_quad = self.quad.get_ut_of_x_der(self.x, self.t_range)
        from_nonper = self.nonper.get_ut_of_x_der(self.x, self.t_range)
        assert np.allclose(from_quad, from_nonper)

    def test_ut_of_x_dder(self):
        from_quad = self.quad.get_ut_of_x_dder(self.x, self.t_range)
        from_nonper = self.nonper.get_ut_of_x_dder(self.x, self.t_range)
        assert np.allclose(from_quad, from_nonper)
