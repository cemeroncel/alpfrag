"""Test modules for alpfrag.symplectic."""

import numpy as np
import alpfrag.symplectic as symplectic


class TestYoshidaCoefs:

    def test_vv2(self):
        w = symplectic.get_yoshida_coefs_vec('vv2')
        assert np.isclose(np.sum(w), 1.)

    def test_vv4(self):
        w = symplectic.get_yoshida_coefs_vec('vv4')
        assert np.isclose(np.sum(w), 1.)

    def test_vv6(self):
        w = symplectic.get_yoshida_coefs_vec('vv6')
        assert np.isclose(np.sum(w), 1.)

    def test_vv8(self):
        w = symplectic.get_yoshida_coefs_vec('vv8')
        assert np.isclose(np.sum(w), 1.)

    def test_vv10(self):
        w = symplectic.get_yoshida_coefs_vec('vv10')
        assert np.isclose(np.sum(w), 1.)
