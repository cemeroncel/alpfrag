"""Tests for alpfrag.perturbations."""
import numpy as np
import alpfrag.perturbations as pt
import pytest
import sys


class TestModeTime:
    def test_tm_validity_zero(self):
        with pytest.raises(ValueError, match=r".*needs*."):
            pt.mode_time(0., np.random.rand())

    def test_tm_validity_minus(self):
        with pytest.raises(ValueError, match=r".*needs*."):
            pt.mode_time(-np.random.rand(), np.random.rand())

    def test_kt_validity_zero(self):
        with pytest.raises(ValueError, match=r".*needs*."):
            pt.mode_time(np.random.rand(), 0.)

    def test_kt_validity_minus(self):
        with pytest.raises(ValueError, match=r".*needs*."):
            pt.mode_time(np.random.rand(), -np.random.rand())

    def test_bch_point(self):
        expected = 3.61330873300359
        found = pt.mode_time(1.7, 2.4)
        assert np.isclose(expected, found)


class TestGetPhitkRad:
    def test_tk_validity_zero(self):
        with pytest.raises(ValueError, match=r".*needs*."):
            pt.get_phitk_rad(-np.random.rand())

    def test_small_tk(self):
        expected = (1., -1.99999999999986e-7, -0.1999999999999571)
        found = pt.get_phitk_rad(1e-6)
        assert np.allclose(expected, found)

    def test_border_tk(self):
        below = pt.get_phitk_rad(5e-3 - sys.float_info.epsilon)
        above = pt.get_phitk_rad(5e-3 + sys.float_info.epsilon)
        assert np.allclose(below, above, rtol=1e-5, atol=1e-5)

    def test_medium_tk(self):
        expected = (0.840887175519203, -0.230050549274149, -0.1330393315987455)
        found = pt.get_phitk_rad(1.3)
        assert np.allclose(expected, found)

    def test_large_tk(self):
        expected = (2.85637442046038e-8, -9.17700078990895e-9,
                    -2.856007340428782e-8)
        found = pt.get_phitk_rad(1e4)
        assert np.allclose(expected, found)


class TestGetDeltaCdmRad:
    def test_tk_validity_zero(self):
        with pytest.raises(ValueError, match=r".*needs*."):
            pt.get_delta_cdm_rad(-np.random.rand())

    def test_small_tk(self):
        expected = (1.5, 2.1e-6)
        found = pt.get_delta_cdm_rad(1e-6)
        assert np.allclose(expected, found)

    def test_border_tk(self):
        below = pt.get_delta_cdm_rad(5e-3 - 2.*sys.float_info.epsilon)
        above = pt.get_delta_cdm_rad(5e-3 + 2.*sys.float_info.epsilon)
        assert np.allclose(below, above, rtol=1e-6, atol=1e-6)

    def test_medium_tk(self):
        expected = (3.19269816751044, 2.48185361897349)
        found = pt.get_delta_cdm_rad(1.3)
        assert np.allclose(expected, found)

    def test_large_tk(self):
        expected = (83.5880041605087, 0.000900055036297370)
        found = pt.get_delta_cdm_rad(1e4)
        assert np.allclose(expected, found)

