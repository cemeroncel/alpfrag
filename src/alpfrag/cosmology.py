"""Module for the cosmological calculations."""

from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.units import Quantity
from alpfrag.effectivedofs import grho_fit, gs_fit, effective_dof_interp
from functools import cached_property
import natpy
import numpy as np
from alpfrag.constants import M_PLANCK_REDUCED
from scipy.optimize import root

# This sets hbar, c, and kB (Boltzmanns's constant) to one
natpy.set_active_units('HEP')


class Cosmology(FlatLambdaCDM):
    def __init__(self,
                 h: float = 0.6737,
                 odm0: float = 0.1198,
                 ob0: float = 0.02233,
                 As: float = 2.1e-9,
                 ns: float = 0.9649,
                 Tcmb0: float = 2.72548,
                 Neff: float = 3.04,
                 use_fit_for_eff_dofs: bool = False
                 ):
        super().__init__(H0=100.*h,
                         Om0=(odm0 + ob0)/(h**2),
                         Tcmb0=Tcmb0,
                         Neff=Neff,
                         m_nu=0.,
                         Ob0=ob0/(h**2))
        self.As = As,
        self.ns = ns

        if use_fit_for_eff_dofs:
            self.gs_fun = gs_fit
            self.grho_fun = grho_fit
        else:
            interp = effective_dof_interp()
            self.gs_fun = interp['gs']
            self.grho_fun = interp['grho']

    def gs(self, lnT: float) -> float:
        return self.gs_fun(lnT)

    def grho(self, lnT: float) -> float:
        return self.grho_fun(lnT)

    @cached_property
    def zeq(self):
        return z_at_value(self.Om, 0.5, zmin=3000., zmax=4000.,
                          bracket=[3800., 3900.]).value

    def z_at_T(self, T: float | Quantity):
        # Convert the CMB temparature today to GeV, since the
        # effective degree of freedom functions have input lnT/GeV
        Tcmb0_GeV = natpy.convert(self.Tcmb0, natpy.GeV).value

        # If T is float, assume that it is in GeV.
        if isinstance(T, Quantity):
            T_GeV = natpy.convert(T, natpy.GeV).value
        else:
            T_GeV = T

        gsT = self.gs(np.log(T_GeV))
        gsT0 = self.gs(np.log(Tcmb0_GeV))

        return ((gsT/gsT0)**(1./3.))*(T_GeV/Tcmb0_GeV) - 1.

    def H_at_T_in_rad(self, T: float | Quantity,
                      res_unit: Quantity | None = None):
        # If T is float, assume that it is in GeV.
        if isinstance(T, Quantity):
            _T = natpy.convert(T, natpy.GeV).value
        else:
            _T = T

        grho = self.grho(np.log(_T))
        res = (np.pi/3.)*np.sqrt(0.1*grho)*(_T**2)/M_PLANCK_REDUCED.value

        if res_unit is not None:
            return natpy.convert(res*natpy.GeV, res_unit)
        else:
            return res

    def inv_H_at_T_in_rad(self, H: float | Quantity,
                          res_unit: Quantity | None = None,
                          T_guess: float | Quantity = 1.):

        if isinstance(H, Quantity):
            _H = natpy.convert(H, natpy.GeV).value
        else:
            _H = H

        if isinstance(T_guess, Quantity):
            _T_guess = natpy.convert(T_guess, natpy.GeV).value
        else:
            _T_guess = T_guess

        def fun(lgT):
            return np.log(self.H_at_T_in_rad(np.exp(lgT))/_H)

        res = np.exp(root(fun, np.log(_T_guess)).x[0])

        if res_unit is not None:
            return natpy.convert(res*natpy.GeV, res_unit)
        else:
            return res
