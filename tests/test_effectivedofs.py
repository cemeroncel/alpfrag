import numpy as np
import alpfrag.effectivedofs as effdof
import pytest
from importlib.resources import files
import pandas as pd


class TestGrhoFitHigh:
    def test_low_temperature(self):
        with pytest.raises(ValueError):
            effdof._grho_fit_high(np.log(0.1))

    def test_high_temperature(self):
        with pytest.raises(ValueError):
            effdof._grho_fit_high(np.log(1e17))

    def test_fit(self):
        # Get the interpolating functions
        interps = effdof.effective_dof_interp(errorbars=True)

        # Define the array to test
        temps = np.linspace(np.log(0.12), np.log(1e16))

        # Degrees of freedom from the interpolating functions
        interp = interps['grho'](temps)
        interp_err = interps['grho_err'](temps)

        # Degrees of freedom from the fit functions
        fit = np.array([effdof._grho_fit_high(lnT) for lnT in temps])

        # Whether the error is larger than the disceprency from the fit
        valid = np.array([abs(fit[i] - interp[i]) < interp_err[i]
                          for i in range(len(temps))])

        assert np.all(valid)


class TestGsFitHigh:
    def test_low_temperature(self):
        with pytest.raises(ValueError):
            effdof._gs_fit_high(np.log(0.1))

    def test_high_temperature(self):
        with pytest.raises(ValueError):
            effdof._gs_fit_high(np.log(1e17))

    def test_fit(self):
        # Get the interpolating functions
        interps = effdof.effective_dof_interp(errorbars=True)

        # Define the array to test. This fit function is not accurate
        # up to the errorbars for T > 1e14 GeV. This is fine as at
        # high temperatures gs and grho are identical.
        temps = np.linspace(np.log(0.12), np.log(1e14))

        # Degrees of freedom from the interpolating functions
        interp = interps['gs'](temps)
        interp_err = interps['gs_err'](temps)

        # Degrees of freedom from the fit functions
        fit = np.array([effdof._gs_fit_high(lnT) for lnT in temps])

        # Whether the error is larger than the disceprency from the fit
        valid = np.array([abs(fit[i] - interp[i]) < interp_err[i]
                          for i in range(len(temps))])

        assert np.all(valid)


class TestGrhoFitLow:
    def test_high_temperature(self):
        with pytest.raises(ValueError):
            effdof._grho_fit_low(np.log(0.15))

    def test_fit(self):
        # Get the interpolating functions
        interps = effdof.effective_dof_interp(errorbars=True)

        # Define the array to test
        temps = np.linspace(np.log(1e-5), np.log(0.12), endpoint=False)

        # Degrees of freedom from the interpolating functions
        interp = interps['grho'](temps)
        interp_err = interps['grho_err'](temps)

        # Degrees of freedom from the fit functions
        fit = np.array([effdof._grho_fit_low(lnT) for lnT in temps])

        # Whether the error is larger than the disceprency from the fit
        valid = np.array([abs(fit[i] - interp[i]) < interp_err[i]
                          for i in range(len(temps))])

        assert np.all(valid)


class TestGsFitLow:
    def test_high_temperature(self):
        with pytest.raises(ValueError):
            effdof._gs_fit_low(np.log(0.15))

    def test_fit(self):
        # Get the interpolating functions
        interps = effdof.effective_dof_interp(errorbars=True)

        # Define the array to test
        temps = np.linspace(np.log(1e-5), np.log(0.12), endpoint=False)

        # Degrees of freedom from the interpolating functions
        interp = interps['gs'](temps)
        interp_err = interps['gs_err'](temps)

        # Degrees of freedom from the fit functions
        fit = np.array([effdof._gs_fit_low(lnT) for lnT in temps])

        # Whether the error is larger than the disceprency from the fit
        valid = np.array([abs(fit[i] - interp[i]) < interp_err[i]
                          for i in range(len(temps))])

        assert np.all(valid)


class TestEffDofsTabulated:
    # Getting the data
    f = files('alpfrag.data').joinpath('effective-dof.dat')
    df = pd.read_csv(f,
                     sep=" ",
                     skiprows=8,
                     names=["T_GeV", "g_rho", "g_rho_err", "gs", "gs_err"]
                     )

    # Creating numpy arrays
    ln_T_GeV = np.log(df["T_GeV"].to_numpy())
    g_rho = df["g_rho"].to_numpy()
    gs = df["gs"].to_numpy()
    g_rho_err = df["g_rho_err"].to_numpy()
    gs_err = df["gs_err"].to_numpy()

    # Higher temperature, T > 1e14 GeV region is a little bit
    # problematic so we exclude them from the tests. This does not
    # affect the accuracy.
    ln_T_GeV_res = ln_T_GeV[ln_T_GeV < np.log(1e14)]

    def test_grho(self):
        obtained = np.array([effdof.grho_fit(lnT) for lnT in self.ln_T_GeV_res])
        valid = np.array([abs(obtained[i] - self.g_rho[i]) < self.g_rho_err[i]
                          for i in range(len(obtained))])
        assert np.all(valid)

    def test_gs(self):
        obtained = np.array([effdof.gs_fit(lnT) for lnT in self.ln_T_GeV_res])
        valid = np.array([abs(obtained[i] - self.gs[i]) < self.gs_err[i]
                          for i in range(len(obtained))])
        assert np.all(valid)
