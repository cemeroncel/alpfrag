"""Routines for cosmological calculations."""
from importlib.resources import files
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


# Coefficients for the fitting functions to obtain the effective
# degrees of freedom at high temperatures. Taken from the Appendix C
# of https://arxiv.org/pdf/1803.01038.pdf
_FIT_A = np.zeros(12)
_FIT_B = np.zeros(12)
_FIT_C = np.zeros(12)
_FIT_D = np.zeros(12)

_FIT_A[0] = 1.
_FIT_A[1] = 1.11724
_FIT_A[2] = 3.12672e-1
_FIT_A[3] = -4.68049e-2
_FIT_A[4] = -2.65004e-2
_FIT_A[5] = -1.19760e-3
_FIT_A[6] = 1.82812e-4
_FIT_A[7] = 1.36436e-4
_FIT_A[8] = 8.55051e-5
_FIT_A[9] = 1.22840e-5
_FIT_A[10] = 3.82259e-7
_FIT_A[11] = -6.87035e-9

_FIT_B[0] = 1.43382e-2
_FIT_B[1] = 1.37559e-2
_FIT_B[2] = 2.92108e-3
_FIT_B[3] = -5.38533e-4
_FIT_B[4] = -1.62496e-4
_FIT_B[5] = -2.87906e-5
_FIT_B[6] = -3.84278e-6
_FIT_B[7] = 2.78776e-6
_FIT_B[8] = 7.40342e-7
_FIT_B[9] = 1.17210e-7
_FIT_B[10] = 3.72499e-9
_FIT_B[11] = -6.74107e-11

_FIT_C[0] = 1.
_FIT_C[1] = 6.07896e-1
_FIT_C[2] = -1.54485e-1
_FIT_C[3] = -2.24034e-1
_FIT_C[4] = -2.82147e-2
_FIT_C[5] = 2.90620e-2
_FIT_C[6] = 6.86778e-3
_FIT_C[7] = -1.00005e-3
_FIT_C[8] = -1.69104e-4
_FIT_C[9] = 1.06301e-5
_FIT_C[10] = 1.69528e-6
_FIT_C[11] = -9.33311e-8

_FIT_D[0] = 7.07388e1
_FIT_D[1] = 9.18011e1
_FIT_D[2] = 3.31892e1
_FIT_D[3] = -1.39779
_FIT_D[4] = -1.52558
_FIT_D[5] = -1.97857e-2
_FIT_D[6] = -1.60146e-1
_FIT_D[7] = 8.22615e-5
_FIT_D[8] = 2.02651e-2
_FIT_D[9] = -1.82134e-5
_FIT_D[10] = 7.83943e-5
_FIT_D[11] = 7.13518e-5

_M_E: float = 511e-6
_M_MU: float = 0.1056
_M_PI0: float = 0.135
_M_PIPM: float = 0.140
_M1: float = 0.5
_M2: float = 0.77
_M3: float = 1.2
_M4: float = 2.


def effective_dof_interp(errorbars=False) -> dict:
    """
    Interpolating functions for the effective degrees of freedom.

    Returns interpolating functions of effective degrees of freedom in the
    energy density and effective degrees of freedom in the entropy assuming
    the Standard Model. If `errorbars` is `True`, then it also returns the
    interpolating functions for the errorbars. The variable for all the
    functions is ln(T/GeV)

    Parameters
    ----------
    errorbars : boolean, optional
        Whether to return interpolating functions for the errorbars.
        The default is False.

    Returns
    -------
    dict
        A dict which contains the interpolating function for the effective
        degrees of freedom in the energy density, and in the entropy as a
        function of ln(T/GeV). If `errorbars` is `True` then it also
        contains functions for the errorbars in the same order.
        - `grho` : Interpolating function for

    Notes
    -----
    The data for the interpolating functions are taken from the tabulated
    values of [1]_. The minimum and maximum values of the tabulated data
    are 2e-6 GeV and 10e17 GeV respectively.

    .. [1]  e-Print: 1803.01038 [hep-ph]
    """
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

    # Creating interpolating functions
    g_rho_interp = interp1d(ln_T_GeV, g_rho, kind="cubic",
                            bounds_error=False,
                            fill_value=(g_rho[0], g_rho[-1]))
    gs_interp = interp1d(ln_T_GeV, gs, kind="cubic",
                         bounds_error=False,
                         fill_value=(gs[0], gs[-1]))

    # Repeat the steps if errorbars is True
    if errorbars:
        g_rho_err = df["g_rho_err"].to_numpy()
        gs_err = df["gs_err"].to_numpy()
        g_rho_err_interp = interp1d(ln_T_GeV, g_rho_err, kind="cubic",
                                    bounds_error=False,
                                    fill_value=(0, 0))
        gs_err_interp = interp1d(ln_T_GeV, gs_err, kind="cubic",
                                 bounds_error=False,
                                 fill_value=(0, 0))
        return {
            'grho': g_rho_interp,
            'gs': gs_interp,
            'grho_err': g_rho_err_interp,
            'gs_err': gs_err_interp}

    # Return g_rho and g_s only if errorbars is True
    return {
        'grho': g_rho_interp,
        'gs': gs_interp}


def _grho_fit_high(lnT: float) -> float:
    if lnT < np.log(0.12) or lnT > np.log(1e16):
        raise ValueError("Temperature should be between 120 MeV and 1e16 GeV!")
    numerator_array = np.array([_FIT_A[i]*(lnT**float(i))
                                for i in range(len(_FIT_A))])
    denominator_array = np.array([_FIT_B[i]*(lnT**float(i))
                                  for i in range(len(_FIT_B))])
    return np.sum(numerator_array)/np.sum(denominator_array)


def _gs_fit_high(lnT: float) -> float:
    if lnT < np.log(0.12) or lnT > np.log(1e14):
        raise ValueError("Temperature should be between 120 MeV and 1e14 GeV!")
    grho = _grho_fit_high(lnT)
    numerator_array = np.array([_FIT_C[i]*(lnT**float(i))
                                for i in range(len(_FIT_C))])
    denominator_array = np.array([_FIT_D[i]*(lnT**float(i))
                                  for i in range(len(_FIT_D))])
    return grho/(1. + np.sum(numerator_array)/np.sum(denominator_array))


def _poly_fun(x: float, a: list[float]) -> float:
    return np.sum(np.array([a[i]*(x**float(i)) for i in range(len(a))]))


def _f_rho(x: float) -> float:
    a = [1., 1.03757, 0.508630, 0.0893988]
    return np.exp(-1.04855*x)*_poly_fun(x, a)


def _b_rho(x: float) -> float:
    a = [1., 1.03317, 0.398264, 0.0648056]
    return np.exp(-1.04855*x)*_poly_fun(x, a)


def _f_s(x: float) -> float:
    a = [1., 1.03400, 0.456426, 0.0595248]
    return np.exp(-1.04190*x)*_poly_fun(x, a)


def _b_s(x: float) -> float:
    a = [1., 1.03397, 0.342548, 0.0506182]
    return np.exp(-1.03365*x)*_poly_fun(x, a)


def _sfit(x: float) -> float:
    a = [1., 1.034, 0.456426, 0.0595249]
    return 1. + 1.75*np.exp(-1.0419*x)*_poly_fun(x, a)


def _grho_fit_low(lnT: float) -> float:
    if lnT >= np.log(0.12):
        raise ValueError("Temperature should be lower than 120 MeV!")
    T = np.exp(lnT)
    return (2.030
            + 1.353*_sfit(_M_E/T)**(4./3.)
            + 3.495*_f_rho(_M_E/T)
            + 3.446*_f_rho(_M_MU/T)
            + 1.05*_b_rho(_M_PI0/T)
            + 2.08*_b_rho(_M_PIPM/T)
            + 4.165*_b_rho(_M1/T)
            + 30.55*_b_rho(_M2/T)
            + 89.4*_b_rho(_M3/T)
            + 8209.*_b_rho(_M4/T)
            )


def _gs_fit_low(lnT: float) -> float:
    if lnT >= np.log(0.12):
        raise ValueError("Temperature should be lower than 120 MeV!")
    T = np.exp(lnT)
    return (2.008
            + 1.923*_sfit(_M_E/T)
            + 3.442*_f_s(_M_E/T)
            + 3.468*_f_s(_M_MU/T)
            + 1.034*_b_s(_M_PI0/T)
            + 2.068*_b_s(_M_PIPM/T)
            + 4.16*_b_s(_M1/T)
            + 30.55*_b_s(_M2/T)
            + 90.*_b_s(_M3/T)
            + 6209.*_b_s(_M4/T)
            )


def grho_fit(lnT: float) -> float:
    if lnT > np.log(1e14):
        return _grho_fit_high(np.log(1e14))
    elif (lnT <= np.log(1e16)) and (lnT >= np.log(0.12)):
        return _grho_fit_high(lnT)
    elif (lnT < np.log(0.12)) and (lnT > np.log(1e-5)):
        return _grho_fit_low(lnT)
    else:
        return _grho_fit_low(np.log(1e-5))


def gs_fit(lnT: float) -> float:
    if lnT > np.log(1e14):
        return _gs_fit_high(np.log(1e14))
    elif (lnT <= np.log(1e14)) and (lnT >= np.log(0.12)):
        return _gs_fit_high(lnT)
    elif (lnT < np.log(0.12)) and (lnT > np.log(1e-5)):
        return _gs_fit_low(lnT)
    else:
        return _gs_fit_low(np.log(1e-5))
