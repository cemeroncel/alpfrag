"""Script to run the whole pipeline."""
import configparser
import alpfrag.potentials as potential
import alpfrag.models as models
import natpy as nat
import alpfrag.cosmology as cosmology
import alpfrag.background as bg
import alpfrag.perturbations as pt
import numpy as np
from astropy.units import Quantity
from pathlib import Path
import h5py


class Pipeline:
    def __init__(self, config_file: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.cosmo = cosmology.Cosmology()

        # Validate and set the ALP mass
        try:
            self.m0_eV = self.config['model'].getfloat('m0')
        except ValueError:
            print('`m0` should be a float!')

        # Define the potential object
        pot_desc = self.config['model'].get('potential', 'free')
        if pot_desc == 'free':
            self.pot = potential.Free()
            self.theta_ini = 1.
        elif pot_desc == 'periodic':
            err = '`theta_ini` needs to be defined for this potential.'
            assert 'theta_ini' in self.config['model'], err
            try:
                self.theta_ini = self.config['model'].getfloat('theta_ini')
            except ValueError:
                print('`theta_ini` should be a float!')
            self.pot = potential.Periodic()
        elif pot_desc == 'non-periodic':
            err = '`theta_ini` needs to be defined for this potential.'
            assert 'theta_ini' in self.config['model'], err
            try:
                self.theta_ini = self.config['model'].getfloat('theta_ini')
            except ValueError:
                print('`theta_ini` should be a float!')
            err = '`p` needs to be defined for this potential.'
            assert 'p' in self.config['model'], err
            try:
                p = self.config['model'].getfloat('p')
            except ValueError:
                print('`p` should be a flot.')
            self.pot = potential.NonPeriodic(p)
        else:
            raise ValueError(f"Unknown potential {pot_desc}!")

        # Define the model
        self.model = models.StandardALP(self.theta_ini, self.pot,
                                        m0=self.m0_eV*nat.eV,
                                        cosmo=self.cosmo)

    def _run_background(self):
        ti = self.config['simulation'].getfloat('tm_ini', 1e-3)
        tf = self.config['simulation'].getfloat('tf_ini', 2e3)
        verbose = self.config['verbose'].getboolean('bg_verbose')
        self.model.bg_field_evolve(ti, tf, stop_after_convergence=False,
                                   verbose=verbose)

    def _evolve_single_mode(self, kt):
        # Background solution dict for convenience
        bgf = self.model.bg_field
        tf = bgf[-1].t[-1]

        # Run from the start until the onset of oscillations
        early = self.model.pt_mode_evolve_wo_scaling(kt, 0., 0., bgf[1])

        # Run from the onset of oscillations until the start of the
        # density contrast averaging
        x0, v0 = bg.scaled_initial_conditions(early.y[0][-1],
                                              early.y[1][-1],
                                              early.t[-1])
        t_avg = self.config['simulation'].getfloat('tm_avg', 1e3)
        intermediate = self.model.pt_mode_evolve_w_scaling(kt, x0, v0, bgf[1],
                                                           tf=t_avg,
                                                           t_eval=[t_avg])

        # Run for the density contrast averaging
        h = self.config['simulation'].getfloat('tm_avg_stepsize', 1e-3)
        late = self.model.pt_mode_evolve_w_scaling(kt, intermediate.y[0][-1],
                                                   intermediate.y[1][-1],
                                                   bgf[1],
                                                   ti=intermediate.t[-1],
                                                   tf=tf,
                                                   t_eval=np.arange(t_avg, tf, h))

        # Get the density contrast evolution for averaging
        dc = self.model.get_dc_evolution(kt, late, bgf[-1])

        # Average the density contrast
        t_avg, dc_avg, dc_der_avg = pt.get_avg_dc_evolution(late.t, dc)

        # Get the simulation time at matching
        match_index = self.config['simulation'].getint('match_index', -2)
        tm_match = t_avg[match_index]

        # Convert tm_match to redshift
        z_start = self.model.convert_tm_to_redshift(tm_match)

        # Perform the WKB solution. Note that we need to convert the
        # tm derivative to the ln(y) derivative where y=a/aeq
        z_end = self.config['simulation'].getfloat('z_final', 99)
        wkb = pt.dc_eval_latetime(kt, dc_avg[match_index],
                                  2*t_avg[match_index]*dc_der_avg[match_index],
                                  self.cosmo, z_start, z_end)

        # Convert the momentum variables
        kMpc, khMpc = self.convert_kt_to_kMpc(kt)

        return {
            'kt': kt,
            'kMpc': kMpc,
            'khMpc': khMpc,
            'z_arr': (1. + self.cosmo.zeq)*np.exp(-wkb.t) - 1.,
            'delta': wkb.y[0],
            'sol': wkb
        }

    def run():
        pass


class SMMPipeline:
    def __init__(self, m0: Quantity, pot: potential.Potential,
                 cosmo: cosmology.Cosmology, theta_ini: float | None = None,
                 tm_ini: float = 1e-3, tm_fin: float = 2e3, tm_avg: float = 1e3,
                 bg_verbose: bool = True, tm_avg_stepsize: float = 1e-3,
                 match_index: int = -2, z_end: float = 99., kt_min: float = 0.05,
                 kt_max: float = 3., kt_num: int = 100, kt_sampling: str = 'log',
                 pt_verbose: bool = True):
        self.m0 = m0
        self.pot = pot
        self.cosmo = cosmo

        if theta_ini is None:
            err = '`theta_ini` needs to be defined for this potential.'
            assert isinstance(self.pot, potential.Free), err
            self.theta_ini = 1.
        else:
            self.theta_ini = theta_ini

        self.model = self.create_model()
        self.tm_ini = tm_ini
        self.tm_fin = tm_fin
        self.tm_avg = tm_avg
        self.tm_avg_stepsize = tm_avg_stepsize
        self.bg_verbose = bg_verbose
        self.match_index = match_index
        self.z_end = z_end
        self.kt_min = kt_min
        self.kt_max = kt_max
        self.kt_num = kt_num
        self.kt_sampling = kt_sampling
        self.pt_verbose = pt_verbose

    def create_model(self):
        return models.StandardALP(self.theta_ini, self.pot, self.m0,
                                  self.cosmo)

    def _run_background(self, ti, tf, bg_verbose):
        self.model.bg_field_evolve(ti, tf, stop_after_convergence=False,
                                   verbose=bg_verbose)

    def _evolve_single_mode(self, kt):
        # Background solution dict for convenience
        bgf = self.model.bg_field
        tf = bgf[-1].t[-1]

        # Run from the start until the onset of oscillations
        early = self.model.pt_mode_evolve_wo_scaling(kt, 0., 0., bgf[0])

        # Run from the onset of oscillations until the start of the
        # density contrast averaging
        x0, v0 = bg.scaled_initial_conditions(early.y[0][-1],
                                              early.y[1][-1],
                                              early.t[-1])
        intermediate = self.model.pt_mode_evolve_w_scaling(kt, x0, v0, bgf[1],
                                                           tf=self.tm_avg,
                                                           t_eval=[self.tm_avg])

        # Run for the density contrast averaging
        t_eval = np.arange(self.tm_avg, self.tm_fin, self.tm_avg_stepsize)
        late = self.model.pt_mode_evolve_w_scaling(kt, intermediate.y[0][-1],
                                                   intermediate.y[1][-1],
                                                   bgf[1],
                                                   ti=intermediate.t[-1],
                                                   tf=tf,
                                                   t_eval=t_eval)

        # Get the density contrast evolution for averaging
        dc = self.model.get_dc_evolution(kt, late, bgf[-1])

        # Average the density contrast
        t_avg, dc_avg, dc_der_avg = pt.get_avg_dc_evolution(late.t, dc)

        # Get the simulation time at matching
        tm_match = t_avg[self.match_index]

        # Convert tm_match to redshift
        z_start = self.model.convert_tm_to_redshift(tm_match)

        # Perform the WKB solution. Note that we need to convert the
        # tm derivative to the ln(y) derivative where y=a/aeq
        wkb = pt.dc_eval_latetime(kt, dc_avg[self.match_index],
                                  2*t_avg[self.match_index]*dc_der_avg[self.match_index],
                                  self.cosmo, z_start, self.z_end)

        # Convert the momentum variables
        kMpc, khMpc = self.model.convert_kt_to_kMpc(kt)

        return {
            'kt': kt,
            'kMpc': kMpc,
            'khMpc': khMpc,
            'z_arr': (1. + self.cosmo.zeq)*np.exp(-wkb.t) - 1.,
            'delta': wkb.y[0],
            'sol': wkb
        }

    def _run_perturbations(self, kt_min: float, kt_max: float, kt_num: int,
                           pt_verbose: bool, kt_sampling: str):
        if kt_sampling == 'log':
            self.kt_list = np.geomspace(kt_min, kt_max, kt_num)
        elif kt_sampling == 'linear':
            self.kt_list = np.linspace(kt_min, kt_max, kt_num)
        else:
            raise ValueError('`kt_sampling` is either "log" or "linear".')
        self.kMpc_list = np.zeros(len(self.kt_list))
        self.khMpc_list = np.zeros(len(self.kt_list))
        self.delta_list = np.zeros(len(self.kt_list))
        self.delta_list_cdm = np.zeros(len(self.kt_list))
        self.pt_sol_dicts = []

        for i, kt in enumerate(self.kt_list):
            if pt_verbose:
                print(f"Solving the mode {i + 1} of {len(self.kt_list)}.")
            sol_dict = self._evolve_single_mode(kt)
            self.kMpc_list[i] = sol_dict['kMpc']
            self.khMpc_list[i] = sol_dict['khMpc']
            self.delta_list[i] = sol_dict['delta'][-1]
            self.pt_sol_dicts.append(sol_dict)

            # CDM stuff
            # Convert to tk
            tk = pt.mode_time(2e3, kt)
            # Get the delta and the derivative
            delta, delta_tk_der = pt.get_delta_cdm_rad(tk)
            # Convert delta_tk_der to delta_lny_der
            delta_lny_der = tk*delta_tk_der
            # Get z_match
            z_match = self.model.convert_tm_to_redshift(2e3)
            # Solve the equation
            sol = pt.dc_eval_latetime(0., delta, delta_lny_der,
                                      self.cosmo, z_match,
                                      z_end=self.z_end)
            # Get the quantitity we are interested
            self.delta_list_cdm[i] = sol.y[0][-1]

    def _evolve_cdm_mode(self, kt, tk, z_end):
        delta, delta_der = pt.get_delta_cdm_rad(tk)
        delta_der_lny = tk*delta_der
        tm = 0.75*((tk/kt)**2)
        z_start = self.model.convert_tm_to_redshift(tm)
        wkb = pt.dc_eval_latetime(0., delta, delta_der_lny, self.cosmo,
                                  z_start, z_end)
        return {
            'z_arr': (1. + self.cosmo.zeq)*np.exp(-wkb.t) - 1.,
            'delta': wkb.y[0],
            'sol': wkb
        }

    def _run_cdm_perturbations(self, z_end, tk):
        self.delta_cdm_list = np.zeros(len(self.kt_list))
        self.cdm_sol_dicts = []

        for i, kt in enumerate(self.kt_list):
            sol_dict = self._evolve_cdm_mode(kt, tk, z_end)
            self.delta_cdm_list[i] = sol_dict['delta'][-1]
            self.cdm_sol_dicts.append(sol_dict)

    def run(self):
        self._run_background(self.tm_ini, self.tm_fin, self.bg_verbose)
        self._run_perturbations(self.kt_min, self.kt_max, self.kt_num,
                                self.pt_verbose, self.kt_sampling)
        self._run_cdm_perturbations(self.z_end, self.tm_fin)

    def save(self, savefile: Path | str):
        with h5py.File(savefile, 'a') as f:
            f.create_dataset('kt', data=self.kt_list)
            f.create_dataset('kMpc', data=self.kMpc_list)
            f.create_dataset('khMpc', data=self.khMpc_list)
            f.create_dataset('delta', data=self.delta_list)

            f.attrs['mass_in_eV'] = nat.convert(self.m0, nat.eV).value
            f.attrs['potential'] = str(self.model.pot)
            f.attrs['theta_initial'] = self.theta_ini
            f.attrs['zeq'] = self.cosmo.zeq
            f.attrs['tm_ini'] = self.tm_ini
            f.attrs['tm_fin'] = self.tm_fin
            f.attrs['tm_avg_stepsize'] = self.tm_avg_stepsize
            f.attrs['match_index'] = self.match_index
            f.attrs['z_end'] = self.z_end
            f.attrs['tm_zero_cross'] = self.model.t_zero_cross
            f.attrs['tm_first_min'] = self.model.t_first_min

            f.close

        print(f"Results saved in {str(savefile)}.")
