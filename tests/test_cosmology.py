import numpy as np
import alpfrag.cosmology as cosmology


class TestGenericCosmology:
    def test_z_at_T(self):
        cosmo = cosmology.Cosmology()
        assert np.isclose(cosmo.z_at_T(cosmo.Tcmb0), 0.)

    def test_inv_H_at_T_in_rad_unitless(self):
        cosmo = cosmology.Cosmology(use_fit_for_eff_dofs=False)
        H = cosmo.H_at_T_in_rad(1.)
        print(H)
        obtained = cosmo.inv_H_at_T_in_rad(H)
        assert np.isclose(obtained, 1.)
