from sys import float_info
import unittest
from pyanomtrans.special_functions import fermi_dirac, _LN_DBL_MIN

class TestFermiDirac(unittest.TestCase):
    def test_fermi_dirac(self):
        tol = 10.0*float_info.epsilon
        beta = 10.0
        E_below_min = 2*_LN_DBL_MIN/beta
        E_above_max = -E_below_min

        self.assertAlmostEqual(fermi_dirac(beta, E_below_min), 1.0, delta=tol)
        self.assertAlmostEqual(fermi_dirac(beta, E_above_max), 0.0, delta=tol)
        self.assertAlmostEqual(fermi_dirac(beta, 0.0), 0.5, delta=tol)

if __name__ == '__main__':
    unittest.main()
