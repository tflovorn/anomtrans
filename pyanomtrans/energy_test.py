import unittest
from pyanomtrans.grid_basis import kmBasis
from pyanomtrans.square_tb_Hamiltonian import SquareTBHamiltonian
from pyanomtrans.energy import get_energies

class TestGetEnergies(unittest.TestCase):
    def test_get_energies(self):
        Nk = [8, 8]
        Nbands = 1
        kmb = kmBasis(Nk, Nbands)

        t = 1.0
        tp = -0.3
        H = SquareTBHamiltonian(t, tp, Nk)

        Ekm = get_energies(kmb, H)

        for ikm in range(kmb.end_ikm):
            ikm_comps = kmb.decompose(ikm)
            energy = H.energy(ikm_comps)
            self.assertEqual(Ekm[ikm], energy)

if __name__ == "__main__":
    unittest.main()
