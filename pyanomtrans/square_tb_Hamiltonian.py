from math import pi, cos
from pyanomtrans.grid_basis import km_at

class SquareTBHamiltonian:
    def __init__(self, t, tp, Nk):
        self.t = t
        self.tp = tp
        self.Nk = Nk

    def energy(self, ikm_comps):
        k, m = km_at(self.Nk, ikm_comps)
        
        if m != 0:
            raise ValueError("SquareTBHamiltonian is not defined for Nbands > 1")

        kx, ky = 2*pi*k[0], 2*pi*k[1]

        E = -2*self.t*(cos(kx) + cos(ky)) + 4*self.tp*cos(kx)*cos(ky)
        return E

    def basis_component(self, ikm_comps, i):
        return complex(1.0, 0.0)
