from pyanomtrans.special_functions import fermi_dirac

def make_rho0(energies, beta, E_F):
    def fd(E):
        return fermi_dirac(beta, E - E_F)

    rho0 = list(map(fd, energies))
    return rho0
