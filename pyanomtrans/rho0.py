from pyanomtrans.special_functions import fermi_dirac

def make_rho0(energies, beta, mu):
    def fd(E):
        return fermi_dirac(beta, E - mu)

    rho0 = list(map(fd, energies))
    return rho0
