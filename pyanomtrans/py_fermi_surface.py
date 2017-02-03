from math import sqrt
from grid_basis import kmBasis, km_at
from square_tb_Hamiltonian import SquareTBHamiltonian
from energy import get_energies
from rho0 import make_rho0
from derivative import make_d_dk_recip_csr
from plot_2d_bz import plot_2d_bz_slice

def _main():
    t = 1.0
    tp = -0.3

    Nk = [128, 128]
    Nbands = 1
    assert(Nbands == 1)
    beta = 10.0

    kmb = kmBasis(Nk, Nbands)

    deriv_order = 2
    d_dk = make_d_dk_recip_csr(kmb, deriv_order)

    H = SquareTBHamiltonian(t, tp, Nk)
    Ekm = get_energies(kmb, H)
 
    num_E_Fs = 40
    E_Fs = np.linspace(min(Ekm), max(Ekm), num_E_Fs)

    norm_d_rho0_dks = []
    for E_F in E_Fs:
        rho0_km = np.array(make_rho0(Ekm, beta, E_F))
        d_rho0_dk = []
        for d in range(kmb.k_dim()):
            d_rho0_dk.append(d_dk[d].dot(rho0_km))

        norm_d_rho0_dk = []
        for ikm in range(kmb.end_ikm):
            norm2 = 0.0
            for d in range(kmb.k_dim()):
                norm2 += d_rho0_dk[d][ikm]**2

            norm_d_rho0_dk.append(sqrt(norm2))

        norm_d_rho0_dks.append(norm_d_rho0_dk)

    all_k0s, all_k1s = [], []
    for ikm in range(kmb.end_ikm):
        # TODO handle Nbands > 1
        # (add contributions at same k from different m?)
        k, m = km_at(kmb.Nk, kmb.decompose(ikm))
        all_k0s.append(k[0])
        all_k1s.append(k[1])

    plot_2d_bz_slice("energy_square_tb", "$E$", all_k0s, all_k1s, Ekm)

    plot_prefix = "fermi_surface_square_tb"
    for E_F_index, E_F in enumerate(E_Fs):
        plot_path = "{}_E_F_{}".format(plot_prefix, E_F_index)
        title = "{}/{}".format(E_F_index, len(E_Fs))
        plot_2d_bz_slice(plot_path, title, all_k0s, all_k1s, norm_d_rho0_dks[E_F_index])

if __name__ == "__main__":
    _main()
