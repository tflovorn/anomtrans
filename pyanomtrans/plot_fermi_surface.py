#import argparse
from math import sqrt
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from grid_basis import kmBasis, km_at
from square_tb_Hamiltonian import SquareTBHamiltonian
from energy import get_energies
from rho0 import make_rho0
from derivative import make_d_dk_recip_csr

def plot_2d_bz_slice(plot_path, title, all_k0s, all_k1s, all_vals):
    xs_set, ys_set = set(), set()
    for x, y in zip(all_k0s, all_k1s):
        xs_set.add(x)
        ys_set.add(y)

    num_xs, num_ys = len(xs_set), len(ys_set)
    C_E = np.array(all_vals).reshape((num_xs, num_ys))

    plt.imshow(C_E, origin='lower', interpolation='none', cmap=cm.viridis)
    plt.axes().set_aspect('auto', 'box')
    plt.colorbar()
    plt.title(title)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    num_k0_ticks, num_k1_ticks = 6, 6

    k0_tick_locs = np.linspace(xmin, xmax, num_k0_ticks)
    k0_tick_vals = np.linspace(min(xs_set), max(xs_set), num_k0_ticks)
    k0_tick_labels = ["{:.1f}".format(x) for x in k0_tick_vals]
    plt.xticks(k0_tick_locs, k0_tick_labels)
    plt.xlabel("$k_0$")

    k1_tick_locs = np.linspace(ymin, ymax, num_k1_ticks)
    k1_tick_vals = np.linspace(min(ys_set), max(ys_set), num_k1_ticks)
    k1_tick_labels = ["{:.1f}".format(x) for x in k1_tick_vals]
    plt.yticks(k1_tick_locs, k1_tick_labels)
    plt.ylabel("$k_1$")

    plt.savefig("{}.png".format(plot_path), bbox_inches='tight', dpi=500)
    plt.clf()

def _main():
    # TODO general interface
    #parser = argparse.ArgumentParser("Plot Fermi surface as shown by |df(Emk)/dk|",
    #        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--system_type', type=str, default='analytic',
    #        choices=['analytic', 'precomputed'],
    #        help="Use an 'analytic' or 'precomputed' H(k), dH/dk")
    #parser.add_argument('--Hamiltonian', type=str, default='SquareTBHamiltonian',
    #        help=("If system_type == 'analytic', this gives the name of the"
    #              "Hamiltonian class; if system_type == 'precomputed', this"
    #              "gives the path to the Hamiltonian file"))
    #parser.add_argument('--param_path', type=str,
    #        default='plot_fermi_surface_default_params.json',
    #        help=("If system_type == 'analytic', this gives the path to a json"
    #        "file containing the parameters to use for the calculation")

    # TODO should the Hamiltonian classes take a dictionary in their constructor
    # to simplify calling here?

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

if __name__ == '__main__':
    _main()
