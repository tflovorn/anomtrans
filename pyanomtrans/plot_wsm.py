import argparse
import os.path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pyanomtrans.grid_basis import kmBasis
from pyanomtrans.plot_2d_bz import extract_sorted_data, make_k_list, plot_2d_bz_slice
from pyanomtrans.plot_Rashba import list_to_array, array_to_list

def plot_bz(prefix, fdata):
    keys = ['current_S_B_comp', 'current_xi_B_comp']

    only = None
    kmb, sorted_data = extract_sorted_data(only, fdata, only_list=keys)
    Nk, Nbands = kmb.Nk, kmb.Nbands

    all_k0s, all_k1s = make_k_list(kmb, sorted_data['k_comps'])
    kmb_2d_oneband = kmBasis([Nk[0], Nk[1]], 1)

    assert(len(Nk) == 3)
    assert(Nbands == 2)

    titles = [r'$\int dk_z \, \langle j^S_{z,+} \rangle / B_z$',
              r'$\int dk_z \, \langle j^{\xi}_{z,+} \rangle / B_z$']
    titles_units = [r'$e^2 \Delta \left(\frac{\Delta}{\hbar v_F}\right)^{-2}$',
                    r'$e^2 \Delta \left(\frac{\Delta}{\hbar v_F}\right)^{-2}$']
    xlabel = r"$k_x$ [$\frac{\Delta}{\hbar v_F}$]"
    ylabel = r"$k_y$ [$\frac{\Delta}{\hbar v_F}$]"

    for key, title, title_units in zip(keys, titles, titles_units):
        for mu_index, val_list in enumerate(sorted_data[key]):
            val_arr = list_to_array(kmb, val_list)

            # `val_arr` is already normalized according to k-space sampling
            # region size and sampling density. Do not need to multiply
            # by kmb.k_step(2) here for average. However, we want to display
            # (kx, ky) points with the proper metric such that if we integrated
            # (instead of summed), we get the right answer - so we need to
            # divide by kmb.k_step(0) * kmb.k_step(1) to remove this factor.
            kxy_area = kmb.k_step(0) * kmb.k_step(1)
            val_kz_avg_band_sum = np.sum(val_arr, axis=(2, 3)) / kxy_area

            min_val, max_val = np.amin(val_kz_avg_band_sum), np.amax(val_kz_avg_band_sum)
            max_abs = max([abs(min_val), abs(max_val)])
            scale_power = math.floor(math.log10(max_abs))
            scale = 10.0**scale_power
            val_kz_avg_band_sum /= scale

            if scale_power != 0:
                title_scale_part = r"$\times 10^{" + str(int(scale_power)) + r"}$"
                full_title = "{} [{} {}]".format(title, title_scale_part, title_units)
            else:
                full_title = "{} [{}]".format(title, title_units)

            val_band_sum_list = array_to_list(kmb_2d_oneband, val_kz_avg_band_sum, band_index=False)

            plot_prefix = "{}_{}_band_sum_mu_{}".format(prefix, key, str(mu_index))

            plot_2d_bz_slice(plot_prefix, full_title, all_k0s, all_k1s, val_band_sum_list,
                    xlabel=xlabel, ylabel=ylabel)

def plot_series(prefix, fdata):
    mus = fdata['mus']

    #chiralities = fdata['chirality']
    chiralities = [1] * len(mus)
    assert(all([c == chiralities[0] for c in chiralities]))

    current_Ss = fdata['_series_current_S_B']
    current_xis = fdata['_series_current_xi_B']

    mu_extra = 10
    interpolate_mus = []
    interpolate_chiralities = []
    for mu_index, mu in enumerate(mus[:-1]):
        if mu_index == len(mus) - 2:
            endpoint = True
        else:
            endpoint = False

        mu_next = mus[mu_index + 1]

        interpolate_mus.extend(np.linspace(mu, mu_next, mu_extra, endpoint=endpoint))
        interpolate_chiralities.extend([chiralities[0]] * mu_extra)

    current_S_expecteds = [(2/(3 * 4 * math.pi**2)) * chirality * mu
            for chirality, mu in zip(interpolate_chiralities, interpolate_mus)]

    S_title = r'$\langle j^S_z(\tilde{\mu}_{\nu}) \rangle / B_z$ [$e^2 \Delta$]'
    plt.title(S_title, fontsize='large')

    S_xlabel = r'$\tilde{\mu}_{\nu}$ [$\Delta$]'
    plt.xlabel(S_xlabel, fontsize='large')

    plt.plot(interpolate_mus, current_S_expecteds, 'g-')
    plt.plot(mus, current_Ss, 'k.')

    S_plot_path = "{}_j_S".format(prefix)
    plt.savefig("{}.png".format(S_plot_path), bbox_inches='tight', dpi=500)
    plt.clf()

    assert(len(mus) % 2 == 0)
    # j^{\xi}(\mu) - j^{\xi}(-\mu) = j^S(\mu)
    current_xi_mu_diffs = []

    for mu_index, mu in enumerate(mus):
        mu_minus_index = len(mus) - mu_index - 1
        mu_minus = mus[mu_minus_index]
        eps = 1e-9
        assert(abs(mu + mu_minus) < eps)

        xi_diff = current_xis[mu_index] - current_xis[mu_minus_index]
        current_xi_mu_diffs.append(xi_diff)

    xi_title = r'$\left[\langle j^{\xi}_z(\tilde{\mu}_{\nu}) \rangle - \langle j^{\xi}_z(-\tilde{\mu}_{\nu}) \rangle\right] / B_z$ [$e^2 \Delta$]'
    plt.title(xi_title, fontsize='large')

    xi_xlabel = S_xlabel
    plt.xlabel(xi_xlabel, fontsize='large')

    plt.plot(interpolate_mus, current_S_expecteds, 'g-')
    plt.plot(mus, current_xi_mu_diffs, 'k.')

    xi_plot_path = "{}_j_xi_diff".format(prefix)
    plt.savefig("{}.png".format(xi_plot_path), bbox_inches='tight', dpi=500)
    plt.clf()

def _main():
    parser = argparse.ArgumentParser("Plot WSM CME node data averaged over kz and summed over bands",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for file giving plot data: should be in the form prefix.json")
    parser.add_argument("in_dir", type=str,
            help="Directory containing file giving plot data")
    args = parser.parse_args()

    fpath = os.path.join(args.in_dir, "{}.json".format(args.prefix))
    with open(fpath, 'r') as fp:
        fdata = json.load(fp)

    #plot_bz(args.prefix, fdata)
    plot_series(args.prefix, fdata)

if __name__ == '__main__':
    _main()
