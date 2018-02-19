import argparse
import os.path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pyanomtrans.grid_basis import kmBasis, km_at
from pyanomtrans.plot_2d_bz import extract_sorted_data, make_k_list, plot_2d_bz_slice
from pyanomtrans.plot_Rashba import list_to_array, array_to_list

def plot_bz(prefix, fdata):
    keys = ['current_S_B_comp', 'current_xi_B_comp']

    only = None
    kmb, sorted_data = extract_sorted_data(only, fdata, only_list=keys)
    Nk, Nbands = kmb.Nk, kmb.Nbands

    all_k0s, all_k1s = make_k_list(kmb, sorted_data['k_comps']) # TODO kx, ky [\Delta / \hbar v_F]
    kmb_2d_oneband = kmBasis([Nk[0], Nk[1]], 1)

    assert(len(Nk) == 3)
    assert(Nbands == 2)

    titles = [r'$\int dk_z \, \langle j^S_{z,+} \rangle / B_z$',
              r'$\int dk_z \, \langle j^{\xi}_{z,+} \rangle / B_z$']
    titles_units = [r'$\Delta \left(\frac{\hbar v_F}{\Delta}\right)^2$',
                    r'$\Delta \left(\frac{\hbar v_F}{\Delta}\right)^2$']

    for key, title, title_units in zip(keys, titles, titles_units):
        for mu_index, val_list in enumerate(sorted_data[key]):
            val_arr = list_to_array(kmb, val_list)
            # TODO: need to include factor of (kz_max - kz_min)
            val_kz_avg_band_sum = np.sum(np.sum(val_arr, axis=2), axis=2) / (Nk[2] - 1)

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
            plot_2d_bz_slice(plot_prefix, full_title, all_k0s, all_k1s, val_band_sum_list)
            # TODO: increase font size on titles.

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

    plot_bz(args.prefix, fdata)

if __name__ == '__main__':
    _main()
