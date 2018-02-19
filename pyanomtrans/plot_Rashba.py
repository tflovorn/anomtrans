import argparse
import os.path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pyanomtrans.grid_basis import kmBasis, km_at
from pyanomtrans.plot_2d_bz import extract_sorted_data, make_k_list, plot_2d_bz_slice

def list_to_array(kmb, vals):
    dim = len(kmb.Nk)
    shape = [kmb.Nk[d] for d in range(dim)]
    shape.append(kmb.Nbands)

    arr = np.zeros(shape)
    for ikm, v in enumerate(vals):
        k, m = kmb.decompose(ikm)

        # TODO - is there a general way to write this?
        # arr[...] doesn't take a list.
        if dim == 1:
            arr[k[0], m] = v
        elif dim == 2:
            arr[k[0], k[1], m] = v
        elif dim == 3:
            arr[k[0], k[1], k[2], m] = v
        else:
            raise ValueError("unimplemented dim")

    return arr

def array_to_list(kmb, arr, band_index=True):
    ls = [0.0] * kmb.end_ikm
    for ikm in range(kmb.end_ikm):
        k, m = kmb.decompose(ikm)
        if band_index:
            ls[ikm] = arr[k[0], k[1], m]
        else:
            ls[ikm] = arr[k[0], k[1]]

    return ls

def plot_bz(prefix, fdata):
    keys = ['s_x', 'js_sz_vx_intrinsic', 'js_sz_vx_extrinsic',
            's_y', 'js_sz_vy_intrinsic', 'js_sz_vy_extrinsic']

    only = None
    kmb, sorted_data = extract_sorted_data(only, fdata, only_list=keys)
    Nk, Nbands = kmb.Nk, kmb.Nbands

    all_k0s, all_k1s = make_k_list(kmb, sorted_data['k_comps'])
    kmb_oneband = kmBasis(Nk, 1)

    assert(len(Nk) == 2)
    assert(Nbands == 2)

    titles = [r'$\langle s_x \rangle / E_x$', r'$\sigma^{s_z, int}_{xx}$',
              r'$\sigma^{s_z, ext}_{xx}$',
              r'$\langle s_y \rangle / E_x$', r'$\sigma^{s_z, int}_{xy}$',
              r'$\sigma^{s_z, ext}_{xy}$']
    titles_units = [r'$\hbar$', r'$e$', r'$e$',
                    r'$\hbar$', r'$e$', r'$e$']

    for key, title, title_units in zip(keys, titles, titles_units):
        for mu_index, val_list in enumerate(sorted_data[key]):
            val_arr = list_to_array(kmb, val_list)
            val_band_sum = np.sum(val_arr, axis=2)

            min_val, max_val = np.amin(val_band_sum), np.amax(val_band_sum)
            max_abs = max([abs(min_val), abs(max_val)])
            scale_power = math.floor(math.log10(max_abs))
            scale = 10.0**scale_power
            val_band_sum /= scale

            title_scale_part = r"$\times 10^{" + str(-int(scale_power)) + r"}$"
            full_title = "{} [{} {}]".format(title, title_scale_part, title_units)

            val_band_sum_list = array_to_list(kmb_oneband, val_band_sum, band_index=False)

            plot_prefix = "{}_{}_band_sum_mu_{}".format(prefix, key, str(mu_index))
            plot_2d_bz_slice(plot_prefix, full_title, all_k0s, all_k1s, val_band_sum_list)
            # TODO: increase font size on titles.

def _main():
    parser = argparse.ArgumentParser("Plot data on the 2D Brillouin zone, or slices of the 3D zone",
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
