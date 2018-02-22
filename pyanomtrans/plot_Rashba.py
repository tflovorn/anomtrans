import argparse
import os.path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pyanomtrans.grid_basis import kmBasis
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

def shifted_sample(kmb, s0, s1, O):
    '''Given `O[n0, n1]` representing a periodic observable on the k-space
    given by `kmb`, return an `Os[n0, n1]` with its origin shifted by `(s0, s1)`.
    Also return the lists `all_k0s`, `all_k1s` giving the k-point values in the
    shifted space, in `ikm` order.
    '''
    Os = np.zeros(O.shape)
    all_k0s, all_k1s = [], []
    for n0 in range(kmb.Nk[0]):
        for n1 in range(kmb.Nk[1]):
            Os[n0, n1] = O[(n0 + s0) % kmb.Nk[0], (n1 + s1) % kmb.Nk[1]]
            ks, _ = kmb.km_at(([n0 + s0, n1 + s1], 0))
            all_k0s.append(ks[0])
            all_k1s.append(ks[1])

    return Os, all_k0s, all_k1s

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
    titles_units = [r'$n_i^{-1} e \hbar \left(\frac{2\pi}{a}\right)^{-2}$',
                    r'$e \left(\frac{2\pi}{a}\right)^{-2}$',
                    r'$e \left(\frac{2\pi}{a}\right)^{-2}$',
                    r'$n_i^{-1} e \hbar \left(\frac{2\pi}{a}\right)^{-2}$',
                    r'$e \left(\frac{2\pi}{a}\right)^{-2}$',
                    r'$e \left(\frac{2\pi}{a}\right)^{-2}$']
    xlabel = r"$k_x$ [$2\pi / a$]"
    ylabel = r"$k_y$ [$2\pi / a$]"

    for key, title, title_units in zip(keys, titles, titles_units):
        for mu_index, val_list in enumerate(sorted_data[key]):
            val_arr = list_to_array(kmb, val_list)

            # We want to display (kx, ky) points with the proper metric such
            # that if we integrated (instead of summed), we get the right
            # answer. We need to divide by kmb.k_step(0) * kmb.k_step(1)
            # to remove the area factor which has been included already.
            kxy_area = kmb.k_step(0) * kmb.k_step(1)
            val_band_sum = np.sum(val_arr, axis=2) / kxy_area

            # Shift origin by (-1/2, -1/2) so that we cover area [-1/2, 1/2) x [-1/2, 1/2).
            s0, s1 = -(kmb.Nk[0] // 2), -(kmb.Nk[1] // 2)
            val_band_sum_shifted, all_k0s, all_k1s = shifted_sample(kmb, s0, s1, val_band_sum)

            min_val, max_val = np.amin(val_band_sum_shifted), np.amax(val_band_sum_shifted)
            max_abs = max([abs(min_val), abs(max_val)])
            scale_power = math.floor(math.log10(max_abs))
            scale = 10.0**scale_power
            val_band_sum_shifted /= scale

            if scale_power != 0:
                title_scale_part = r"$\times 10^{" + str(int(scale_power)) + r"}$"
                full_title = "{} [{} {}]".format(title, title_scale_part, title_units)
            else:
                full_title = "{} [{}]".format(title, title_units)

            val_band_sum_list = array_to_list(kmb_oneband, val_band_sum_shifted, band_index=False)

            plot_prefix = "{}_{}_band_sum_mu_{}".format(prefix, key, str(mu_index))

            plot_2d_bz_slice(plot_prefix, full_title, all_k0s, all_k1s, val_band_sum_list,
                    xlabel=xlabel, ylabel=ylabel)

def get_interpolated_mus(mus, interp_per_point):
    mu_extra = 10
    interpolated_mus = []
    for mu_index, mu in enumerate(mus[:-1]):
        if mu_index == len(mus) - 2:
            endpoint = True
        else:
            endpoint = False

        mu_next = mus[mu_index + 1]

        interpolated_mus.extend(np.linspace(mu, mu_next, mu_extra, endpoint=endpoint))

    return interpolated_mus

def plot_series(prefix, fdata):
    Emin, Emax = min(fdata['Ekm']), max(fdata['Ekm'])
    a = fdata['a']
    t = fdata['t']
    tr = fdata['tr']
    U0 = fdata['U0']

    # Skip first and last points, since these are at energy bounds and have FS only
    # due to finite temperature.
    eps = 1e-9
    assert(abs(fdata['mus'][0] - Emin) < eps)
    assert(abs(fdata['mus'][-1] - Emax) < eps)
    mus = fdata['mus'][1:-1]

    # Interpolated mus over small region at low mu.
    # Analytic results are in this regime.
    interp_per_point = 10
    interpolated_mus_full = get_interpolated_mus(fdata['mus'], interp_per_point)
    interpolated_mus_left = interpolated_mus_full[:len(interpolated_mus_full)//10]

    sys = fdata['_series_sy'][1:-1]
    js_sz_vy_ints = fdata['_series_js_sz_vy_intrinsic'][1:-1]
    js_sz_vy_exts = fdata['_series_js_sz_vy_extrinsic'][1:-1]

    # <s_y>
    sy_expected = -a * tr / (math.pi * U0**2)
    sy_expected_line = [sy_expected] * len(interpolated_mus_left)

    sy_title = r'$\langle s_y \rangle / E_x$ [$n_i^{-1} e \hbar$]'
    plt.title(sy_title, fontsize='large')

    sy_xlabel = r'$\mu$ [$t$]'
    plt.xlabel(sy_xlabel, fontsize='large')
    plt.xlim(Emin, Emax)

    plt.axhline(0.0, color='k')

    plt.plot(mus, sys, 'ko')
    plt.plot(interpolated_mus_left, sy_expected_line, 'g-', linewidth=3)

    sy_plot_path = "{}_sy".format(prefix)
    plt.savefig("{}.png".format(sy_plot_path), bbox_inches='tight', dpi=500)
    plt.clf()

    # <j_{s_z, v_y}>
    js_sz_vy_total = [intrinsic + extrinsic for intrinsic, extrinsic in zip(js_sz_vy_ints, js_sz_vy_exts)]
    js_sz_vy_int_expected = 1 / (8 * math.pi)
    js_sz_vy_ext_expected = -1 / (8 * math.pi)
    js_sz_vy_int_expected_line = [js_sz_vy_int_expected] * len(interpolated_mus_left)
    js_sz_vy_ext_expected_line = [js_sz_vy_ext_expected] * len(interpolated_mus_left)

    js_title = r'$\sigma^{s_z}_{xy}$ [$e$]'
    plt.title(js_title, fontsize='large')

    js_xlabel = sy_xlabel
    plt.xlabel(js_xlabel, fontsize='large')
    plt.xlim(Emin, Emax)

    plt.axhline(0.0, color='k')

    plt.plot(mus, js_sz_vy_ints, 'bo', label="Int.")
    plt.plot(mus, js_sz_vy_exts, 'ro', label="Ext.")
    plt.plot(mus, js_sz_vy_total, 'ko', label="Total")
    plt.plot(interpolated_mus_left, js_sz_vy_int_expected_line, 'c-', linewidth=3)
    plt.plot(interpolated_mus_left, js_sz_vy_ext_expected_line, 'm-', linewidth=3)

    plt.legend(loc=(0.75, 0.6), fontsize='large')

    js_plot_path = "{}_js".format(prefix)
    plt.savefig("{}.png".format(js_plot_path), bbox_inches='tight', dpi=500)
    plt.clf()

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
    plot_series(args.prefix, fdata)

if __name__ == '__main__':
    _main()
