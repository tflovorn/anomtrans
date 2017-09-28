import argparse
import os.path
import json
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pyanomtrans.grid_basis import kmBasis, km_at

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

    if title is not None:
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

def get_Nk(k_comps):
    max_k = None
    for k in k_comps:
        if max_k is None:
            max_k = [-1]*len(k)

        for d, k_d in enumerate(k):
            if k_d > max_k[d]:
                max_k[d] = k_d

    Nk = [max_k_d + 1 for max_k_d in max_k]
    return Nk

def get_Nbands(ms):
    return max(ms) + 1

def is_list(x):
    return hasattr(x, '__iter__')

def sorted_by_km(kmb, k_comps, ms, vals):
    km_val_tuple = zip(k_comps, ms, vals)
    def sort_fn(kmval):
        ikm_comps = [kmval[0], kmval[1]]
        return kmb.compose(ikm_comps)

    km_val_sorted = sorted(km_val_tuple, key=sort_fn)
    val_sorted = [kmval[2] for kmval in km_val_sorted]
    return val_sorted

def _ignore_key(key):
    ignore_key_prefix = ["_series", "mus"]
    for key_prefix in ignore_key_prefix:
        if key.startswith(key_prefix):
            return True

    return False

def split_bands(val, Nbands):
    # Naive implementation: assume values are ordered s.t. values for each band are grouped.
    # Entries [0:prod(Nk)] are band 0, [prod(Nk):2*prod(Nk)] are band 1, ...
    assert(len(val) % Nbands == 0)
    Nk_tot = int(len(val) / Nbands)

    split_val = []
    for m in range(Nbands):
        split_val.append(val[m*Nk_tot:(m+1)*Nk_tot])

    return split_val

def _main():
    parser = argparse.ArgumentParser("Plot Fermi surface as shown by |df(Emk)/dk|",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for file giving plot data: should be in the form prefix.json")
    parser.add_argument("in_dir", type=str,
            help="Directory containing file giving plot data")
    parser.add_argument("--only", type=str, default=None,
            help="If specified, plot only the series with the specified name")
    args = parser.parse_args()

    fpath = os.path.join(args.in_dir, "{}.json".format(args.prefix))
    with open(fpath, 'r') as fp:
        fdata = json.load(fp)

    all_data = {}
    for key, val in fdata.items():
        if _ignore_key(key):
            continue

        # Assume all values are lists or lists of lists
        if key not in all_data:
            all_data[key] = val
        else:
            if is_list(val[0]) and key not in ['k_comps']:
                # val is a list of lists
                # Treat k_comps separately (the lists represent values we want to keep together)
                for subval_index, subval in enumerate(val):
                    all_data[key][subval_index].extend(subval)
            else:
                # val is a list or k_comps
                all_data[key].extend(val)


    # Sort data by k's.
    # Assume that a key 'k_comps' is present.
    # Assume that a key 'ms' is present.
    Nk = get_Nk(all_data['k_comps'])
    Nbands = get_Nbands(all_data['ms'])

    kmb = kmBasis(Nk, Nbands)

    sorted_data = {}
    for key, val in all_data.items():
        # TODO could just sort once, put all vals in one tuple
        if is_list(val[0]) and key not in ['k_comps']:
            if key not in sorted_data:
                sorted_data[key] = []

            for subval_index, subval in enumerate(val):
                this_sorted = sorted_by_km(kmb, all_data['k_comps'], all_data['ms'], subval)
                sorted_data[key].append(this_sorted)

        else:
            sorted_data[key] = sorted_by_km(kmb, all_data['k_comps'], all_data['ms'], val)

    # TODO handle d != 2
    if len(Nk) != 2:
        raise ValueError("d != 2 unsupported")

    all_k0s, all_k1s = [], []
    for iks in split_bands(sorted_data['k_comps'], Nbands)[0]:
        k, _ = km_at(kmb.Nk, (iks, 0))
        all_k0s.append(k[0])
        all_k1s.append(k[1])

    for key, val in sorted_data.items():
        # Don't plot the keys containing indices (these aren't values to plot)
        if key in ('k_comps', 'ms'):
            continue

        if args.only is not None and key != args.only:
            continue

        if is_list(val[0]):
            # val is a list of lists
            for subval_index, subval_all_bands in enumerate(val):
                subval_split = split_bands(subval_all_bands, Nbands)
                for band_index, subval_band in enumerate(subval_split):
                    # TODO incorporate plot titles
                    title = None
                    plot_2d_bz_slice("{}_{}_m{}_{}".format(args.prefix, key, band_index, subval_index), title, all_k0s, all_k1s, subval_band)
        else:
            # val is a single list
            val_split = split_bands(val, Nbands)
            for band_index, val_band in enumerate(val_split):
                # TODO incorporate plot titles
                title = None
                plot_2d_bz_slice("{}_{}_m{}".format(args.prefix, key, band_index), title, all_k0s, all_k1s, val_band)

if __name__ == '__main__':
    _main()
