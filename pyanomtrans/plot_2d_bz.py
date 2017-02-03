import argparse
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

def _main():
    parser = argparse.ArgumentParser("Plot Fermi surface as shown by |df(Emk)/dk|",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for files giving plot data: should be in the form prefix_#.json where # is an integer")
    parser.add_argument("in_dir", type=str,
            help="Directory containing files giving plot data")
    args = parser.parse_args()

    from pathlib import Path

    all_data = {}
    for fpath in Path(args.in_dir).glob("{}_*.json".format(args.prefix)):
        if fpath.is_dir():
            continue

        with fpath.open('r') as fp:
            fdata = json.load(fp)

        for key, val in fdata.items():
            # Assume all values are lists
            if key not in all_data:
                all_data[key] = val
            else:
                all_data[key].extend(val)


    # Sort data by k's.
    # Assume that a key 'k_comps' is present.
    # TODO: handle multiple m values.
    # Assume that a key 'ms' is present.
    Nk = get_Nk(all_data['k_comps'])
    Nbands = get_Nbands(all_data['ms'])
    if Nbands > 1:
        raise ValueError("nbands > 1 unsupported")

    kmb = kmBasis(Nk, Nbands)

    sorted_data = {}
    for key, val in all_data.items():
        # TODO could just sort once, put all vals in one tuple
        km_val_tuple = zip(all_data['k_comps'], all_data['ms'], val)
        def sort_fn(kmval):
            ikm_comps = [kmval[0], kmval[1]]
            return kmb.compose(ikm_comps)

        km_val_sorted = sorted(km_val_tuple, key=sort_fn)
        val_sorted = [kmval[2] for kmval in km_val_sorted]
        sorted_data[key] = val_sorted

    # TODO handle d != 2
    if len(Nk) != 2:
        raise ValueError("d != 2 unsupported")

    all_k0s, all_k1s = [], []
    for ikm, iks in enumerate(sorted_data['k_comps']):
        m = sorted_data['ms'][ikm]
        assert((iks, m) == kmb.decompose(ikm))
        k, m = km_at(kmb.Nk, (iks, m))
        all_k0s.append(k[0])
        all_k1s.append(k[1])

    for key, val in sorted_data.items():
        # Don't plot the keys containing indices (these aren't values to plot)
        if key in ('k_comps', 'ms'):
            continue

        if hasattr(val[0], '__getitem__'):
            # val is a list of lists
            for subval_index, subval in enumerate(val):
                # TODO incorporate plot titles
                title = None
                plot_2d_bz_slice("{}_{}_{}".format(args.prefix, key, subval_index), title, all_k0s, all_k1s, val)
        else:
            # val is a single list
            # TODO incorporate plot titles
            title = None
            plot_2d_bz_slice("{}_{}".format(args.prefix, key), title, all_k0s, all_k1s, val)

if __name__ == '__main__':
    _main()
