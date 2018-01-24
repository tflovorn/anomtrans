import argparse
import os.path
import json
import matplotlib.pyplot as plt
from plot_2d_bz import extract_sorted_data, split_bands, _one_band_list, _is_list

def plot_slice_total(prefix, kmb, all_values):
    dim = len(kmb.Nk)
    k_totals = [[0.0] * kmb.Nk[d] for d in range(dim)]

    for i, contrib in enumerate(all_values):
        # Assumes that each band is grouped together in the basis:
        # [(k0, m0), (k1, m0), ..., (k0, m1), (k1, m1), ...
        # Allows for `all_values` to be a single band's worth of contribution
        # (or already summed over bands).
        k, _ = kmb.decompose(i)
        for d in range(dim):
            k_totals[d][k[d]] += contrib

    for d in range(dim):
        xs = list(range(kmb.Nk[d]))
        plt.plot(xs, k_totals[d], 'k-')
        plt.savefig("{}_k{}.png".format(prefix, d), bbox_inches='tight', dpi=500)
        plt.clf()

def process_data(prefix, only, kmb, sorted_data, split_bands=False):
    for key, val in sorted_data.items():
        # Don't plot the keys containing indices (these aren't values to plot)
        if key in ('k_comps', 'ms'):
            continue

        if only is not None and key != only:
            continue

        if _is_list(val[0]):
            # val is a list of lists (e.g. one result for each chemical potential)
            for subval_index, subval_all_bands in enumerate(val):
                if split_bands:
                    subval_split = split_bands(subval_all_bands, kmb.Nbands)
                    for band_index, subval_band in enumerate(subval_split):
                        band_prefix = "{}_{}_m{}_{}".format(prefix, key, band_index, subval_index)
                        plot_slice_total(band_prefix, kmb, subval_band)
                else:
                    subval_prefix = "{}_{}_{}".format(prefix, key, subval_index)
                    plot_slice_total(subval_prefix, kmb, subval_all_bands)
        else:
            # val is a single list (e.g. result same for all chemical potentials)
            if split_bands:
                val_split = split_bands(val, kmb.Nbands)
                for band_index, val_band in enumerate(val_split):
                    band_prefix = "{}_{}_m{}".format(prefix, key, band_index)
                    plot_slice_total(band_prefix, kmb, val_band)
            else:
                val_prefix = "{}_{}".format(prefix, key)
                plot_slice_total(val_prefix, kmb, val)

def _main():
    parser = argparse.ArgumentParser("Plot data summed over bands and some directions in the Brillouin zone (e.g. total contribution from all bands and all (k0, k1) at fixed k2)",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for file giving plot data: should be in the form prefix.json")
    parser.add_argument("in_dir", type=str,
            help="Directory containing file giving plot data")
    parser.add_argument("--only", type=str, default=None,
            help="If specified, plot only the series with the specified name")
    parser.add_argument("--split_bands", action='store_true',
            help="Split contributions from different bands instead of summing")
    args = parser.parse_args()

    fpath = os.path.join(args.in_dir, "{}.json".format(args.prefix))
    with open(fpath, 'r') as fp:
        fdata = json.load(fp)

    kmb, sorted_data = extract_sorted_data(fdata)

    process_data(args.prefix, args.only, kmb, sorted_data, args.split_bands)

if __name__ == '__main__':
    _main()
