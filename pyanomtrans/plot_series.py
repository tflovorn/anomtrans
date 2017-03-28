import argparse
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_series(plot_path, ys, xs=None, xlabel=None, ylabel=None):
    if xs is None:
        xs = range(len(ys))

    plt.plot(xs, ys, 'ko')

    plt.axhline(0.0, color='k', linestyle='-')

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.savefig("{}.png".format(plot_path), bbox_inches='tight', dpi=500)
    plt.clf()

def _ignore_key(key):
    if key.startswith("_series"):
        return False

    return True

def _main():
    parser = argparse.ArgumentParser("Plot Fermi surface as shown by |df(Emk)/dk|",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for file giving plot data: should be in the form prefix.json")
    parser.add_argument("in_dir", type=str,
            help="Directory containing file giving plot data")
    args = parser.parse_args()

    fpath = os.path.join(args.in_dir, "{}.json".format(args.prefix))
    with open(fpath, 'r') as fp:
        fdata = json.load(fp)

    series_data = {}
    for key, val in fdata.items():
        if _ignore_key(key):
            continue

        series_data[key] = val

    special_keys = {
            '_series_Hall_conductivity': {
                    'xs': 'mus',
                    'xlabel': "$\\mu$",
                    'ylabel': "$\\sigma^{Hall}_{xy}$"
            }
    }

    for key, ys in series_data.items():
        if key in special_keys:
            xs = fdata[special_keys[key]['xs']]
            xlabel = special_keys[key]['xlabel']
            ylabel = special_keys[key]['ylabel']

            plot_series("{}_{}".format(args.prefix, key), ys, xs=xs, xlabel=xlabel, ylabel=ylabel)
        else:
            plot_series("{}_{}".format(args.prefix, key), ys)

if __name__ == '__main__':
    _main()
