import argparse
import os.path
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_series(plot_path, val):
    xs = range(len(val))
    plt.plot(xs, val, 'ko')

    plt.axhline(0.0, color='k', linestyle='-')

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

    all_data = {}
    for key, val in fdata.items():
        if _ignore_key(key):
            continue

        all_data[key] = val

    for key, val in all_data.items():
        plot_series("{}_{}".format(args.prefix, key), val)

if __name__ == '__main__':
    _main()
