import argparse
import os.path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pyanomtrans.plot_Rashba import get_interpolated_mus

def plot_series(prefix, fdata):
    mu5s = fdata['mu5s']

    interp_per_point = 10
    interpolated_mu5s = get_interpolated_mus(mu5s, interp_per_point)

    current_Ss = fdata['_series_current_S_B']
    current_xis = fdata['_series_current_xi_B']

    # j_z^S
    current_S_expecteds = [(2/(3 * 2 * math.pi**2)) * mu5 for mu5 in interpolated_mu5s]

    S_title = r'$\langle j^S_z(\mu_5) \rangle / B_z$ [$e^2 \Delta$]'
    plt.title(S_title, fontsize='large')

    S_xlabel = r'$\mu_5$ [$\Delta$]'
    plt.xlabel(S_xlabel, fontsize='large')

    plt.axhline(0.0, color='k')

    plt.plot(mu5s, current_Ss, 'ko')
    plt.plot(interpolated_mu5s, current_S_expecteds, 'g-')

    S_plot_path = "{}_j_S".format(prefix)
    plt.savefig("{}.png".format(S_plot_path), bbox_inches='tight', dpi=500)
    plt.clf()

    # j_z^{\xi}(\mu5) = (1/2) j_z^S(\mu)
    current_xi_expecteds = [j_S / 2.0 for j_S in current_S_expecteds]

    xi_title = r'$\langle j^{\xi}_z(\mu_5) \rangle / B_z$ [$e^2 \Delta$]'
    plt.title(xi_title, fontsize='large')

    xi_xlabel = S_xlabel
    plt.xlabel(xi_xlabel, fontsize='large')

    plt.axhline(0.0, color='k')

    plt.plot(mu5s, current_xis, 'ko')
    plt.plot(interpolated_mu5s, current_xi_expecteds, 'g-')

    xi_plot_path = "{}_j_xi".format(prefix)
    plt.savefig("{}.png".format(xi_plot_path), bbox_inches='tight', dpi=500)
    plt.clf()

def _main():
    parser = argparse.ArgumentParser("Plot WSM finite mu5 data averaged over kz and summed over bands",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for file giving plot data: should be in the form prefix.json")
    parser.add_argument("in_dir", type=str,
            help="Directory containing file giving plot data")
    args = parser.parse_args()

    fpath = os.path.join(args.in_dir, "{}.json".format(args.prefix))
    with open(fpath, 'r') as fp:
        fdata = json.load(fp)

    plot_series(args.prefix, fdata)

if __name__ == '__main__':
    _main()
