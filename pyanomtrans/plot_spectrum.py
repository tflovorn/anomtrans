import argparse
import os.path
import json
import matplotlib.pyplot as plt
from plot_2d_bz import extract_sorted_data, _one_band_list, _is_list

def process_data(prefix, kmb, sorted_data):
    Ekm_result = sorted_data['Ekm']
    dim = len(kmb.Nk)

    if (dim != 3):
        print("dim != 3 unimplemented")

    # For now, to avoid generating a ton of plots, only plot
    # spectrum as a function of kz at each kx for fixed ky.
    # Useful for plotting WSM spectrum.
    #
    # More general implementation: choose d0 != d1 != d2.
    # For each pair of values on (d0, d1), plot spectrum as a function of d0 value.
    # Repeat for each combination of d's.
    # i.e. for each (kx, ky), plot spectrum as a function of kz;
    # for each (kx, kz), plot spectrum as a function of ky; ...

    k1 = kmb.Nk[1] // 2
    for k0 in range(kmb.Nk[0]):
        Emks = []
        for m in range(kmb.Nbands):
            Emks.append([])

        for m in range(kmb.Nbands):
            for k2 in range(kmb.Nk[2]):
                ikm = kmb.compose([[k0, k1, k2], m])
                Emks[m].append(Ekm_result[ikm])

        xs = list(range(kmb.Nk[2]))
        for m in range(kmb.Nbands):
            plt.plot(xs, Emks[m], 'k-')

        out_path = "{}_k0_{}_k1_{}.png".format(prefix, k0, k1)
        plt.savefig(out_path, bbox_inches='tight', dpi=500)
        plt.clf()

def _main():
    parser = argparse.ArgumentParser("Plot spectrum along constant k0, k1, or k2",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for file giving plot data: should be in the form prefix.json")
    parser.add_argument("in_dir", type=str,
            help="Directory containing file giving plot data")
    args = parser.parse_args()

    fpath = os.path.join(args.in_dir, "{}.json".format(args.prefix))
    with open(fpath, 'r') as fp:
        fdata = json.load(fp)

    kmb, sorted_data = extract_sorted_data(fdata)

    process_data(args.prefix, kmb, sorted_data)

if __name__ == '__main__':
    _main()
