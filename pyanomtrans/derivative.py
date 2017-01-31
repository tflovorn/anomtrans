from math import pi
import numpy as np
from scipy.sparse import dok_matrix

_all_Deltas_1d = {2: (1, -1)}
_all_vals_1d = {2: (0.5, -0.5)}

def finite_difference(kmb, order, row_ikm, deriv_dir):
    if order % 2 == 1:
        raise ValueError("Only even-order central finite differences are defined")

    if order not in _all_Deltas_1d or order not in _all_vals_1d:
        raise ValueError("The given finite-difference order is not implemented")

    Deltas_1d = _all_Deltas_1d[order]
    vals_1d = _all_vals_1d[order]

    h = 1.0/kmb.Nk[deriv_dir]

    column_ikms, column_vals = [], []

    for Delta_index in range(len(Deltas_1d)):
        Delta = []
        for d_Delta in range(kmb.k_dim()):
            if d_Delta == deriv_dir:
                Delta.append(Deltas_1d[Delta_index])
            else:
                Delta.append(0)

        column_ikms.append(kmb.add(row_ikm, Delta))
        column_vals.append(vals_1d[Delta_index] / h)

    return column_ikms, column_vals

def make_d_dk_recip_dok(kmb, order):
    d_dk_recip = []
    for d in range(kmb.k_dim()):
        d_dk_i = dok_matrix((kmb.end_ikm, kmb.end_ikm), dtype=np.float64)
        d_dk_recip.append(d_dk_i)

    for row_ikm in range(kmb.end_ikm):
        for d in range(kmb.k_dim()):
            column_ikms, column_vals = finite_difference(kmb, order, row_ikm, d)
            for col_ikm, val in zip(column_ikms, column_vals):
                d_dk_recip[d][row_ikm, col_ikm] = val

    return d_dk_recip

def make_d_dk_recip_csr(kmb, order):
    d_dk_recip_dok = make_d_dk_recip_dok(kmb, order)

    d_dk_recip_csr = []
    for d_dk_i_dok in d_dk_recip_dok:
        d_dk_recip_csr.append(d_dk_i_dok.tocsr())

    return d_dk_recip_csr

def make_d_dk_Cartesian_dok(D, kmb, order):
    d_dk_recip = make_d_dk_recip_dok(kmb, order)

    d_dk_Cart = []
    for d in range(kmb.k_dim()):
        d_dk_c = dok_matrix((kmb.end_ikm, kmb.end_ikm), dtype=np.float64)
        d_dk_Cart.append(d_dk_c)

    for dc in range(kmb.k_dim()):
        for di in range(kmb.k_dim()):
            coeff = D[c, i] / (2*pi)
            d_dk_Cart += coeff * d_dk_recip[di]

    return d_dk_Cart

def make_d_dk_Cartesian_csr(D, kmb, order):
    d_dk_Cart_dok = make_d_dk_Cartesian_dok(D, kmb, order)

    d_dk_Cart_csr = []
    for d_dk_c_dok in d_dk_Cart_dok:
        d_dk_Cart_csr.append(d_dk_c_dok.tocsr())

    return d_dk_Cart_csr
