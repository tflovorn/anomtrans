from sys import float_info
from math import exp, log

_LN_DBL_MIN = log(float_info.min)

def fermi_dirac(beta, E):
    x = -beta*E

    if x < _LN_DBL_MIN:
        return 0.0
    elif x < 0.0:
        ex = exp(x)
        return ex / (1.0 + ex)
    else:
        emx = exp(-x)
        return 1.0 / (1.0 + emx)
