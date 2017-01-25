#include "special_functions.h"

namespace anomtrans {

double fermi_dirac(double E, double beta) {
  double x = -beta*E;

  if (x < LN_DBL_MIN) {
    return 0.0;
  } else if (x < 0) {
    double ex = std::exp(x);
    return ex / (1.0 + ex);
  } else {
    double emx = std::exp(-x);
    return 1.0 / (1.0 + emx);
  }
}

} // namespace anomtrans
