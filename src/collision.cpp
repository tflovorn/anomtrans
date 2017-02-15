#include "collision.h"

namespace anomtrans {

double delta_Gaussian(double sigma, double x) {
  double exp_arg = -x*x/(2*sigma*sigma);
  if (exp_arg < LN_DBL_MIN) {
    return 0.0;
  }
  double coeff = 1/(std::sqrt(2*pi)*sigma);
  return coeff * std::exp(exp_arg);
}

} // namespace anomtrans
