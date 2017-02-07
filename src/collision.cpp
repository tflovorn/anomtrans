#include "collision.h"

namespace anomtrans {

double delta_Gaussian(double sigma, double x) {
  double coeff = 1/(std::sqrt(2*pi)*sigma);
  return coeff * std::exp(-x*x/(2*sigma*sigma));
}

} // namespace anomtrans
