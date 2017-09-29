#include "observables/pauli.h"

namespace anomtrans {

std::array<Eigen::Matrix2cd, 3> pauli_matrices() {
  std::complex<double> i1(0.0, 1.0);

  Eigen::Matrix2cd sigma_x, sigma_y, sigma_z;
  sigma_x << 0.0, 1.0,
             1.0, 0.0;
  sigma_y << 0.0, -i1,
             i1, 0.0;
  sigma_z << 1.0, 0.0,
             0.0, -1.0;

  return {sigma_x, sigma_y, sigma_z};
}

} // namespace anomtrans
