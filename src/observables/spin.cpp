#include "observables/spin.h"

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

std::array<PetscScalar, 3> calculate_spin_ev(std::array<Mat, 3> spin, Mat rho) {
  std::array<PetscScalar, 3> result;
  for (std::size_t dc = 0; dc < 3; dc++) {
    std::array<Mat, 2> prod_Mats = {spin.at(dc), rho};
    result.at(dc) = Mat_product_trace(prod_Mats);
  }

  return result;
}

} // namespace anomtrans
