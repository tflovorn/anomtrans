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
  // TODO can optimize this: only diagonal elements of AB needed.
  // Don't need to do full product.
  for (std::size_t dc = 0; dc < 3; dc++) {
    Mat prod;
    // TODO good estimate for fill?
    // If rho is off-diagonal part, nnz(A) = nnz(B) = nnz(C).
    PetscErrorCode ierr = MatMatMult(spin.at(dc), rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &prod);CHKERRXX(ierr);

    PetscScalar ev;
    ierr = MatGetTrace(prod, &ev);CHKERRXX(ierr);
    result.at(dc) = ev;
  }

  return result;
}

} // namespace anomtrans
