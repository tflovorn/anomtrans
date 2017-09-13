#ifndef ANOMTRANS_DRIVING_H
#define ANOMTRANS_DRIVING_H

#include <cstddef>
#include <stdexcept>
#include <petscksp.h>
#include "grid_basis.h"
#include "util.h"
#include "derivative.h"
#include "mat.h"

namespace anomtrans {

/** @brief Applies the electric field driving term:
 *         given <rho>, output: hbar/(e * |E|) * Dbar_E(<rho>).
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param Ehat_dot_grad_k Dot product of the electric field direction (Cartesian unit vector)
 *                         with the (Cartesian) k-gradient matrix.
 *  @param Ehat_dot_R Dot product of the electric field direction (Cartesian unit vector)
 *                    with the (Cartesian) Berry connection.
 *  @param rho Density matrix to apply the electric driving term to.
 *  @todo Implement Berry connection contribution: add argument Mat Ehat_dot_R.
 *  @todo Is it appropriate for E to have the same dimension as k?
 *        Term E dot d<rho>/dk has vanishing contributions from E components
 *        where there are no corresponding k components.
 *        Possibly some interesting situations where E is perpendicular to a
 *        surface, though.
 *  @todo Determine appropriate fill ratio for MatMatMult in apply_deriv.
 *  @todo Determine appropriate fill ratio commutator?
 *  @todo Same nonzero pattern in grad and R part terms?
 */
template <std::size_t k_dim>
Mat apply_driving_electric(const kmBasis<k_dim> &kmb, Mat Ehat_dot_grad_k,
    Mat Ehat_dot_R, Mat rho) {
  Mat result = apply_deriv(kmb, Ehat_dot_grad_k, rho);

  Mat R_part = commutator(Ehat_dot_R, rho);

  PetscErrorCode ierr = MatAXPY(result, std::complex<double>(0.0, -1.0), R_part,
      DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);

  ierr = MatDestroy(&R_part);CHKERRXX(ierr);

  return result;
}

/** @brief Magnetic field driving term hbar^2/e * Dbar_B.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param DH0_cross_Bhat Cross product of the (Cartesian) covariant derivative of the unperturbed
 *                     Hamiltonian with the magnetic field direction (Cartesian unit vector).
 *  @param d_dk_Cart Components of the k-gradient matrix given in Cartesian coordinates.
 *  @param R Berry connection (in Cartesian coordinates).
 *  @param rho Density matrix to apply the magnetic driving term to.
 *  @note `DH0_cross_Bhat` only has finite contribution to dot product with D<rho>/Dk
 *        for components in range 0 <= dc < k_dim, since D<rho>/Dk vanishes if dc >= k_dim.
 *        Only components of `DH0_cross_B` which can contribute to the result are given.
 *  @todo Implement Berry connection contribution:
 *        add argument std::array<Mat, 3> berry_connection.
 *  @todo Determine appropriate fill ratio for MatMatMult in apply_deriv.
 *  @todo Optimize to reduce number of allocations.
 */
template <std::size_t k_dim>
Mat apply_driving_magnetic(const kmBasis<k_dim> &kmb, std::array<Mat, k_dim> DH0_cross_Bhat,
    std::array<Mat, k_dim> d_dk_Cart, std::array<Mat, k_dim> R, Mat rho) {
  static_assert(k_dim > 0, "must have at least 1 spatial dimension");

  std::array<Mat, k_dim> rho_part;
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    // rho_part = grad_k <rho>
    rho_part.at(dc) = apply_deriv(kmb, d_dk_Cart.at(dc), rho);

    // R_part = [R, <rho>]
    Mat R_part = commutator(R.at(dc), rho);

    // rho_part <- grad_k <rho> - i [R, <rho>]
    PetscErrorCode ierr = MatAXPY(rho_part.at(dc), std::complex<double>(0.0, -1.0), R_part,
        DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);

    ierr = MatDestroy(&R_part);CHKERRXX(ierr);
  }

  // Symmetrized product: {a dot b} = a dot b + b dot a.
  // a dot b part:
  Mat result = nullptr;
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    Mat prod;
    PetscErrorCode ierr = MatMatMult(DH0_cross_Bhat.at(dc), rho_part.at(dc), MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, &prod);CHKERRXX(ierr);
    if (dc == 0) {
      result = prod;
    } else {
      // TODO - possible that there is same nonzero pattern for each term?
      PetscErrorCode ierr = MatAXPY(result, 1.0, prod, DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
    }
  }

  // b dot a part:
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    Mat prod;
    PetscErrorCode ierr = MatMatMult(rho_part.at(dc), DH0_cross_Bhat.at(dc), MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, &prod);CHKERRXX(ierr);
    ierr = MatAXPY(result, 1.0, prod, DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
  }

  PetscErrorCode ierr = MatScale(result, 0.5);CHKERRXX(ierr);

  for (std::size_t dc = 0; dc < k_dim; dc++) {
    PetscErrorCode ierr = MatDestroy(&rho_part.at(dc));CHKERRXX(ierr);
  }

  return result;
}

} // namespace anomtrans

#endif // ANOMTRANS_DRIVING_H
