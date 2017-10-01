#ifndef ANOMTRANS_SPIN_H
#define ANOMTRANS_SPIN_H

#include <cstddef>
#include <array>
#include <Eigen/Core>
#include <petscksp.h>
#include "grid_basis.h"
#include "util/mat.h"

namespace anomtrans {

/** @brief Returns the 2x2 Pauli matrices (sigma_x, sigma_y, sigma_z).
 *  @note For convenient access of all components, these are Eigen matrices which
 *        are local on each rank.
 */
std::array<Eigen::Matrix2cd, 3> pauli_matrices();

/** @brief Compute the spin operator [S_a]_{km, km'}.
 *  @returns The Cartesian components of the spin operator (units of hbar).
 */
template <std::size_t k_dim, typename Hamiltonian>
std::array<Mat, 3> calculate_spin_operator(const kmBasis<k_dim> &kmb,
    const Hamiltonian &H) {
  std::array<Mat, 3> spin;

  for (std::size_t dc = 0; dc < 3; dc++) {
    spin.at(dc) = make_Mat(kmb.end_ikm, kmb.end_ikm, kmb.Nbands);
    PetscInt begin, end;
    PetscErrorCode ierr = MatGetOwnershipRange(spin.at(dc), &begin, &end);CHKERRXX(ierr);

    for (PetscInt ikm = begin; ikm < end; ikm++) {
      std::vector<PetscInt> row_cols;
      std::vector<PetscScalar> row_vals;
      row_cols.reserve(kmb.Nbands);
      row_vals.reserve(kmb.Nbands);

      auto k = std::get<0>(kmb.decompose(ikm));

      for (unsigned int mp = 0; mp < kmb.Nbands; mp++) {
        PetscInt ikmp = kmb.compose(std::make_tuple(k, mp));
        row_cols.push_back(ikmp);

        row_vals.push_back(H.spin(ikm, mp));
      }

      assert(row_cols.size() == row_vals.size());
      ierr = MatSetValues(spin.at(dc), 1, &ikm, row_cols.size(), row_cols.data(), row_vals.data(),
          INSERT_VALUES);CHKERRXX(ierr);
    }

    ierr = MatAssemblyBegin(spin.at(dc), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    ierr = MatAssemblyEnd(spin.at(dc), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  }

  return spin;
}

/** @brief Compute the spin expectation value Tr[S_a <rho>].
 *  @returns The Cartesian components of <S> (units of hbar).
 *  @param spin The spin operator components, as returned by calculate_spin_operator().
 *  @param rho The density matrix <rho>.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
std::array<PetscScalar, 3> calculate_spin_ev(std::array<Mat, 3> spin, Mat rho);

/** @brief Compute the spin current expectation value
 *         (1/2) Tr[(S_a v_b + v_b s_a) <rho>].
 *  @returns A nested array giving the components of the spin current tensor;
 *           the first index specifies the spin direction, and the second specifies
 *           the velocity direction.
 *  @param spin The spin operator components, as returned by calculate_spin_operator().
 *  @param v The velocity operator components, as returned by calculate_velocity_operator().
 *  @param rho The density matrix <rho>.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
/*
template <std::size_t k_dim>
std::array<std::array<PetscScalar, k_dim>, 3> calculate_spin_current_ev(std::array<Mat, 3> spin,
    std::array<Mat, k_dim> v, Mat rho) {

}
*/

} // namespace anomtrans

#endif // ANOMTRANS_SPIN_H
