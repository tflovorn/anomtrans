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
    auto spin_elem = [&H, dc](PetscInt ikm, unsigned int mp)->PetscScalar {
      return H.spin(ikm, mp).at(dc);
    };
    spin.at(dc) = construct_k_diagonal_Mat(kmb, spin_elem);
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
