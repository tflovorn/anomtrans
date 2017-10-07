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
  auto spin_elem = [&H](PetscInt ikm, unsigned int mp)->std::array<PetscScalar, 3> {
    return H.spin(ikm, mp);
  };

  std::array<Mat, 3> spin = construct_k_diagonal_Mat_array<3>(kmb, spin_elem);

  return spin;
}

/** @brief Compute the spin expectation value Tr[S_a <rho>].
 *  @returns The Cartesian components of <S> (units of hbar).
 *  @param spin The spin operator components, as returned by calculate_spin_operator().
 *  @param rho The density matrix <rho>.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
template <std::size_t k_dim>
std::array<PetscScalar, 3> calculate_spin_ev(const kmBasis<k_dim> &kmb, std::array<Mat, 3> spin, Mat rho) {
  std::array<PetscScalar, 3> result;
  for (std::size_t dc = 0; dc < 3; dc++) {
    std::array<Mat, 2> prod_Mats = {spin.at(dc), rho};
    result.at(dc) = Mat_product_trace_normalized(kmb, prod_Mats);
  }

  return result;
}

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
template <std::size_t k_dim>
std::array<std::array<PetscScalar, k_dim>, 3> calculate_spin_current_ev(const kmBasis<k_dim> &kmb,
    std::array<Mat, 3> spin, std::array<Mat, k_dim> v, Mat rho) {
  std::array<std::array<PetscScalar, k_dim>, 3> js_ev;

  for (std::size_t dc_s = 0; dc_s < 3; dc_s++) {
    for (std::size_t dc_v = 0; dc_v < k_dim; dc_v++) {
      std::array<Mat, 3> prod_Mats_sv = {spin.at(dc_s), v.at(dc_v), rho};
      std::array<Mat, 3> prod_Mats_vs = {v.at(dc_v), spin.at(dc_s), rho};

      js_ev.at(dc_s).at(dc_v) = 0.5 * (Mat_product_trace_normalized(kmb, prod_Mats_sv)
          + Mat_product_trace_normalized(kmb, prod_Mats_vs));
    }
  }

  return js_ev;
}

} // namespace anomtrans

#endif // ANOMTRANS_SPIN_H
