#ifndef ANOMTRANS_OBSERVABLES_SPIN_H
#define ANOMTRANS_OBSERVABLES_SPIN_H

#include <cstddef>
#include <array>
#include <boost/optional.hpp>
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
 *  @param ret_Mat Iff true, the matrix product is returned, normalized according to the
 *                 k-space metric.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
template <std::size_t k_dim>
ArrayResult<3> calculate_spin_ev(const kmBasis<k_dim> &kmb, std::array<Mat, 3> spin, Mat rho,
    bool ret_Mat) {
  ArrayResult<3> result;
  for (std::size_t dc = 0; dc < 3; dc++) {
    std::array<Mat, 2> prod_Mats = {spin.at(dc), rho};
    result.at(dc) = Mat_product_trace_normalized(kmb, prod_Mats, ret_Mat);
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
 *  @param ret_Mat Iff true, the matrix product is returned, normalized according to the
 *                 k-space metric.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
template <std::size_t k_dim>
NestedArrayResult<3, k_dim> calculate_spin_current_ev(const kmBasis<k_dim> &kmb,
    std::array<Mat, 3> spin, std::array<Mat, k_dim> v, Mat rho, bool ret_Mat) {
  NestedArrayResult<3, k_dim> result;

  for (std::size_t dc_s = 0; dc_s < 3; dc_s++) {
    for (std::size_t dc_v = 0; dc_v < k_dim; dc_v++) {
      std::array<Mat, 3> prod_Mats_sv = {spin.at(dc_s), v.at(dc_v), rho};
      std::array<Mat, 3> prod_Mats_vs = {v.at(dc_v), spin.at(dc_s), rho};

      auto js_sv = Mat_product_trace_normalized(kmb, prod_Mats_sv, ret_Mat);
      auto js_vs = Mat_product_trace_normalized(kmb, prod_Mats_vs, ret_Mat);

      result.at(dc_s).at(dc_v).first = 0.5 * (js_sv.first + js_vs.first);

      if (ret_Mat) {
        // js_sv.second <- 0.5 * (js_sv.second + js_vs.second)
        PetscErrorCode ierr = MatAXPY(*js_sv.second, 1.0, *js_vs.second,
            DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
        ierr = MatScale(*js_sv.second, 0.5);CHKERRXX(ierr);

        // No longer need js_vs.
        ierr = MatDestroy(&(*js_vs.second));CHKERRXX(ierr);

        result.at(dc_s).at(dc_v).second = js_sv.second;
      }
    }
  }

  return result;
}

} // namespace anomtrans

#endif // ANOMTRANS_OBSERVABLES_SPIN_H
