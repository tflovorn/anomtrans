#ifndef ANOMTRANS_OBSERVABLES_CURRENT_H
#define ANOMTRANS_OBSERVABLES_CURRENT_H

#include <stdexcept>
#include <array>
#include <utility>
#include <algorithm>
#include <petscksp.h>
#include "util/vec.h"
#include "util/mat.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief Compute \hbar times <v>, the expectation value of the velocity operator:
 *         \hbar <v> = \hbar Tr[v <rho>] = \sum_{kmm'} <km|\grad_k H_k|km'> <rho>_k^{mm'}.
 *  @returns The Cartesian components of \hbar <v>.
 *  @param ret_Mat Iff true, the matrix product is returned, normalized according to the
 *                 k-space metric.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
template <std::size_t k_dim>
ArrayResult<k_dim> calculate_velocity_ev(const kmBasis<k_dim> &kmb,
    std::array<OwnedMat, k_dim>& v, Mat rho, bool ret_Mat) {
  ArrayResult<k_dim> result;
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    std::array<Mat, 2> prod_Mats = {v.at(dc).M, rho};
    result.at(dc) = Mat_product_trace_normalized(kmb, prod_Mats, ret_Mat);
  }

  return result;
}

/** @brief Compute the expectation value of the current operator:
 *         \sigma_a = Tr[j_a <rho>] = -e/hbar Tr[v_a <rho>].
 *  @param ret_Mat Iff true, the matrix product is returned, normalized according to the
 *                 k-space metric.
 *  @returns The Cartesian components of \sigma_a, in units of hbar/e.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
template <std::size_t k_dim>
ArrayResult<k_dim> calculate_current_ev(const kmBasis<k_dim> &kmb,
    std::array<OwnedMat, k_dim>& v, Mat rho, bool ret_Mat) {
  auto result = calculate_velocity_ev(kmb, v, rho, ret_Mat);

  for (std::size_t d = 0; d < k_dim; d++) {
    result.at(d).first = -result.at(d).first;
    if (ret_Mat) {
      PetscErrorCode ierr = MatScale((*(result.at(d).second)).M, -1.0);CHKERRXX(ierr);
    }
  }

  return result;
}

/** @brief Compute the velocity operator in units of \hbar:
 *         v_{km, km'} = (1/\hbar) <km|\grad_k H_k|km'>
 *  @returns The Cartesian components of v.
 */
template <std::size_t k_dim, typename Hamiltonian>
std::array<OwnedMat, k_dim> calculate_velocity(const kmBasis<k_dim> &kmb,
    const Hamiltonian &H) {
  auto v_elem = [&kmb, &H](PetscInt ikm, unsigned int mp)->std::array<PetscScalar, k_dim> {
    auto ikm_comps = kmb.decompose(ikm);
    return H.gradient(ikm_comps, mp);
  };

  auto v = construct_k_diagonal_Mat_array<k_dim>(kmb, v_elem);

  return v;
}

} // namespace anomtrans

#endif // ANOMTRANS_OBSERVABLES_CURRENT_H
