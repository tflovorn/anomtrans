#ifndef ANOMTRANS_CURRENT_H
#define ANOMTRANS_CURRENT_H

#include <stdexcept>
#include <tuple>
#include <algorithm>
#include <petscksp.h>
#include "util/vec.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief Compute \hbar times <v>, the expectation value of the velocity operator:
 *         \hbar <v> = \hbar Tr[v <rho>] = \sum_{kmm'} <km|\grad_k H_k|km'> <rho>_k^{mm'}.
 *  @returns The Cartesian components of \hbar <v>.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
template <std::size_t k_dim>
std::array<PetscScalar, k_dim> calculate_velocity_ev(std::array<Mat, k_dim> v, Mat rho) {
  std::array<PetscScalar, k_dim> result;
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    std::array<Mat, 2> prod_Mats = {v.at(dc), rho};
    result.at(dc) = Mat_product_trace(prod_Mats);
  }

  return result;
}

/** @brief Compute the expectation value of the current operator:
 *         \sigma_a = Tr[j_a <rho>] = -e/hbar Tr[v_a <rho>].
 *  @returns The Cartesian components of \sigma_a, in units of hbar/e.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 */
template <std::size_t k_dim>
std::array<PetscScalar, k_dim> calculate_current_ev(std::array<Mat, k_dim> v, Mat rho) {
  auto v_ev = calculate_velocity_ev(v, rho);

  std::array<PetscScalar, k_dim> conductivity;
  std::transform(v_ev.begin(), v_ev.end(), conductivity.begin(),
      [](PetscScalar v_ev)->PetscScalar { return -v_ev; });

  return conductivity;
}

/** @brief Compute the velocity operator in units of \hbar:
 *         v_{km, km'} = <km|\grad_k H_k|km'>
 *  @returns The Cartesian components of v.
 */
template <std::size_t k_dim, typename Hamiltonian>
std::array<Mat, k_dim> calculate_velocity(const kmBasis<k_dim> &kmb,
    const Hamiltonian &H) {
  auto v_elem = [&kmb, &H](PetscInt ikm, unsigned int mp)->std::array<PetscScalar, k_dim> {
    auto ikm_comps = kmb.decompose(ikm);
    return H.gradient(ikm_comps, mp);
  };

  std::array<Mat, k_dim> v = construct_k_diagonal_Mat_array<k_dim>(kmb, v_elem);

  return v;
}

} // namespace anomtrans

#endif // ANOMTRANS_CURRENT_H
