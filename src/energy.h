#ifndef ANOMTRANS_ENERGY_H
#define ANOMTRANS_ENERGY_H

#include <cstddef>
#include <vector>
#include <petscksp.h>
#include "vec.h"
#include "grid_basis.h"
#include "derivative.h"

namespace anomtrans {

/** @brief Construct a vector of energies using the given k-space discretization
 *         and Hamiltonian.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have
 *           the method energy(kmComps<dim>).
 *  @note Vec is a reference type (Petsc implementation: typedef struct _p_Vec* Vec).
 *        We can safely create an object of type Vec on the stack, initialize it with
 *        VecCreate, and return it. The caller will need to call VecDestroy(&v) on
 *        the returned Vec.
 */
template <std::size_t k_dim, typename Hamiltonian>
Vec get_energies(kmBasis<k_dim> kmb, Hamiltonian H) {
  // TODO may want to just pass ikm to H: look up precomputed v without
  // conversion to ikm_comps and back.
  auto E_elem = [kmb, H](PetscInt ikm)->PetscScalar {
    auto ikm_comps = kmb.decompose(ikm);
    return H.energy(ikm_comps);
  };

  return vector_index_apply(kmb.end_ikm, E_elem);
}

/** @brief Find the maximum value of |E_{k+dk_i, m} - E_{k, m}| where dk_i is
 *         a step in one direction in reciprocal space, with the step distance
 *         given by the minimum step in that direction (1/kmb.Nk.at(i)).
 */
template <std::size_t k_dim, typename Hamiltonian>
PetscReal find_max_energy_difference(const kmBasis<k_dim> &kmb, const Hamiltonian &H) {
  Vec Ekm = get_energies(kmb, H);

  // First-order approximant for foward difference first derivative
  // = (E_{k+dk_i, m} - E_{k,m})/kmb.Nk.at(i)
  DerivStencil<1> stencil(DerivApproxType::forward, 1);

  std::array<Mat, k_dim> d_dk = make_d_dk_recip(kmb, stencil);

  Vec dE_dk;
  PetscErrorCode ierr = VecDuplicate(Ekm, &dE_dk);CHKERRXX(ierr);

  PetscReal dE_dk_max = 0.0;
  for (std::size_t d = 0; d < k_dim; d++) {
    ierr = MatMult(d_dk.at(d), Ekm, dE_dk);CHKERRXX(ierr);
    PetscReal dE_dk_d_max = get_Vec_MaxAbs(dE_dk) * kmb.Nk.at(d);

    if (dE_dk_d_max > dE_dk_max) {
      dE_dk_max = dE_dk_d_max;
    }
  }

  return dE_dk_max;
}

} // namespace anomtrans

#endif // ANOMTRANS_ENERGY_H
