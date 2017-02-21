#ifndef ANOMTRANS_ENERGY_H
#define ANOMTRANS_ENERGY_H

#include <cstddef>
#include <vector>
#include <petscksp.h>
#include "vec.h"
#include "grid_basis.h"

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

} // namespace anomtrans

#endif // ANOMTRANS_ENERGY_H
