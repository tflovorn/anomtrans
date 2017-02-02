#ifndef ANOMTRANS_ENERGY_H
#define ANOMTRANS_ENERGY_H

#include <cstddef>
#include <vector>
#include <petscksp.h>
#include "grid_basis.h"

namespace anomtrans {

/** @brief Construct a vector of energies using the given k-space discretization
 *         and Hamiltonian.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class giving the Hamiltonian of the system. Should have the method
 *           energy(kmComps<dim>).
 *  @note Vec is a reference type (Petsc implementation: typedef struct _p_Vec* Vec).
 *        We can safely create an object of type Vec on the stack, initialize it with
 *        VecCreate, and return it. The caller will need to call VecDestroy(&v) on
 *        the returned Vec.
 */
template <std::size_t k_dim, typename Hamiltonian>
Vec get_energies(kmBasis<k_dim> kmb, Hamiltonian H) {
  Vec Ekm;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm, &Ekm);CHKERRXX(ierr);

  // This node is assigned elements in the range begin <= i < end.
  PetscInt begin, end;
  ierr = VecGetOwnershipRange(Ekm, &begin, &end);CHKERRXX(ierr);

  std::vector<PetscInt> local_rows;
  local_rows.reserve(end - begin);
  std::vector<PetscScalar> local_values;
  local_values.reserve(end - begin);

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    auto ikm_comps = kmb.decompose(local_row);
    double energy = H.energy(ikm_comps);

    local_rows.push_back(local_row);
    local_values.push_back(energy);
  }

  assert(local_rows.size() == local_values.size());

  // TODO would we be better off adding these elements one at a time (contrary to
  // the PETSc manual's advice), since we don't have them precomputed?
  // Doing it this way uses extra memory inside this scope.
  ierr = VecSetValues(Ekm, local_rows.size(), local_rows.data(), local_values.data(), INSERT_VALUES);CHKERRXX(ierr);

  ierr = VecAssemblyBegin(Ekm);CHKERRXX(ierr);
  ierr = VecAssemblyEnd(Ekm);CHKERRXX(ierr);

  return Ekm;
}

} // namespace anomtrans

#endif // ANOMTRANS_ENERGY_H
