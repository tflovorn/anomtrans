#ifndef ANOMTRANS_CONDUCTIVITY_H
#define ANOMTRANS_CONDUCTIVITY_H

#include <stdexcept>
#include <tuple>
#include <petscksp.h>
#include "vec.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief Calculate the Hall conductivity
 *         sigma_{xy} = (-e/E_y B_z) Tr[j_x rho_B^{(1)}].
 *  @todo Sure this calculation is correct? Expect an extra factor of e...
 *  @todo Could factor out velocity calculation - general function for
 *        constructing PETSc vector from kmb and function mapping row index
 *        to PETSc scalar would be useful. Could also replace get_energies
 *        with such a function.
 *  @todo This is for single-band case only. In multi-band case, need to
 *        consider difference between dH/dk and dE/dk?
 */
template <std::size_t k_dim, typename Hamiltonian>
PetscScalar calculate_Hall_conductivity(const kmBasis<k_dim> &kmb,
    const Hamiltonian &H, Vec rho1_B) {
  Vec vx;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm, &vx);CHKERRXX(ierr);

  PetscInt begin, end;
  ierr = VecGetOwnershipRange(vx, &begin, &end);CHKERRXX(ierr);

  PetscInt begin_rho, end_rho;
  ierr = VecGetOwnershipRange(rho1_B, &begin_rho, &end_rho);CHKERRXX(ierr);
  if (begin != begin_rho or end != end_rho) {
    ierr = VecDestroy(&vx);CHKERRXX(ierr);
    throw std::runtime_error("got different row distribution for vx and rho");
  }

  std::vector<PetscInt> local_rows;
  local_rows.reserve(end - begin);
  std::vector<PetscScalar> local_vals;
  local_vals.reserve(end - begin);

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    auto ikm_comps = kmb.decompose(local_row);
    auto velocity = H.velocity(ikm_comps);

    local_rows.push_back(local_row);
    local_vals.push_back(std::get<0>(velocity));
  }

  assert(local_rows.size() == local_vals.size());

  ierr = VecSetValues(vx, local_rows.size(), local_rows.data(), local_vals.data(), INSERT_VALUES);CHKERRXX(ierr);

  ierr = VecAssemblyBegin(vx);CHKERRXX(ierr);
  ierr = VecAssemblyEnd(vx);CHKERRXX(ierr);

  PetscScalar sigma_Hall;
  // VecDot(u, v) = v^{\dagger} u
  ierr = VecDot(rho1_B, vx, &sigma_Hall);

  ierr = VecDestroy(&vx);CHKERRXX(ierr);

  return sigma_Hall;
}

} // namespace anomtrans

#endif // ANOMTRANS_CONDUCTIVITY_H
