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
  // TODO may want to just pass ikm to H: look up precomputed v without
  // conversion to ikm_comps and back.
  auto vx_elem = [kmb, H](PetscInt ikm)->PetscScalar {
    auto ikm_comps = kmb.decompose(ikm);
    auto velocity = H.velocity(ikm_comps);
    return std::get<0>(velocity);
  };

  Vec vx = vector_index_apply(kmb.end_ikm, vx_elem);

  // Make sure vx and rho1_B have the same local row distribution.
  // TODO this full check would not be necessary if we could assume that vx and
  // rho1_B were created with the same row distribution (which they would
  // be if they have the same number of global elements and were created
  // with local elements set by PETSC_DECIDE). This would be a benefit
  // of wrapping Vec in a class (can just check if instances of the Vec
  // class have the same number of global rows, and always create with
  // PETSC_DECIDE setting local rows).
  PetscInt begin_vx, end_vx;
  PetscErrorCode ierr = VecGetOwnershipRange(vx, &begin_vx, &end_vx);CHKERRXX(ierr);

  PetscInt begin_rho, end_rho;
  ierr = VecGetOwnershipRange(rho1_B, &begin_rho, &end_rho);CHKERRXX(ierr);
  if (begin_vx != begin_rho or end_vx != end_rho) {
    ierr = VecDestroy(&vx);CHKERRXX(ierr);
    throw std::runtime_error("got different row distribution for vx and rho");
  }

  // Get Hall conductivity: sigma_Hall = -e vx^T rho1_B
  PetscScalar sigma_Hall;
  // VecDot(u, v) = v^{\dagger} u
  ierr = VecDot(rho1_B, vx, &sigma_Hall);

  ierr = VecDestroy(&vx);CHKERRXX(ierr);

  return sigma_Hall;
}

} // namespace anomtrans

#endif // ANOMTRANS_CONDUCTIVITY_H
