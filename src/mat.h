#ifndef ANOMTRANS_MAT_H
#define ANOMTRANS_MAT_H

#include <cstddef>
#include <petscksp.h>
#include "grid_basis.h"

namespace anomtrans {

template <std::size_t k_dim>
Mat make_Mat(kmBasis<k_dim> kmb, PetscInt expected_elems_per_row) {
  Mat M;
  PetscErrorCode ierr = MatCreate(PETSC_COMM_WORLD, &M);CHKERRXX(ierr);
  ierr = MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE,
      kmb.end_ikm, kmb.end_ikm);CHKERRXX(ierr);

  // TODO do we want to use MatSetFromOptions here instead?
  ierr = MatSetType(M, MATMPIAIJ);CHKERRXX(ierr);
  // The two expected_elems_per_row arguments below give the elements to preallocate per
  // row in the 'diagonal part' and 'off-diagonal part' of the matrix respectively.
  // The diagonal part is the block (r1,r2)x(c1,c2) where rows r1->r2 belong to this
  // process and columns c1->c2 belong to a vector owned by this process.
  // The off-diagonal part is the remaining columns.
  // It's not worth it here to think too hard about this distinction, so allocate enough
  // for both cases of all elements in the diagonal part or all elements in the
  // off-diagonal part (or any other distribution in between).
  // TODO can/should we be more precise about this?
  // Preallocating a bit too much here is not really a problem unless we are
  // very tight on memory.
  ierr = MatMPIAIJSetPreallocation(M, expected_elems_per_row, nullptr,
      expected_elems_per_row, nullptr);CHKERRXX(ierr);
  // Since we specified the type MATMPIAIJ above, won't call preallocation for MatSeq also.
  // From inspection of the implementation it looks like there would be no meaningful
  // performance penalty for calling both (calling the Seq preallocation here would
  // look for a method on the MPIAIJ matrix that doesn't exist, see this, and return).
  // Should call both if we use MatSetFromOptions above instead of MatSetType.

  return M;
}

} // namespace anomtrans

#endif // ANOMTRANS_MAT_H
