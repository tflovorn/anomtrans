#include "mat.h"

namespace anomtrans {

Mat make_Mat(PetscInt m, PetscInt n, PetscInt expected_elems_per_row) {
  Mat A;
  PetscErrorCode ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRXX(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n);CHKERRXX(ierr);

  // TODO do we want to use MatSetFromOptions here instead?
  ierr = MatSetType(A, MATMPIAIJ);CHKERRXX(ierr);
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
  ierr = MatMPIAIJSetPreallocation(A, expected_elems_per_row, nullptr,
      expected_elems_per_row, nullptr);CHKERRXX(ierr);
  // Since we specified the type MATMPIAIJ above, won't call preallocation for MatSeq also.
  // From inspection of the implementation it looks like there would be no meaningful
  // performance penalty for calling both (calling the Seq preallocation here would
  // look for a method on the MPIAIJ matrix that doesn't exist, see this, and return).
  // Should call both if we use MatSetFromOptions above instead of MatSetType.
  //
  // TODO is using MatXAIJSetPreallocation the right thing to do here instead?

  return A;
}

bool check_Mat_equal(Mat A, Mat B, double tol) {
  PetscInt A_m, A_n, B_m, B_n;
  PetscErrorCode ierr = MatGetSize(A, &A_m, &A_n);CHKERRXX(ierr);
  ierr = MatGetSize(B, &B_m, &B_n);CHKERRXX(ierr);
  if (A_m != B_m or A_n != B_n) {
    throw std::invalid_argument("A and B must have same global sizes");
  }

  PetscInt A_begin, A_end, B_begin, B_end;
  ierr = MatGetOwnershipRange(A, &A_begin, &A_end);CHKERRXX(ierr);
  ierr = MatGetOwnershipRange(B, &B_begin, &B_end);CHKERRXX(ierr);
  if (A_begin != B_begin or A_end != B_end) {
    throw std::invalid_argument("A and B must have same local row ownership ranges");
  }

  for (PetscInt local_row = A_begin; local_row < A_end; local_row++) {
    PetscInt ncols_A, ncols_B;
    const PetscInt *cols_A, *cols_B;
    const PetscScalar *vals_A, *vals_B;
    ierr = MatGetRow(A, local_row, &ncols_A, &cols_A, &vals_A);CHKERRXX(ierr);
    ierr = MatGetRow(B, local_row, &ncols_B, &cols_B, &vals_B);CHKERRXX(ierr);

    // TODO would like common method to handle the paths to MatRestoreRow calls here.
    // Could wrap the GetRow/RestoreRow in a class constructor/destructor (using RAII).
    if (ncols_A != ncols_B) {
      ierr = MatRestoreRow(A, local_row, &ncols_A, &cols_A, &vals_A);CHKERRXX(ierr);
      ierr = MatRestoreRow(B, local_row, &ncols_B, &cols_B, &vals_B);CHKERRXX(ierr);
      return false;
    }

    for (PetscInt i = 0; i < ncols_A; i++) {
      if (cols_A[i] != cols_B[i] or abs(vals_A[i] - vals_B[i]) > tol) {
        ierr = MatRestoreRow(A, local_row, &ncols_A, &cols_A, &vals_A);CHKERRXX(ierr);
        ierr = MatRestoreRow(B, local_row, &ncols_B, &cols_B, &vals_B);CHKERRXX(ierr);
        return false;
      }
    }

    ierr = MatRestoreRow(A, local_row, &ncols_A, &cols_A, &vals_A);CHKERRXX(ierr);
    ierr = MatRestoreRow(B, local_row, &ncols_B, &cols_B, &vals_B);CHKERRXX(ierr);
  }

  return true;
}

} // namespace anomtrans
