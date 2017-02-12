#ifndef ANOMTRANS_MAT_H
#define ANOMTRANS_MAT_H

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <array>
#include <vector>
#include <petscksp.h>

namespace anomtrans {

/** @brief Create and preallocate a m x n PETSc matrix. The returned matrix
 *         is ready for MatSetValues() calls. Before use in matrix-vector
 *         products, MatAssemblyBegin() and MatAssemblyEnd() should be called.
 *  @param m First dimension of the matrix to construct (m x n matrix).
 *  @param n Second dimension of the matrix to construct (m x n matrix).
 *  @param expected_elems_per_row Maximum number of elements in one row
 *                                expected to be included; used to preallocate
 *                                memory.
 */
Mat make_Mat(PetscInt m, PetscInt n, PetscInt expected_elems_per_row);

/** @brief Return true iff each element of A and B is equal to within tol.
 *  @pre A and B should have the same global sizes and same local row
 *       distributions.
 *  @todo Support complex PetscScalar (compare absolute values).
 */
bool check_Mat_equal(Mat A, Mat B, double tol);

/** @brief Construct a matrix A = \sum_i coeffs(i) * Bs(i).
 *  @pre The length of coeffs and Bs should be at least 1.
 *  @pre All matrices in Bs should have the same dimensions.
 *  @todo Generate expected_elems_per_row from input matrix nnzs (number of
 *        nonzeros)?
 */
template<std::size_t len>
Mat Mat_from_sum(std::array<PetscScalar, len> coeffs, std::array<Mat, len> Bs,
    PetscInt expected_elems_per_row) {
  // Need at least one matrix/coeff pair to have a well-defined output.
  if (len == 0) {
    throw std::invalid_argument("must have at least one coeff/matrix to add");
  }

  // Get dimensions of Ms and check that they are all equal.
  std::vector<PetscInt> ms, ns;
  for (auto B : Bs) {
    PetscInt this_m, this_n;
    PetscErrorCode ierr = MatGetSize(B, &this_m, &this_n);CHKERRXX(ierr);
    ms.push_back(this_m);
    ns.push_back(this_n);
  }
  // TODO would like to replace this loop with zip(ms, ns)
  // Combination of boost::zip_iterator and C++17 structured bindings would
  // make a convenient zip loop.
  // https://stackoverflow.com/questions/8511035/sequence-zip-function-for-c11
  for (std::vector<PetscInt>::size_type i = 0; i < ms.size(); i++) {
    if (ms.at(i) != ns.at(i)) {
      throw std::invalid_argument("matrices in Bs must have the same size");
    }
  }

  Mat A = make_Mat(ms.at(0), ns.at(0), expected_elems_per_row);

  // Build the sum.
  // Do this using MatGetRow/MatSetValues to avoid losing the preallocation
  // of A and changing the sparse structure with each addition.
  // TODO is this the correct usage, or would repeated application of
  // MatAXPY be better? Could benchmark this.
  // TODO would like to replace this loop with zip(coeffs, Bs)
  for (std::size_t i = 0; i < len; i++) {
    PetscInt begin, end;
    PetscErrorCode ierr = MatGetOwnershipRange(Bs.at(i), &begin, &end);CHKERRXX(ierr);

    for (PetscInt local_row = begin; local_row < end; local_row++) {
      PetscInt ncols;
      const PetscInt *cols;
      const PetscScalar *vals;
      ierr = MatGetRow(Bs.at(i), local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
      
      std::vector<PetscInt> mult_cols;
      mult_cols.reserve(ncols);
      std::vector<PetscScalar> mult_vals;
      mult_vals.reserve(ncols);
      for (PetscInt col_index = 0; col_index < ncols; col_index++) {
        PetscScalar val = coeffs.at(i) * vals[col_index];
        // Ignore zeros.
        // TODO: ignore all values with magnitude below some epsilon?
        if (val == 0.0) {
          continue;
        }
        mult_cols.push_back(cols[col_index]);
        mult_vals.push_back(val);
      }
      ierr = MatSetValues(A, 1, &local_row, mult_cols.size(), mult_cols.data(), mult_vals.data(), ADD_VALUES);CHKERRXX(ierr);

      ierr = MatRestoreRow(Bs.at(i), local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
    }
  }

  PetscErrorCode ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  return A;
}

} // namespace anomtrans

#endif // ANOMTRANS_MAT_H
