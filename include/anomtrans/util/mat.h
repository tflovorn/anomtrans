#ifndef ANOMTRANS_MAT_H
#define ANOMTRANS_MAT_H

#include <cstddef>
#include <cassert>
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

/** @brief Create a diagonal PETSC matrix, with diagonal elements given by `v`.
 */
Mat make_diag_Mat(Vec v);

/** @brief Set all diagonal entries of `M` to `alpha`.
 *  @pre `M` must be a square matrix.
 *  @todo MatDiagonalSet warns that it is slow if diagonal entries of `M` do not already have
 *        a value. What is a general way to avoid this?
 *  @todo Check that diagonal entries have values.
 *  @todo Possible to implement this function without allocating a Vec?
 */
void set_Mat_diagonal(Mat M, PetscScalar alpha);

/** @brief Return true iff each element of A and B is equal to within tol.
 *  @pre A and B should have the same global sizes and same local row
 *       distributions.
 *  @todo Support complex PetscScalar (compare absolute values).
 */
bool check_Mat_equal(Mat A, Mat B, double tol);

/** @brief Construct a matrix A = \sum_d coeffs(d) * Bs(d).
 *  @pre The length of coeffs and Bs should be at least 1.
 *  @pre All matrices in Bs should have the same dimensions.
 *  @todo Generate expected_elems_per_row from input matrix nnzs (number of
 *        nonzeros)?
 */
template<std::size_t len>
Mat Mat_from_sum_const(std::array<PetscScalar, len> coeffs, std::array<Mat, len> Bs,
    PetscInt expected_elems_per_row) {
  static_assert(len > 0, "must have at least 1 Mat for Mat_from_sum_const");

  auto coeff_fn = [coeffs](std::size_t d, PetscInt row, PetscInt col)->PetscScalar {
    return coeffs.at(d);
  };
  return Mat_from_sum_fn(coeff_fn, Bs, expected_elems_per_row);
}

/** @brief Calculate the commutator AB - BA.
 *  @todo Add parameter for MatMatMult fill parameter?
 *  @todo Add parameter for MatAYPX nonzero pattern?
 */
Mat commutator(Mat A, Mat B);

/** @brief Construct a matrix [A]_{ij} = \sum_d coeffs(d, i, j) * [Bs(d)]_{ij}.
 *  @param coeffs A function taking 3 arguments: a list dimension `d` from 0 to len
 *                and PetscInt values `i` and `j` denoting matrix row and column
 *                indices.
 *  @param Bs An array of matrices.
 *  @param expected_elems_per_row The number of elements per row to preallocate.
 *  @pre The length of Bs should be at least 1.
 *  @pre All matrices in Bs should have the same dimensions.
 *  @todo Generate expected_elems_per_row from input matrix nnzs (number of
 *        nonzeros)?
 *  @todo Would it be better to specialize this to constant/row/(row+column)
 *        cases to avoid function call for every element? If that is done, then
 *        have the problem that the data for coeffs(d, i, j) must be allocated.
 */
template<std::size_t len, typename F>
Mat Mat_from_sum_fn(F coeffs, std::array<Mat, len> Bs,
    PetscInt expected_elems_per_row) {
  // Need at least one matrix/coeff pair to have a well-defined output.
  static_assert(len > 0, "must have at least 1 Mat for Mat_from_sum_fn");

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
  for (std::size_t d = 0; d < len; d++) {
    PetscInt begin, end;
    PetscErrorCode ierr = MatGetOwnershipRange(Bs.at(d), &begin, &end);CHKERRXX(ierr);

    for (PetscInt local_row = begin; local_row < end; local_row++) {
      PetscInt ncols;
      const PetscInt *cols;
      const PetscScalar *vals;
      ierr = MatGetRow(Bs.at(d), local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
      
      std::vector<PetscInt> mult_cols;
      mult_cols.reserve(ncols);
      std::vector<PetscScalar> mult_vals;
      mult_vals.reserve(ncols);
      for (PetscInt col_index = 0; col_index < ncols; col_index++) {
        PetscScalar val = coeffs(d, local_row, col_index) * vals[col_index];
        // Ignore zeros.
        // TODO: ignore all values with magnitude below some epsilon?
        if (val == 0.0) {
          continue;
        }
        mult_cols.push_back(cols[col_index]);
        mult_vals.push_back(val);
      }
      ierr = MatSetValues(A, 1, &local_row, mult_cols.size(), mult_cols.data(), mult_vals.data(), ADD_VALUES);CHKERRXX(ierr);

      ierr = MatRestoreRow(Bs.at(d), local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
    }
  }

  PetscErrorCode ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  return A;
}

/** @brief Calculate the trace the product of the matrices given by xs, in the order of the
 *         elements of xs.
 *  @todo Add fill parameter as input?
 */
template <std::size_t num_Mats>
PetscScalar Mat_product_trace(std::array<Mat, num_Mats> xs) {
  static_assert(num_Mats > 0, "must have at least one Mat to trace over");

  // TODO - can use constexpr if when available.
  if (num_Mats == 1) {
    PetscScalar tr;
    PetscErrorCode ierr = MatGetTrace(xs.at(0), &tr);CHKERRXX(ierr);
    return tr;
  } else if (num_Mats == 2) {
    // TODO can optimize this?
    // Only diagonal elements of AB needed.
    Mat prod;
    PetscErrorCode ierr = MatMatMult(xs.at(0), xs.at(1), MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, &prod);CHKERRXX(ierr);

    PetscScalar tr;
    ierr = MatGetTrace(prod, &tr);CHKERRXX(ierr);

    ierr = MatDestroy(&prod);CHKERRXX(ierr);
    return tr;
  } else if (num_Mats == 3) {
    // TODO can optimize this?
    // Only diagonal elements of ABC needed.
    Mat prod;
    PetscErrorCode ierr = MatMatMatMult(xs.at(0), xs.at(1), xs.at(2), MAT_INITIAL_MATRIX,
        PETSC_DEFAULT, &prod);CHKERRXX(ierr);

    PetscScalar tr;
    ierr = MatGetTrace(prod, &tr);CHKERRXX(ierr);

    ierr = MatDestroy(&prod);CHKERRXX(ierr);
    return tr;
  } else {
    throw std::invalid_argument("num_Mats > 3 in Mat_product_trace not implemented");
  }
}

} // namespace anomtrans

#endif // ANOMTRANS_MAT_H
