#include <cstddef>
#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "util/MPIPrettyUnitTestResultPrinter.h"
#include "util/vec.h"
#include "util/mat.h"
#include "util/util.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

  testing::TestEventListener *default_rp = listeners.default_result_printer();
  listeners.Release(default_rp);
  listeners.Append(new anomtrans::MPIPrettyUnitTestResultPrinter(default_rp, rank));

  int test_result = RUN_ALL_TESTS();
 
  PetscErrorCode ierr = PetscFinalize();CHKERRXX(ierr);

  return test_result;
}

/** @brief Check that commutator(A, B) correctly calculates AB - BA.
 */
TEST( mat, commutator ) {
  anomtrans::OwnedMat A = anomtrans::make_Mat(2, 2, 2);
  anomtrans::OwnedMat B = anomtrans::make_Mat(2, 2, 2);

  std::vector<PetscInt> used_cols = {0, 1};
  std::vector<std::vector<PetscScalar>> A_elems = {{1.0, 2.0}, {3.0, 4.0}};
  std::vector<std::vector<PetscScalar>> B_elems = {{5.0, 6.0}, {7.0, 8.0}};

  // Construct A = [[1, 2], and B = [[5, 6],
  //                [3, 4]]          [7, 8]].
  // TODO - add a utility function to generate a PETSc Mat from an
  // Eigen Matrix to generalize this. Eigen Matrix construction is transparent.
  // Also add function to convert the other way, from PETSc Mat to Eigen Matrix.
  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(A.M, &begin, &end);CHKERRXX(ierr);

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    std::vector<PetscScalar> A_row_elems = A_elems.at(local_row);
    std::vector<PetscScalar> B_row_elems = B_elems.at(local_row);

    assert(used_cols.size() == A_row_elems.size());
    assert(used_cols.size() == B_row_elems.size());
    ierr = MatSetValues(A.M, 1, &local_row, used_cols.size(), used_cols.data(),
        A_row_elems.data(), INSERT_VALUES);CHKERRXX(ierr);
    ierr = MatSetValues(B.M, 1, &local_row, used_cols.size(), used_cols.data(),
        B_row_elems.data(), INSERT_VALUES);CHKERRXX(ierr);
  }

  ierr = MatAssemblyBegin(A.M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A.M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  ierr = MatAssemblyBegin(B.M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(B.M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  // AB - BA = [[-4, 12],
  //            [12, 4]].   Check that commutator(A, B) has this value.
  anomtrans::OwnedMat comm = anomtrans::commutator(A.M, B.M);

  std::vector<std::vector<PetscScalar>> expected_elems = {{-4.0, -12.0}, {12.0, 4.0}};

  auto macheps = std::numeric_limits<PetscReal>::epsilon();

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    PetscInt ncols;
    const PetscInt *cols;
    const PetscScalar *vals;
    ierr = MatGetRow(comm.M, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);

    for (PetscInt col_index = 0; col_index < ncols; col_index++) {
      PetscInt col = cols[col_index];
      PetscScalar val = vals[col_index];

      ASSERT_TRUE(anomtrans::scalars_approx_equal(expected_elems.at(local_row).at(col), val,
            -1.0, 10.0*macheps));
    }

    ierr = MatRestoreRow(comm.M, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
  }
}

/** @brief Use make_diag_Mat to create a diagonal matrix and verify it has
 *         the expected structure and values.
 */
TEST( mat, make_diag_Mat ) {
  PetscInt vec_len = 10;

  auto index_fn = [](PetscInt i)->PetscScalar {
    return 2.0 * i * i;
  };

  Vec v = anomtrans::vector_index_apply(vec_len, index_fn);

  anomtrans::OwnedMat m = anomtrans::make_diag_Mat(v);

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(m.M, &begin, &end);CHKERRXX(ierr);

  PetscInt ncols;
  const PetscInt *cols;
  const PetscScalar *vals;

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    ierr = MatGetRow(m.M, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);

    ASSERT_EQ(ncols, 1);
    ASSERT_EQ(cols[0], local_row);
    ASSERT_EQ(vals[0], index_fn(local_row));

    ierr = MatRestoreRow(m.M, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
  }

  ierr = VecDestroy(&v);CHKERRXX(ierr);
}

/** @brief Use mat_from_sum_fn to construct a matrix and verify that it has
 *         the correct elements.
 */
TEST( mat, Mat_from_sum_fn ) {
  const std::size_t num_Mats = 3;
  const PetscInt Mat_dim = 4;

  // Want to construct a matrix with these elements:
  // [A]_{ij} = \sum_{n = 1}^3 (ni + nj) * (n^2 i + n^2 j)
  auto expected = [](PetscInt i, PetscInt j)->PetscScalar {
    PetscScalar result = 0.0;
    for (std::size_t d = 0; d < num_Mats; d++) {
      std::size_t n = d + 1;
      result += (n*(i + 1) + n*(j + 1)) * (n*n*(i + 1) + n*n*(j + 1));
    }

    return result;
  };

  // Function each matrix will be multiplied by elementwise.
  auto coeffs = [](std::size_t d, PetscInt row, PetscInt col)->PetscScalar {
    std::size_t n = d + 1;
    return n*(row + 1) + n*(col + 1);
  };

  // Matrices to combine.
  std::array<anomtrans::OwnedMat, num_Mats> Bs;
  for (std::size_t d = 0; d < num_Mats; d++) {
    std::size_t nsq = (d + 1) * (d + 1);

    Bs.at(d) = anomtrans::make_Mat(Mat_dim, Mat_dim, Mat_dim);

    PetscInt begin, end;
    PetscErrorCode ierr = MatGetOwnershipRange(Bs.at(d).M, &begin, &end);CHKERRXX(ierr);

    for (PetscInt local_row = begin; local_row < end; local_row++) {
      std::vector<PetscInt> local_cols;
      local_cols.reserve(Mat_dim);
      std::vector<PetscScalar> local_vals;
      local_vals.reserve(Mat_dim);

      for (PetscInt col = 0; col < Mat_dim; col++) {
        local_cols.push_back(col);
        local_vals.push_back(nsq * (local_row + 1) + nsq * (col + 1));
      }

      assert(local_cols.size() == local_vals.size());
      ierr = MatSetValues(Bs.at(d).M, 1, &local_row, local_cols.size(), local_cols.data(),
          local_vals.data(), INSERT_VALUES);CHKERRXX(ierr);
    }

    ierr = MatAssemblyBegin(Bs.at(d).M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    ierr = MatAssemblyEnd(Bs.at(d).M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  }

  // Construct total and check elements.
  anomtrans::OwnedMat total = anomtrans::Mat_from_sum_fn(coeffs, anomtrans::unowned(Bs), Mat_dim);

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(total.M, &begin, &end);CHKERRXX(ierr);

  auto macheps = std::numeric_limits<PetscReal>::epsilon();

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    PetscInt ncols;
    const PetscInt *cols;
    const PetscScalar *vals;
    ierr = MatGetRow(total.M, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);

    for (PetscInt col_index = 0; col_index < ncols; col_index++) {
      PetscInt col = cols[col_index];

      ASSERT_TRUE(anomtrans::scalars_approx_equal(vals[col_index], expected(local_row, col),
          -1.0, 10.0*macheps));
    }
    ierr = MatRestoreRow(total.M, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
  }
}
