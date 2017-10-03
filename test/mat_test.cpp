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

/** @brief Use make_diag_Mat to create a diagonal matrix and verify it has
 *         the expected structure and values.
 */
TEST( mat, make_diag_Mat ) {
  PetscInt vec_len = 10;

  auto index_fn = [](PetscInt i)->PetscScalar {
    return 2.0 * i * i;
  };

  Vec v = anomtrans::vector_index_apply(vec_len, index_fn);

  Mat m = anomtrans::make_diag_Mat(v);

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(m, &begin, &end);CHKERRXX(ierr);

  PetscInt ncols;
  const PetscInt *cols;
  const PetscScalar *vals;

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    ierr = MatGetRow(m, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);

    ASSERT_EQ(ncols, 1);
    ASSERT_EQ(cols[0], local_row);
    ASSERT_EQ(vals[0], index_fn(local_row));

    ierr = MatRestoreRow(m, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
  }

  ierr = MatDestroy(&m);CHKERRXX(ierr);
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
  std::array<Mat, num_Mats> Bs;
  for (std::size_t d = 0; d < num_Mats; d++) {
    std::size_t nsq = (d + 1) * (d + 1);

    Bs.at(d) = anomtrans::make_Mat(Mat_dim, Mat_dim, Mat_dim);

    PetscInt begin, end;
    PetscErrorCode ierr = MatGetOwnershipRange(Bs.at(d), &begin, &end);CHKERRXX(ierr);

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
      ierr = MatSetValues(Bs.at(d), 1, &local_row, local_cols.size(), local_cols.data(),
          local_vals.data(), INSERT_VALUES);CHKERRXX(ierr);
    }

    ierr = MatAssemblyBegin(Bs.at(d), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    ierr = MatAssemblyEnd(Bs.at(d), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  }

  // Construct total and check elements.
  Mat total = anomtrans::Mat_from_sum_fn(coeffs, Bs, Mat_dim);

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(total, &begin, &end);CHKERRXX(ierr);

  auto macheps = std::numeric_limits<PetscReal>::epsilon();

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    PetscInt ncols;
    const PetscInt *cols;
    const PetscScalar *vals;
    ierr = MatGetRow(total, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);

    for (PetscInt col_index = 0; col_index < ncols; col_index++) {
      PetscInt col = cols[col_index];

      ASSERT_TRUE(anomtrans::scalars_approx_equal(vals[col_index], expected(local_row, col),
          -1.0, 10.0*macheps));
    }
    ierr = MatRestoreRow(total, local_row, &ncols, &cols, &vals);CHKERRXX(ierr);
  }

  ierr = MatDestroy(&total);CHKERRXX(ierr);
  for (std::size_t d = 0; d < num_Mats; d++) {
    ierr = MatDestroy(&(Bs.at(d)));CHKERRXX(ierr);
  }
}
