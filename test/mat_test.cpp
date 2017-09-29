#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "util/MPIPrettyUnitTestResultPrinter.h"
#include "util/vec.h"
#include "util/mat.h"

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
