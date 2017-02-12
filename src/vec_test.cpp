#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "MPIPrettyUnitTestResultPrinter.h"
#include "vec.h"

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

// Map vector of values i -> i^2 and check that the result is correct.
TEST( Vector_Apply, Square ) {
  PetscInt v_in_size = 32;
  std::vector<PetscInt> global_in_rows;
  std::vector<PetscScalar> global_in_vals;
  for (PetscInt i = 0; i < v_in_size; i++) {
    global_in_rows.push_back(i);
    global_in_vals.push_back(i);
  }

  Vec v_in;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, v_in_size, &v_in);CHKERRXX(ierr);
  ierr = VecSetValues(v_in, v_in_size, global_in_rows.data(), global_in_vals.data(), INSERT_VALUES);CHKERRXX(ierr);
  ierr = VecAssemblyBegin(v_in);CHKERRXX(ierr);
  ierr = VecAssemblyEnd(v_in);CHKERRXX(ierr);

  auto f = [](PetscScalar x)->PetscScalar {
    return x*x;
  };

  Vec v_out = anomtrans::vector_elem_apply(v_in, f);

  std::vector<PetscInt> local_in_rows;
  std::vector<PetscScalar> local_in_vals;
  std::tie(local_in_rows, local_in_vals) = anomtrans::get_local_contents(v_in);

  std::vector<PetscInt> local_out_rows;
  std::vector<PetscScalar> local_out_vals;
  std::tie(local_out_rows, local_out_vals) = anomtrans::get_local_contents(v_out);

  ASSERT_EQ(local_in_rows, local_out_rows);

  for (anomtrans::stdvec_size i = 0; i < local_in_rows.size(); i++) {
    PetscScalar this_in = local_in_vals.at(i);
    PetscScalar expected_out = f(this_in);

    ASSERT_EQ( local_out_vals.at(i), expected_out );
  }

  ierr = VecDestroy(&v_in);CHKERRXX(ierr);
  ierr = VecDestroy(&v_out);CHKERRXX(ierr);
}
