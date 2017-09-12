#include <cstddef>
#include <tuple>
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

// Create vectors of values -n, -n + 1, -n + 2, ..., n - 1 and
// -n + 1, -n + 2, ..., n - 1, n and look for the maximum absolute value.
// Should be n for both vectors.
TEST( Vector_Index_Apply_Get_MaxAbs, IntSequence ) {
  PetscInt n = 10;
  PetscInt num_rows = 2*n;
  auto f1 = [n](PetscInt i)->PetscScalar {
    return -n + i;
  };
  auto f2 = [n](PetscInt i)->PetscScalar {
    return -n + i + 1;
  };
  Vec v1 = anomtrans::vector_index_apply(num_rows, f1);
  Vec v2 = anomtrans::vector_index_apply(num_rows, f2);

  PetscReal tol = 1e-12;
  ASSERT_NEAR( anomtrans::get_Vec_MaxAbs(v1), n, tol );
  ASSERT_NEAR( anomtrans::get_Vec_MaxAbs(v2), n, tol );

  auto rv1 = anomtrans::get_local_contents(v1);
  for (std::size_t i = 0; i < std::get<0>(rv1).size(); i++) {
    PetscInt global_index = std::get<0>(rv1).at(i);
    ASSERT_NEAR( std::get<1>(rv1).at(i).real(), f1(global_index).real(), tol );
    ASSERT_NEAR( std::get<1>(rv1).at(i).imag(), f1(global_index).imag(), tol );
  }

  auto rv2 = anomtrans::get_local_contents(v2);
  for (std::size_t i = 0; i < std::get<0>(rv2).size(); i++) {
    PetscInt global_index = std::get<0>(rv2).at(i);
    ASSERT_NEAR( std::get<1>(rv2).at(i).real(), f2(global_index).real(), tol );
    ASSERT_NEAR( std::get<1>(rv2).at(i).imag(), f2(global_index).imag(), tol );
  }

  PetscErrorCode ierr = VecDestroy(&v1);CHKERRXX(ierr);
  ierr = VecDestroy(&v2);CHKERRXX(ierr);
}
