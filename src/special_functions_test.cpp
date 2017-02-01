#include <cfloat>
#include <cmath>
#include <iostream>
#include <sstream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "MPIPrettyUnitTestResultPrinter.h"
#include "special_functions.h"

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
 
  int ierr = PetscFinalize();

  return test_result;
}

TEST( Special_Functions, Fermi_Dirac ) {
  double tol = 10*DBL_EPSILON;

  double beta = 10.0;
  double E_below_min = 2*anomtrans::LN_DBL_MIN/beta;
  double E_above_max = -E_below_min;

  ASSERT_LT( abs(anomtrans::fermi_dirac(beta, E_below_min) - 1.0), tol );
  ASSERT_LT( abs(anomtrans::fermi_dirac(beta, E_above_max) - 0.0), tol );

  ASSERT_LT( abs(anomtrans::fermi_dirac(beta, 0.0) - 0.5), tol );
}
