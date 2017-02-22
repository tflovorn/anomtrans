#include <utility>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "MPIPrettyUnitTestResultPrinter.h"
#include "util.h"

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

TEST( cross, unit_vectors ) {
  PetscScalar tol = 1e-12;
  // Check that \hat{x} cross \hat{y} = \hat{z} and cyclic permutations.
  for (std::size_t d = 0; d < 3; d++) {
    std::array<PetscScalar, 3> u = {0.0, 0.0, 0.0};
    std::array<PetscScalar, 3> v = {0.0, 0.0, 0.0};
    u.at(d) = 1.0;
    v.at((d + 1) % 3) = 1.0;

    auto u_cross_v = anomtrans::cross(u, v);
    ASSERT_NEAR( u_cross_v.at(d), 0.0, tol );
    ASSERT_NEAR( u_cross_v.at((d + 1) % 3), 0.0, tol );
    ASSERT_NEAR( u_cross_v.at((d + 2) % 3), 1.0, tol );
  }
  // Check that \hat{x} cross \hat{y} = -\hat{z} and cyclic permutations.
  for (std::size_t d = 0; d < 3; d++) {
    std::array<PetscScalar, 3> u = {0.0, 0.0, 0.0};
    std::array<PetscScalar, 3> v = {0.0, 0.0, 0.0};
    u.at((d + 1) % 3) = 1.0;
    v.at(d) = 1.0;

    auto u_cross_v = anomtrans::cross(u, v);
    ASSERT_NEAR( u_cross_v.at(d), 0.0, tol );
    ASSERT_NEAR( u_cross_v.at((d + 1) % 3), 0.0, tol );
    ASSERT_NEAR( u_cross_v.at((d + 2) % 3), -1.0, tol );
  }
  // Check that \hat{x} cross \hat{y} = \hat{z} when \hat{x} and \hat{y}
  // are given as 2D vectors.
  std::array<PetscScalar, 2> x_2d = {1.0, 0.0};
  std::array<PetscScalar, 2> y_2d = {0.0, 1.0};

  auto x_cross_y = anomtrans::cross(x_2d, y_2d);
  ASSERT_NEAR( x_cross_y.at(0), 0.0, tol );
  ASSERT_NEAR( x_cross_y.at(1), 0.0, tol );
  ASSERT_NEAR( x_cross_y.at(2), 1.0, tol );
}

TEST( wrap, correct_wrapping ) {
  PetscInt N = 10;
  for (PetscInt x = 0; x < 3*N; x++) {
    ASSERT_EQ( anomtrans::wrap(x, N), x % N );
  }
  for (PetscInt x = -1; x > -(N+1); x--) {
    ASSERT_EQ( anomtrans::wrap(x, N), N + x );
  }
  for (PetscInt x = -(N+1); x > -(2*N+1); x--) {
    ASSERT_EQ( anomtrans::wrap(x, N), N + (x + N) );
  }
}

TEST( invert_vals_indices, short_list ) {
  using SIPair = std::pair<PetscScalar, PetscInt>;
  std::vector<SIPair> xs = {SIPair{1.0, 3}, SIPair{2.0, 0}, SIPair{10.0, 2}, SIPair{3.0, 1}};
  std::vector<PetscInt> ys = anomtrans::invert_vals_indices(xs);
  for (std::size_t i = 0; i < ys.size(); i++) {
    ASSERT_EQ( ys.at(xs.at(i).second), static_cast<PetscInt>(i) );
  }
}
