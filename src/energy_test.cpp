#include <cstdlib>
#include <iostream>
#include <vector>
#include <iterator>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "MPIPrettyUnitTestResultPrinter.h"
#include "grid_basis.h"
#include "square_tb_spectrum.h"
#include "energy.h"

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

TEST( Energy, Square_TB_Energy ) {
  const std::size_t dim = 2;

  std::array<unsigned int, dim> Nk = {8, 8};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<dim> kmb(Nk, Nbands);

  double t = 1.0;
  double tp = -0.3;
  anomtrans::square_tb_Hamiltonian H(t, tp, Nk);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  Vec Ekm = anomtrans::get_energies(kmb, H);

  PetscInt begin, end;
  PetscErrorCode ierr = VecGetOwnershipRange(Ekm, &begin, &end);CHKERRXX(ierr);
  PetscInt num_local_rows = end - begin;
  ASSERT_GE( num_local_rows, 0 );
  ASSERT_LE( num_local_rows, kmb.end_ikm ); 

  // TODO factor out this pattern for getting a std::vector range of elements
  // from a Vec.
  std::vector<PetscInt> local_rows;

  local_rows.reserve(static_cast<std::vector<PetscInt>::size_type>(num_local_rows));

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    local_rows.push_back(local_row);
  }

  std::vector<PetscScalar> local_vals(num_local_rows);
  VecGetValues(Ekm, num_local_rows, local_rows.data(), local_vals.data());

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    auto ikm_comps = kmb.decompose(local_row);
    double energy = H.energy(ikm_comps);

    ASSERT_EQ( local_vals.at(local_row - begin), energy );
  }
}
