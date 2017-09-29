#include <cstdlib>
#include <vector>
#include <tuple>
#include <iterator>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "util/MPIPrettyUnitTestResultPrinter.h"
#include "grid_basis.h"
#include "models/square_tb_spectrum.h"
#include "observables/energy.h"
#include "util/vec.h"

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

TEST( Energy, Square_TB_Energy ) {
  const std::size_t dim = 2;

  std::array<unsigned int, dim> Nk = {8, 8};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<dim> kmb(Nk, Nbands);

  double t = 1.0;
  double tp = -0.3;
  anomtrans::square_tb_Hamiltonian H(t, tp, Nk);

  Vec Ekm = anomtrans::get_energies(kmb, H);

  std::vector<PetscInt> local_rows;
  std::vector<PetscScalar> local_vals;
  std::tie(local_rows, local_vals) = anomtrans::get_local_contents(Ekm);

  for (anomtrans::stdvec_size i = 0; i < local_rows.size(); i++) {
    auto ikm_comps = kmb.decompose(local_rows.at(i));
    double energy = H.energy(ikm_comps);

    ASSERT_EQ( local_vals.at(i), energy );
  }

  PetscErrorCode ierr = VecDestroy(&Ekm);CHKERRXX(ierr);
}
