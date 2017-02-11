#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "MPIPrettyUnitTestResultPrinter.h"
#include "grid_basis.h"
#include "vec.h"
#include "square_tb_spectrum.h"
#include "energy.h"
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
 
  PetscErrorCode ierr = PetscFinalize();CHKERRXX(ierr);

  return test_result;
}

// TODO? could make this more of a unit test: remove dependencies on energy and rho0.
TEST( Vector_Apply, Square_TB_Energy_Fermi ) {
  const std::size_t dim = 2;

  std::array<unsigned int, dim> Nk = {8, 8};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<dim> kmb(Nk, Nbands);

  double t = 1.0;
  double tp = -0.3;
  anomtrans::square_tb_Hamiltonian H(t, tp, Nk);
  double beta = 10.0*t;

  auto fd = [beta](double E)->double {
    return anomtrans::fermi_dirac(beta, E);
  };

  Vec Ekm = anomtrans::get_energies(kmb, H);
  Vec rho0_km = anomtrans::vector_elem_apply(kmb, Ekm, fd);

  std::vector<PetscInt> local_E_rows;
  std::vector<PetscScalar> local_E_vals;
  std::tie(local_E_rows, local_E_vals) = anomtrans::get_local_contents(Ekm);

  std::vector<PetscInt> local_rho0_rows;
  std::vector<PetscScalar> local_rho0_vals;
  std::tie(local_rho0_rows, local_rho0_vals) = anomtrans::get_local_contents(rho0_km);

  ASSERT_EQ(local_E_rows, local_rho0_rows);

  for (anomtrans::stdvec_size i = 0; i < local_E_rows.size(); i++) {
    double energy = local_E_vals.at(i);
    double rho0 = fd(energy);

    ASSERT_EQ( local_rho0_vals.at(i), rho0 );
  }

  PetscErrorCode ierr = VecDestroy(&Ekm);CHKERRXX(ierr);
  ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);
}
