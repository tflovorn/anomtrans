#include <cstddef>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include <json.hpp>
#include "MPIPrettyUnitTestResultPrinter.h"
#include "util.h"
#include "grid_basis.h"
#include "square_tb_spectrum.h"
#include "energy.h"
#include "vec.h"
#include "rho0.h"
#include "collision.h"
#include "driving.h"

using json = nlohmann::json;

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
 
  int ierr = PetscFinalize();CHKERRXX(ierr);

  return test_result;
}

TEST( Driving, square_TB_Hall ) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  const std::size_t k_dim = 2;

  std::array<unsigned int, k_dim> Nk = {128, 128};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands);

  double t = 1.0;
  double tp = -0.3;
  anomtrans::square_tb_Hamiltonian H(t, tp, Nk);

  std::array<double, k_dim> a1 = {1.0, 0.0};
  std::array<double, k_dim> a2 = {0.0, 1.0};
  anomtrans::DimMatrix<k_dim> D = {a1, a2};

  double beta = 10.0/t;

  // Hall effect: chage current in x, magnetic field in z --> electric field in y.
  // By Onsager reciprocity, this is equivalent to:
  // electric field in y, magnetic field in -z --> charge current in x.
  // We will calculate the current linear response to these fields, i.e. the
  // Hall conductivity.
  // TODO is this correct? Is it true that there is a sign change in B in the
  // reciprocal system?
  std::array<double, k_dim> Ehat = {0.0, 1.0};
  std::array<double, 3> Bhat = {0.0, 0.0, -1.0};

  // TODO build a mechanism for choosing a useful value of spread.
  // Choosing an appropriate value of spread is somewhat tricky:
  // it is a balance between getting close to the spread->0 limit and
  // choosing a value large enough that the Gaussian peaks are adequately
  // sampled by the given k-grid. The required sampling density is
  // related to spread and to dE/dk.
  // The spread->0 limit can only be approached when the k-grid also becomes
  // infinitely dense.
  double spread = 0.1*t;

  Vec Ekm = anomtrans::get_energies(kmb, H);

  PetscInt Ekm_min_index, Ekm_max_index;
  PetscReal Ekm_min, Ekm_max;
  PetscErrorCode ierr = VecMin(Ekm, &Ekm_min_index, &Ekm_min);CHKERRXX(ierr);
  ierr = VecMax(Ekm, &Ekm_max_index, &Ekm_max);CHKERRXX(ierr);

  auto disorder_term = [kmb, H](PetscInt ikm1, PetscInt ikm2, PetscInt ikm3, PetscInt ikm4)->double {
    return anomtrans::on_site_diagonal_disorder(kmb, H, ikm1, ikm2, ikm3, ikm4);
  };

  // TODO include finite disorder correlation length
  Mat collision = anomtrans::make_collision(kmb, H, spread, disorder_term);

  // The collision matrix K should be symmetric. Knowing that this is the case
  // substantially simplifies solution of the associated linear systems.
  // TODO assuming that PetscScalar is not complex here. If it is complex, we
  // should check that K is Hermitian.
  PetscBool K_is_symmetric;
  PetscReal tol = 1e-12;
  ierr = MatIsSymmetric(collision, tol, &K_is_symmetric);CHKERRXX(ierr);
  ASSERT_TRUE( K_is_symmetric );

  ierr = MatSetOption(collision, MAT_SYMMETRIC, PETSC_TRUE);CHKERRXX(ierr);

  // Create the linear solver context.
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRXX(ierr);
  // This uses collision again as the preconditioning matrix.
  // TODO - is there a better choice?
  ierr = KSPSetOperators(ksp, collision, collision);CHKERRXX(ierr);
  // Could use KSPSetFromOptions here. In this case, prefer to keep options
  // hard-coded to have identical output from each test run.

  const unsigned int deriv_order = 2;
  Mat Dbar_E = anomtrans::driving_electric(D, kmb, deriv_order, Ehat);
  Mat Dbar_B = anomtrans::driving_magnetic(D, kmb, deriv_order, H, Bhat);

  unsigned int num_mus = 40;
  auto mus = anomtrans::linspace(Ekm_min, Ekm_max, num_mus);

  std::vector<std::vector<PetscScalar>> all_rho0;
  std::vector<std::vector<PetscScalar>> all_rhs_B0;
  std::vector<std::vector<PetscScalar>> all_rho1_B0;
  std::vector<std::vector<PetscScalar>> all_rhs_Bfinite;
  std::vector<std::vector<PetscScalar>> all_rho1_Bfinite;
  // For each mu, solve the pair of equations:
  // K rho1_B0 = Dbar_E rho0
  // K rho1_Bfinite = Dbar_B rho1_B0
  for (auto mu : mus) {
    Vec rho0_km = anomtrans::make_rho0(Ekm, beta, mu);

    // Set nullspace of K: K rho0_km = 0.
    // Note that this is true regardless of the value of mu
    // (energies only enter K through differences).
    // It is also true for any value of beta (the Fermi-Dirac distribution
    // function does not appear in K, only energy differences).
    MatNullSpace nullspace;
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &rho0_km, &nullspace);CHKERRXX(ierr);
    ierr = MatSetNullSpace(collision, nullspace);CHKERRXX(ierr);
    // NOTE rho0_km must not be modified after this call until we are done with nullspace.

    // Need to initialize rhs_B0, but we don't care what is in it before
    // we call MatMult.
    Vec rhs_B0;
    ierr = VecDuplicate(rho0_km, &rhs_B0);CHKERRXX(ierr);
    ierr = MatMult(Dbar_E, rho0_km, rhs_B0);CHKERRXX(ierr);

    // Need to initialize rho1_B0, but we don't care what is in it before
    // we call KSPSolve.
    Vec rho1_B0;
    ierr = VecDuplicate(rho0_km, &rho1_B0);CHKERRXX(ierr);
    ierr = KSPSolve(ksp, rhs_B0, rho1_B0);CHKERRXX(ierr);

    Vec rhs_Bfinite;
    ierr = VecDuplicate(rho0_km, &rhs_Bfinite);CHKERRXX(ierr);
    ierr = MatMult(Dbar_B, rho1_B0, rhs_Bfinite);CHKERRXX(ierr);

    Vec rho1_Bfinite;
    ierr = VecDuplicate(rho0_km, &rho1_Bfinite);CHKERRXX(ierr);
    ierr = KSPSolve(ksp, rhs_Bfinite, rho1_Bfinite);CHKERRXX(ierr);

    auto collected_rho0 = anomtrans::collect_contents(rho0_km);
    all_rho0.push_back(collected_rho0);
    auto collected_rhs_B0 = anomtrans::collect_contents(rhs_B0);
    all_rhs_B0.push_back(collected_rhs_B0);
    auto collected_rho1_B0 = anomtrans::collect_contents(rho1_B0);
    all_rho1_B0.push_back(collected_rho1_B0);
    auto collected_rhs_Bfinite = anomtrans::collect_contents(rhs_Bfinite);
    all_rhs_Bfinite.push_back(collected_rhs_Bfinite);
    auto collected_rho1_Bfinite = anomtrans::collect_contents(rho1_Bfinite);
    all_rho1_Bfinite.push_back(collected_rho1_Bfinite);

    ierr = VecDestroy(&rho1_Bfinite);CHKERRXX(ierr);
    ierr = VecDestroy(&rhs_Bfinite);CHKERRXX(ierr);
    ierr = VecDestroy(&rho1_B0);CHKERRXX(ierr);
    ierr = VecDestroy(&rhs_B0);CHKERRXX(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRXX(ierr);
    ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);
  }

  auto collected_Ekm = anomtrans::collect_contents(Ekm);

  // Done with PETSc data.
  ierr = MatDestroy(&Dbar_E);CHKERRXX(ierr);
  ierr = MatDestroy(&Dbar_B);CHKERRXX(ierr);
  ierr = KSPDestroy(&ksp);CHKERRXX(ierr);
  ierr = MatDestroy(&collision);CHKERRXX(ierr);
  ierr = VecDestroy(&Ekm);CHKERRXX(ierr);

  if (rank == 0) {
    // Write out the collected data.
    std::vector<anomtrans::kComps<k_dim>> all_k_comps;
    std::vector<unsigned int> all_ms;

    for (PetscInt ikm = 0; ikm < kmb.end_ikm; ikm++) {
      auto this_km = kmb.decompose(ikm);
      all_k_comps.push_back(std::get<0>(this_km));
      all_ms.push_back(std::get<1>(this_km));
    }

    json j_out = {
      {"k_comps", all_k_comps},
      {"ms", all_ms},
      {"Ekm", collected_Ekm},
      {"rho0", all_rho0},
      {"rhs_B0", all_rhs_B0},
      {"rho1_B0", all_rho1_B0},
      {"rhs_Bfinite", all_rhs_Bfinite},
      {"rho1_Bfinite", all_rho1_Bfinite}
    };

    std::stringstream outpath;
    outpath << "driving_test_out.json";

    std::ofstream fp_out(outpath.str());
    fp_out << j_out.dump();
    fp_out.close();

    /*
    // Check for changes from saved old result.
    boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
    if (not test_data_dir) {
      throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
    }

    std::stringstream known_data;
    known_data << *test_data_dir << "/driving_test_out.json";

    ASSERT_TRUE( anomtrans::check_json_equal(outpath.str(), known_data.str()) );
    */
  }
}
