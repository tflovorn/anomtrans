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
#include "disorder.h"
#include "collision.h"
#include "driving.h"
#include "conductivity.h"

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

  double t = 1.0;
  double tp = -0.3;

  // Parameters for result plots.
  /*
  std::array<unsigned int, k_dim> Nk = {256, 256};
  unsigned int num_mus = 40;
  double beta = 10.0/t;
  double sigma = 0.01*t;
  */
  // Parameters for regression test.
  std::array<unsigned int, k_dim> Nk = {8, 8};
  unsigned int num_mus = 5;
  double beta = 0.2/t;
  double sigma = 0.5*t;

  unsigned int Nbands = 1;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands);

  anomtrans::square_tb_Hamiltonian H(t, tp, Nk);

  std::array<double, k_dim> a1 = {1.0, 0.0};
  std::array<double, k_dim> a2 = {0.0, 1.0};
  anomtrans::DimMatrix<k_dim> D = {a1, a2};

  PetscReal max_energy_difference = anomtrans::find_max_energy_difference(kmb, H);
  double beta_max = anomtrans::get_beta_max(max_energy_difference);

  if (beta > beta_max) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: beta > beta_max: beta = %e ; beta_max = %e\n", beta, beta_max);
  }

  // U0 = how far can bands be driven from their average energy?
  // For the disorder form used, this quantity scales out of K: the distribution
  // of rho^(1) over k's has no dependence on it; is acts as an overall scale.
  // (TODO - sure this is correct?)
  double U0 = 0.1*t;

  double sigma_min = anomtrans::get_sigma_min(max_energy_difference);

  if (sigma < sigma_min) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: sigma < sigma_min: sigma = %e ; sigma_min = %e\n", sigma, sigma_min);
  }

  std::array<double, k_dim> Ehat = {0.0, 1.0};
  std::array<double, 3> Bhat = {0.0, 0.0, 1.0};

  Vec Ekm = anomtrans::get_energies(kmb, H);

  PetscInt Ekm_min_index, Ekm_max_index;
  PetscReal Ekm_min, Ekm_max;
  PetscErrorCode ierr = VecMin(Ekm, &Ekm_min_index, &Ekm_min);CHKERRXX(ierr);
  ierr = VecMax(Ekm, &Ekm_max_index, &Ekm_max);CHKERRXX(ierr);

  PetscInt Nk_tot = 1;
  for (std::size_t d = 0; d < k_dim; d++) {
    Nk_tot *= kmb.Nk.at(d);
  }
  double U0_sq = U0*U0;
  double disorder_coeff = U0_sq / Nk_tot;
  /*
  auto disorder_term = [Nbands, H, disorder_coeff](PetscInt ikm1, PetscInt ikm2)->double {
    return disorder_coeff*anomtrans::on_site_diagonal_disorder(Nbands, H, ikm1, ikm2);
  };
  */
  double Lambda = 1e-12;
  anomtrans::SpatialDisorderCorrelation<k_dim> ULambda(kmb, D, Lambda);
  auto disorder_term = [Nbands, H, ULambda, disorder_coeff](PetscInt ikm1, PetscInt ikm2)->double {
    return disorder_coeff*anomtrans::spatially_correlated_diagonal_disorder(Nbands, H, ULambda, ikm1, ikm2);
  };

  // TODO include finite disorder correlation length
  Mat collision = anomtrans::make_collision(kmb, H, sigma, disorder_term);

  // The collision matrix K should be symmetric. Knowing that this is the case
  // substantially simplifies solution of the associated linear systems.
  // TODO assuming that PetscScalar is not complex here. If it is complex, we
  // should check that K is Hermitian.
  // TODO do this check manually? Trying this crashes with an error that MatIsSymmetric
  // is not supported for MATMPIAIJ.
  //PetscBool K_is_symmetric;
  //PetscReal tol = 1e-12;
  //ierr = MatIsSymmetric(collision, tol, &K_is_symmetric);CHKERRXX(ierr);
  //ASSERT_TRUE( K_is_symmetric );

  //ierr = MatSetOption(collision, MAT_SYMMETRIC, PETSC_TRUE);CHKERRXX(ierr);

  // Create the linear solver context.
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRXX(ierr);
  // This uses collision again as the preconditioning matrix.
  // TODO - is there a better choice?
  ierr = KSPSetOperators(ksp, collision, collision);CHKERRXX(ierr);
  // Could use KSPSetFromOptions here. In this case, prefer to keep options
  // hard-coded to have identical output from each test run.

  const unsigned int deriv_approx_order = 2;
  Mat Dbar_E = anomtrans::driving_electric(D, kmb, deriv_approx_order, Ehat);
  Mat Dbar_B = anomtrans::driving_magnetic(D, kmb, deriv_approx_order, H, Bhat);


  auto mus = anomtrans::linspace(Ekm_min, Ekm_max, num_mus);

  std::vector<std::vector<PetscScalar>> all_rho0;
  std::vector<std::vector<PetscScalar>> all_rhs_B0;
  std::vector<std::vector<PetscScalar>> all_rho1_B0;
  std::vector<std::vector<PetscScalar>> all_rhs_Bfinite;
  std::vector<std::vector<PetscScalar>> all_rho1_Bfinite;
  std::vector<PetscScalar> all_Hall_conductivities;
  std::vector<std::vector<PetscScalar>> all_Hall_conductivity_components;
  // For each mu, solve the pair of equations:
  // K rho1_B0 = Dbar_E rho0
  // K rho1_Bfinite = -Dbar_B rho1_B0
  for (auto mu : mus) {
    Vec rho0_km = anomtrans::make_rho0(Ekm, beta, mu);

    // Get normalized version of rho0 to use for nullspace.
    // TODO can we safely pass a nullptr instead of rho0_orig_norm?
    Vec rho0_normalized;
    PetscReal rho0_orig_norm;
    ierr = VecDuplicate(rho0_km, &rho0_normalized);CHKERRXX(ierr);
    ierr = VecCopy(rho0_km, rho0_normalized);CHKERRXX(ierr);
    ierr = VecNormalize(rho0_normalized, &rho0_orig_norm);CHKERRXX(ierr);

    // Set nullspace of K: K rho0_km = 0.
    // Note that this is true regardless of the value of mu
    // (energies only enter K through differences).
    // It is also true for any value of beta (the Fermi-Dirac distribution
    // function does not appear in K, only energy differences).
    // TODO does this mean that the nullspace has dimension larger than 1?
    MatNullSpace nullspace;
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &rho0_normalized, &nullspace);CHKERRXX(ierr);
    ierr = MatSetNullSpace(collision, nullspace);CHKERRXX(ierr);
    // NOTE rho0_normalized must not be modified after this call until we are done with nullspace.

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
    ierr = VecScale(rhs_Bfinite, -1.0);CHKERRXX(ierr);

    Vec rho1_Bfinite;
    ierr = VecDuplicate(rho0_km, &rho1_Bfinite);CHKERRXX(ierr);
    ierr = KSPSolve(ksp, rhs_Bfinite, rho1_Bfinite);CHKERRXX(ierr);

    PetscScalar sigma_Hall;
    Vec sigma_Hall_components;
    std::tie(sigma_Hall, sigma_Hall_components) = calculate_Hall_conductivity(kmb, H, rho1_Bfinite);

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

    all_Hall_conductivities.push_back(sigma_Hall);
    auto collected_sigma_Hall_components = anomtrans::collect_contents(sigma_Hall_components);
    all_Hall_conductivity_components.push_back(collected_sigma_Hall_components);

    ierr = VecDestroy(&sigma_Hall_components);CHKERRXX(ierr);
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
      {"mus", mus},
      {"k_comps", all_k_comps},
      {"ms", all_ms},
      {"Ekm", collected_Ekm},
      {"rho0", all_rho0},
      {"rhs_B0", all_rhs_B0},
      {"rho1_B0", all_rho1_B0},
      {"rhs_Bfinite", all_rhs_Bfinite},
      {"rho1_Bfinite", all_rho1_Bfinite},
      {"_series_Hall_conductivity", all_Hall_conductivities},
      {"Hall_conductivity_components", all_Hall_conductivity_components}
    };

    std::stringstream outpath;
    outpath << "driving_test_out.json";

    std::ofstream fp_out(outpath.str());
    fp_out << j_out.dump();
    fp_out.close();

    // Check for changes from saved old result.
    boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
    if (not test_data_dir) {
      throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
    }

    std::stringstream known_path;
    known_path << *test_data_dir << "/driving_test_out.json";

    json j_known;
    std::ifstream fp_k(known_path.str());
    if (not fp_k.good()) {
      throw std::runtime_error("could not open file in check_json_equal");
    }
    fp_k >> j_known;
    fp_k.close();

    // TODO clean these checks up: could replace these long calls with calls to
    // a function template that takes a type, two jsons, a key, and a tol:
    // ASSERT_TRUE( anomtrans::check_json_elem_equal<T>(j_out, j_known, key, tol) );
    // This function would call check_equal_within in the same way as below.
    //
    // k_comps and ms are integers and should be exactly equal.
    // NOTE - nlohmann::json doesn't implement std::arrays. Use a std::vector
    // here: it has the same JSON representation as the array.
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["k_comps"].get<std::vector<std::vector<unsigned int>>>(),
          j_known["k_comps"].get<std::vector<std::vector<unsigned int>>>(), -1.0) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["ms"].get<std::vector<unsigned int>>(),
        j_known["ms"].get<std::vector<unsigned int>>(), -1.0) );

    // t is an appropriate scale for E.
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["Ekm"].get<std::vector<PetscScalar>>(),
        j_known["Ekm"].get<std::vector<PetscScalar>>(),
        100.0*t*std::numeric_limits<PetscScalar>::epsilon()) );

    // 1 is an appropriate scale for rho: elements range from 0 to 1.
    // TODO using 1 as scale for norm_d_rho0_dk also. Is this appropriate?
    // The k here is has scale 1 (k_recip values from 0 to 1).
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["rho0"].get<std::vector<std::vector<PetscScalar>>>(),
        j_known["rho0"].get<std::vector<std::vector<PetscScalar>>>(),
        100.0*std::numeric_limits<PetscScalar>::epsilon()) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["rhs_B0"].get<std::vector<std::vector<PetscScalar>>>(),
        j_known["rhs_B0"].get<std::vector<std::vector<PetscScalar>>>(),
        100.0*std::numeric_limits<PetscScalar>::epsilon()) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["rho1_B0"].get<std::vector<std::vector<PetscScalar>>>(),
        j_known["rho1_B0"].get<std::vector<std::vector<PetscScalar>>>(),
        100.0*std::numeric_limits<PetscScalar>::epsilon()) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["rhs_Bfinite"].get<std::vector<std::vector<PetscScalar>>>(),
        j_known["rhs_Bfinite"].get<std::vector<std::vector<PetscScalar>>>(),
        100.0*std::numeric_limits<PetscScalar>::epsilon()) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["rho1_Bfinite"].get<std::vector<std::vector<PetscScalar>>>(),
        j_known["rho1_Bfinite"].get<std::vector<std::vector<PetscScalar>>>(),
        1000.0*std::numeric_limits<PetscScalar>::epsilon()) );
  }
}
