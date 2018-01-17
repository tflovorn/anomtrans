#include <cstddef>
#include <cmath>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include <json.hpp>
#include "util/MPIPrettyUnitTestResultPrinter.h"
#include "util/util.h"
#include "util/constants.h"
#include "grid_basis.h"
#include "models/wsm_continuum_Hamiltonian.h"
#include "models/wsm_continuum_node_Hamiltonian.h"
#include "observables/energy.h"
#include "util/vec.h"
#include "util/mat.h"
#include "observables/rho0.h"
#include "hamiltonian.h"
#include "disorder/disorder.h"
#include "disorder/collision.h"
#include "driving.h"
#include "observables/current.h"
#include "berry.h"
#include "dm_graph.h"

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

/** @brief Check that the anomalous Hall conductivity of the Weyl semimetal
 *         has the expected value.
 */
TEST( WsmContinuumHamiltonian, wsm_continuum_ahe ) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  const std::size_t k_dim = 3;

  // Energy unit is Delta.
  double b = 2.0;
  // k unit is hbar v_F / Delta.
  double k0 = std::sqrt(std::pow(b, 2.0) - 1.0);

  // Parameters for result plots.
  /*
  std::array<unsigned int, k_dim> Nk = {16, 16, 16};
  anomtrans::kVals<k_dim> k_min = {-2.0 * k0, -2.0 * k0, -2.0 * k0};
  anomtrans::kVals<k_dim> k_max = {2.0 * k0, 2.0 * k0, 2.0 * k0};
  unsigned int num_mus = 10;
  double beta = 16.0;
  double sigma = 0.0147;
  */
  // Parameters for regression test.
  std::array<unsigned int, k_dim> Nk = {4, 4, 4};
  anomtrans::kVals<k_dim> k_min = {-2.0 * k0, -2.0 * k0, -2.0 * k0};
  anomtrans::kVals<k_dim> k_max = {2.0 * k0, 2.0 * k0, 2.0 * k0};
  unsigned int num_mus = 1;
  double beta = 4.0;
  double sigma = 0.2;

  unsigned int Nbands = 4;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands, k_min, k_max);

  anomtrans::WsmContinuumHamiltonian H(b, kmb);

  // Choose D = 2pi * \delta_{i, j}: Cartesian and reciprocal lattice
  // coordinates are equivalent. Appropriate for continuum model
  // with unitless momenta.
  double pi2 = 2.0 * anomtrans::pi;
  std::array<double, k_dim> a1 = {pi2, 0.0, 0.0};
  std::array<double, k_dim> a2 = {0.0, pi2, 0.0};
  std::array<double, k_dim> a3 = {0.0, 0.0, pi2};
  anomtrans::DimMatrix<k_dim> D = {a1, a2, a3};

  PetscReal max_energy_difference = anomtrans::find_max_energy_difference(kmb, H);
  double beta_max = anomtrans::get_beta_max(max_energy_difference);

  if (beta > beta_max) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: beta > beta_max: beta = %e ; beta_max = %e\n", beta, beta_max);
  }

  // U0 = how far can bands be driven from their average energy?
  // For the disorder form used, this quantity scales out of K: the distribution
  // of rho^(1) over k's has no dependence on it; is acts as an overall scale.
  // (TODO - sure this is correct?)
  double U0 = 1.0;

  double sigma_min = anomtrans::get_sigma_min(max_energy_difference);

  if (sigma < sigma_min) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: sigma < sigma_min: sigma = %e ; sigma_min = %e\n", sigma, sigma_min);
  }

  std::array<double, k_dim> Ehat = {0.0, 1.0, 0.0};

  Vec Ekm = anomtrans::get_energies(kmb, H);

  std::array<Mat, k_dim> v_op = anomtrans::calculate_velocity(kmb, H);

  PetscInt Ekm_min_index, Ekm_max_index;
  PetscReal Ekm_min, Ekm_max;
  PetscErrorCode ierr = VecMin(Ekm, &Ekm_min_index, &Ekm_min);CHKERRXX(ierr);
  ierr = VecMax(Ekm, &Ekm_max_index, &Ekm_max);CHKERRXX(ierr);

  std::size_t Nk_tot = anomtrans::get_Nk_total(Nk);
  double U0_sq = U0*U0;
  double disorder_coeff = U0_sq / Nk_tot;
  auto disorder_term = [Nbands, H, disorder_coeff](PetscInt ikm1, PetscInt ikm2)->double {
    return disorder_coeff*anomtrans::on_site_diagonal_disorder_band_preserved(Nbands, H,
        ikm1, ikm2);
  };
  auto disorder_term_od = [Nbands, H, disorder_coeff](PetscInt ikm1, PetscInt ikm2,
      PetscInt ikm3)->std::complex<double> {
    return disorder_coeff*anomtrans::on_site_diagonal_disorder(Nbands,
        H, ikm1, ikm2, ikm3);
  };

  Mat collision = anomtrans::make_collision(kmb, H, sigma, disorder_term);

  // Create the linear solver context.
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRXX(ierr);
  // This uses collision again as the preconditioning matrix.
  // TODO - is there a better choice?
  ierr = KSPSetOperators(ksp, collision, collision);CHKERRXX(ierr);
  // Could use KSPSetFromOptions here. In this case, prefer to keep options
  // hard-coded to have identical output from each test run.

  const unsigned int deriv_approx_order = 2;
  anomtrans::DerivStencil<1> stencil(anomtrans::DerivApproxType::central, deriv_approx_order);
  auto d_dk_Cart = anomtrans::make_d_dk_Cartesian(D, kmb, stencil);

  // Maximum number of elements expected for sum of Cartesian derivatives.
  PetscInt Ehat_grad_expected_elems_per_row = stencil.approx_order*k_dim*k_dim*k_dim;

  Mat Ehat_dot_grad_k = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ehat),
      d_dk_Cart, Ehat_grad_expected_elems_per_row);

  // TODO - what is a good way to choose broadening for Berry connection?
  double berry_broadening = 1e-4;
  auto R = anomtrans::make_berry_connection(kmb, H, berry_broadening);
  auto Ehat_dot_R = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ehat), R, kmb.Nbands);

  double mu_factor = 0.45;
  double mu_min = (1 - mu_factor) * Ekm_min + mu_factor * Ekm_max;
  double mu_max = mu_factor * Ekm_min + (1 - mu_factor) * Ekm_max;
  auto mus = anomtrans::linspace(mu_min, mu_max, num_mus);

  std::vector<std::vector<PetscReal>> all_rho0;
  std::vector<std::vector<PetscReal>> all_n_E;
  std::vector<PetscReal> all_sigma_yys;
  std::vector<PetscReal> all_sigma_xy_S_E_intrinsic;
  std::vector<PetscReal> all_sigma_xy_S_E_extrinsic;
  // For each mu, construct <n_E^(-1)> and <S_E^(0)>.
  for (auto mu : mus) {
    auto dm_rho0 = anomtrans::make_eq_node<anomtrans::StaticDMGraphNode>(Ekm, beta, mu);
    Vec rho0_km;
    ierr = VecDuplicate(Ekm, &rho0_km);CHKERRXX(ierr);
    ierr = MatGetDiagonal(dm_rho0->rho, rho0_km);CHKERRXX(ierr);

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

    anomtrans::add_linear_response_electric(dm_rho0, kmb, Ehat_dot_grad_k, Ehat_dot_R, ksp,
        H, sigma, disorder_term_od, berry_broadening);
    auto dm_n_E = dm_rho0->children[anomtrans::StaticDMDerivedBy::Kdd_inv_DE];
    Vec n_E;
    ierr = VecDuplicate(rho0_km, &n_E);CHKERRXX(ierr);
    ierr = MatGetDiagonal(dm_n_E->rho, n_E);CHKERRXX(ierr);

    // Have obtained linear response to electric field. Can calculate this
    // part of the longitudinal conductivity.
    // sigma_yy = -e Tr[v_y <rho_{E_y}>] / E_y
    PetscScalar sigma_yy = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_n_E->rho).at(0);
    all_sigma_yys.push_back(sigma_yy.real());

    auto collected_rho0 = anomtrans::split_scalars(anomtrans::collect_contents(rho0_km)).first;
    all_rho0.push_back(collected_rho0);
    auto collected_n_E = anomtrans::split_scalars(anomtrans::collect_contents(n_E)).first;
    all_n_E.push_back(collected_n_E);

    auto dm_S_E_intrinsic = dm_rho0->children[anomtrans::StaticDMDerivedBy::P_inv_DE];
    auto dm_S_E_extrinsic = dm_n_E->children[anomtrans::StaticDMDerivedBy::P_inv_Kod];

    PetscScalar sigma_xy_S_E_intrinsic = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op,
        dm_S_E_intrinsic->rho).at(0);
    all_sigma_xy_S_E_intrinsic.push_back(sigma_xy_S_E_intrinsic.real());

    PetscScalar sigma_xy_S_E_extrinsic = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op,
        dm_S_E_extrinsic->rho).at(0);
    all_sigma_xy_S_E_extrinsic.push_back(sigma_xy_S_E_extrinsic.real());

    ierr = VecDestroy(&n_E);CHKERRXX(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRXX(ierr);
    ierr = VecDestroy(&rho0_normalized);CHKERRXX(ierr);
    ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);
  }

  auto collected_Ekm = anomtrans::split_scalars(anomtrans::collect_contents(Ekm)).first;

  // Done with PETSc data.
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    ierr = MatDestroy(&(d_dk_Cart.at(dc)));CHKERRXX(ierr);
    ierr = MatDestroy(&(R.at(dc)));CHKERRXX(ierr);
    ierr = MatDestroy(&(v_op.at(dc)));CHKERRXX(ierr);
  }

  ierr = MatDestroy(&Ehat_dot_R);CHKERRXX(ierr);
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
      {"n_E", all_n_E},
      {"_series_sigma_yy", all_sigma_yys},
      {"_series_sigma_xy_S_E_intrinsic", all_sigma_xy_S_E_intrinsic},
      {"_series_sigma_xy_S_E_extrinsic", all_sigma_xy_S_E_extrinsic},
    };

    std::stringstream outpath;
    outpath << "wsm_continuum_ahe_test_out.json";

    std::ofstream fp_out(outpath.str());
    fp_out << j_out.dump();
    fp_out.close();

    // Check for changes from saved old result.
    boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
    if (not test_data_dir) {
      throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
    }

    std::stringstream known_path;
    known_path << *test_data_dir << "/wsm_continuum_ahe_test_out.json";

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
          j_known["k_comps"].get<std::vector<std::vector<unsigned int>>>(), -1.0, -1.0) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["ms"].get<std::vector<unsigned int>>(),
        j_known["ms"].get<std::vector<unsigned int>>(), -1.0, -1.0) );

    auto macheps = std::numeric_limits<PetscReal>::epsilon();
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["Ekm"].get<std::vector<PetscReal>>(),
        j_known["Ekm"].get<std::vector<PetscReal>>(),
        100.0*macheps, 10.0*macheps) );

    // TODO - what are appropriate scales for conductivities? Absolute error depends on disorder scale.
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_series_sigma_yy"].get<std::vector<PetscReal>>(),
        j_known["_series_sigma_yy"].get<std::vector<PetscReal>>(),
        100.0*macheps, 100.0*macheps) );

    // 1 is an appropriate scale for rho: elements range from 0 to 1.
    // TODO using 1 as scale for norm_d_rho0_dk also. Is this appropriate?
    // The k here is has scale 1 (k_recip values from 0 to 1).
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["rho0"].get<std::vector<std::vector<PetscReal>>>(),
        j_known["rho0"].get<std::vector<std::vector<PetscReal>>>(),
        100.0*macheps, 10.0*macheps) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["n_E"].get<std::vector<std::vector<PetscReal>>>(),
        j_known["n_E"].get<std::vector<std::vector<PetscReal>>>(),
        100.0*macheps, 10.0*macheps) );
  }
}

/** @brief Check that the chiral magnetic effect on a single Weyl node has the
 *         expected form for the Weyl semimetal continuum model.
 */
TEST( WsmContinuumHamiltonian, wsm_continuum_cme ) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  const std::size_t k_dim = 3;

  // Chirality of the Weyl node.
  int nu = 1;

  // Parameters for result plots.
  /*
  std::array<unsigned int, k_dim> Nk = {16, 16, 16};
  anomtrans::kVals<k_dim> k_min = {-2.0 * k0, -2.0 * k0, -2.0 * k0};
  anomtrans::kVals<k_dim> k_max = {2.0 * k0, 2.0 * k0, 2.0 * k0};
  unsigned int num_mus = 2;
  double beta = 16.0;
  */
  // Parameters for regression test.
  std::array<unsigned int, k_dim> Nk = {4, 4, 4};
  anomtrans::kVals<k_dim> k_min = {-0.5, -0.5, -0.5};
  anomtrans::kVals<k_dim> k_max = {0.5, 0.5, 0.5};
  unsigned int num_mus = 2;
  double beta = 4.0;

  unsigned int Nbands = 2;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands, k_min, k_max);

  anomtrans::WsmContinuumNodeHamiltonian H(nu, kmb);

  // Choose D = 2pi * \delta_{i, j}: Cartesian and reciprocal lattice
  // coordinates are equivalent. Appropriate for continuum model
  // with unitless momenta.
  double pi2 = 2.0 * anomtrans::pi;
  std::array<double, k_dim> a1 = {pi2, 0.0, 0.0};
  std::array<double, k_dim> a2 = {0.0, pi2, 0.0};
  std::array<double, k_dim> a3 = {0.0, 0.0, pi2};
  anomtrans::DimMatrix<k_dim> D = {a1, a2, a3};

  PetscReal max_energy_difference = anomtrans::find_max_energy_difference(kmb, H);
  double beta_max = anomtrans::get_beta_max(max_energy_difference);

  if (beta > beta_max) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: beta > beta_max: beta = %e ; beta_max = %e\n", beta, beta_max);
  }

  std::array<double, k_dim> Bhat = {0.0, 0.0, 1.0};

  Vec Ekm = anomtrans::get_energies(kmb, H);

  std::array<Mat, k_dim> v_op = anomtrans::calculate_velocity(kmb, H);

  PetscInt Ekm_min_index, Ekm_max_index;
  PetscReal Ekm_min, Ekm_max;
  PetscErrorCode ierr = VecMin(Ekm, &Ekm_min_index, &Ekm_min);CHKERRXX(ierr);
  ierr = VecMax(Ekm, &Ekm_max_index, &Ekm_max);CHKERRXX(ierr);

  const unsigned int deriv_approx_order = 2;
  anomtrans::DerivStencil<1> stencil(anomtrans::DerivApproxType::central, deriv_approx_order);
  auto d_dk_Cart = anomtrans::make_d_dk_Cartesian(D, kmb, stencil);

  auto DH0_cross_Bhat = anomtrans::make_DH0_cross_Bhat(kmb, H, Bhat);

  // TODO - what is a good way to choose broadening for Berry connection?
  double berry_broadening = 1e-4;
  auto R = anomtrans::make_berry_connection(kmb, H, berry_broadening);

  auto Omega = anomtrans::make_berry_curvature(kmb, H, berry_broadening);
  auto Bhat_dot_Omega = anomtrans::Vec_from_sum_const(anomtrans::make_complex_array(Bhat), Omega);

  double mu_factor = 0.45;
  double mu_min = (1 - mu_factor) * Ekm_min + mu_factor * Ekm_max;
  double mu_max = mu_factor * Ekm_min + (1 - mu_factor) * Ekm_max;
  auto mus = anomtrans::linspace(mu_min, mu_max, num_mus);

  std::vector<std::vector<PetscReal>> all_rho0;
  std::vector<std::vector<PetscReal>> all_xi_B;
  std::vector<PetscReal> all_current_S_B;
  std::vector<PetscReal> all_current_xi_B;
  for (auto mu : mus) {
    auto dm_rho0 = anomtrans::make_eq_node<anomtrans::StaticDMGraphNode>(Ekm, beta, mu);
    Vec rho0_km;
    ierr = VecDuplicate(Ekm, &rho0_km);CHKERRXX(ierr);
    ierr = MatGetDiagonal(dm_rho0->rho, rho0_km);CHKERRXX(ierr);

    anomtrans::add_linear_response_magnetic(dm_rho0, kmb, DH0_cross_Bhat, d_dk_Cart, R,
        Bhat_dot_Omega, H, berry_broadening);

    auto dm_S_B = dm_rho0->children[anomtrans::StaticDMDerivedBy::P_inv_DB];
    auto dm_xi_B = dm_rho0->children[anomtrans::StaticDMDerivedBy::B_dot_Omega];
    Vec xi_B;
    ierr = VecDuplicate(rho0_km, &xi_B);CHKERRXX(ierr);
    ierr = MatGetDiagonal(dm_xi_B->rho, xi_B);CHKERRXX(ierr);

    auto collected_rho0 = anomtrans::split_scalars(anomtrans::collect_contents(rho0_km)).first;
    all_rho0.push_back(collected_rho0);
    auto collected_xi_B = anomtrans::split_scalars(anomtrans::collect_contents(xi_B)).first;
    all_xi_B.push_back(collected_xi_B);

    // Have obtained linear response to magnetic field. Can calculate this
    // part of the longitudinal conductivity.
    PetscScalar current_S_B = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_S_B->rho).at(2);
    all_current_S_B.push_back(current_S_B.real());

    PetscScalar current_xi_B = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_xi_B->rho).at(2);
    all_current_xi_B.push_back(current_xi_B.real());

    ierr = VecDestroy(&xi_B);CHKERRXX(ierr);
    ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);
  }

  auto collected_Ekm = anomtrans::split_scalars(anomtrans::collect_contents(Ekm)).first;

  // Done with PETSc data.
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    ierr = MatDestroy(&(d_dk_Cart.at(dc)));CHKERRXX(ierr);
    ierr = MatDestroy(&(DH0_cross_Bhat.at(dc)));CHKERRXX(ierr);
    ierr = MatDestroy(&(R.at(dc)));CHKERRXX(ierr);
    ierr = MatDestroy(&(v_op.at(dc)));CHKERRXX(ierr);
    ierr = VecDestroy(&(Omega.at(dc)));CHKERRXX(ierr);
  }

  ierr = VecDestroy(&Bhat_dot_Omega);CHKERRXX(ierr);
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
      {"xi_B", all_xi_B},
      {"_series_current_S_B", all_current_S_B},
      {"_series_current_xi_B", all_current_xi_B},
    };

    std::stringstream outpath;
    outpath << "wsm_continuum_cme_test_out.json";

    std::ofstream fp_out(outpath.str());
    fp_out << j_out.dump();
    fp_out.close();

    // Check for changes from saved old result.
    boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
    if (not test_data_dir) {
      throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
    }

    std::stringstream known_path;
    known_path << *test_data_dir << "/wsm_continuum_cme_test_out.json";

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
          j_known["k_comps"].get<std::vector<std::vector<unsigned int>>>(), -1.0, -1.0) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["ms"].get<std::vector<unsigned int>>(),
        j_known["ms"].get<std::vector<unsigned int>>(), -1.0, -1.0) );

    auto macheps = std::numeric_limits<PetscReal>::epsilon();
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["Ekm"].get<std::vector<PetscReal>>(),
        j_known["Ekm"].get<std::vector<PetscReal>>(),
        100.0*macheps, 10.0*macheps) );

    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_series_current_S_B"].get<std::vector<PetscReal>>(),
        j_known["_series_current_S_B"].get<std::vector<PetscReal>>(),
        100.0*macheps, 100.0*macheps) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_series_current_xi_B"].get<std::vector<PetscReal>>(),
        j_known["_series_current_xi_B"].get<std::vector<PetscReal>>(),
        100.0*macheps, 100.0*macheps) );

    // 1 is an appropriate scale for rho: elements range from 0 to 1.
    // TODO using 1 as scale for norm_d_rho0_dk also. Is this appropriate?
    // The k here is has scale 1 (k_recip values from 0 to 1).
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["rho0"].get<std::vector<std::vector<PetscReal>>>(),
        j_known["rho0"].get<std::vector<std::vector<PetscReal>>>(),
        100.0*macheps, 10.0*macheps) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["xi_B"].get<std::vector<std::vector<PetscReal>>>(),
        j_known["xi_B"].get<std::vector<std::vector<PetscReal>>>(),
        100.0*macheps, 10.0*macheps) );
  }
}

/** @brief Check that the positive quadratic magnetoconductivity has the expected value.
 */
TEST( WsmContinuumHamiltonian, wsm_continuum_quadratic_magnetoconductivity ) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  const std::size_t k_dim = 3;

  // Energy unit is Delta.
  double b = 2.0;
  // k unit is hbar v_F / Delta.
  double k0 = std::sqrt(std::pow(b, 2.0) - 1.0);

  // Parameters for result plots.
  /*
  std::array<unsigned int, k_dim> Nk = {16, 16, 16};
  anomtrans::kVals<k_dim> k_min = {-2.0 * k0, -2.0 * k0, -2.0 * k0};
  anomtrans::kVals<k_dim> k_max = {2.0 * k0, 2.0 * k0, 2.0 * k0};
  unsigned int num_mus = 2;
  double beta = 16.0;
  double sigma = 0.0147;
  */
  // Parameters for regression test.
  std::array<unsigned int, k_dim> Nk = {4, 4, 4};
  anomtrans::kVals<k_dim> k_min = {-2.0 * k0, -2.0 * k0, -2.0 * k0};
  anomtrans::kVals<k_dim> k_max = {2.0 * k0, 2.0 * k0, 2.0 * k0};
  unsigned int num_mus = 1;
  double beta = 4.0;
  double sigma = 0.2;

  unsigned int Nbands = 4;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands, k_min, k_max);

  anomtrans::WsmContinuumHamiltonian H(b, kmb);

  // Choose D = 2pi * \delta_{i, j}: Cartesian and reciprocal lattice
  // coordinates are equivalent. Appropriate for continuum model
  // with unitless momenta.
  double pi2 = 2.0 * anomtrans::pi;
  std::array<double, k_dim> a1 = {pi2, 0.0, 0.0};
  std::array<double, k_dim> a2 = {0.0, pi2, 0.0};
  std::array<double, k_dim> a3 = {0.0, 0.0, pi2};
  anomtrans::DimMatrix<k_dim> D = {a1, a2, a3};

  PetscReal max_energy_difference = anomtrans::find_max_energy_difference(kmb, H);
  double beta_max = anomtrans::get_beta_max(max_energy_difference);

  if (beta > beta_max) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: beta > beta_max: beta = %e ; beta_max = %e\n", beta, beta_max);
  }

  double sigma_min = anomtrans::get_sigma_min(max_energy_difference);

  if (sigma < sigma_min) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: sigma < sigma_min: sigma = %e ; sigma_min = %e\n", sigma, sigma_min);
  }

  std::array<double, k_dim> Ehat = {0.0, 0.0, 1.0};
  std::array<double, k_dim> Bhat = {0.0, 0.0, 1.0};

  Vec Ekm = anomtrans::get_energies(kmb, H);

  std::array<Mat, k_dim> v_op = anomtrans::calculate_velocity(kmb, H);

  // U0 = how far can bands be driven from their average energy?
  // For the disorder form used, this quantity scales out of K: the distribution
  // of rho^(1) over k's has no dependence on it; is acts as an overall scale.
  // (TODO - sure this is correct?)
  double U0 = 1.0;

  std::size_t Nk_tot = anomtrans::get_Nk_total(Nk);
  double U0_sq = U0*U0;
  double disorder_coeff = U0_sq / Nk_tot;
  auto disorder_term = [Nbands, H, disorder_coeff](PetscInt ikm1, PetscInt ikm2)->double {
    return disorder_coeff*anomtrans::on_site_diagonal_disorder_band_preserved(Nbands, H,
        ikm1, ikm2);
  };
  auto disorder_term_od = [Nbands, H, disorder_coeff](PetscInt ikm1, PetscInt ikm2,
      PetscInt ikm3)->std::complex<double> {
    return disorder_coeff*anomtrans::on_site_diagonal_disorder(Nbands,
        H, ikm1, ikm2, ikm3);
  };

  Mat collision = anomtrans::make_collision(kmb, H, sigma, disorder_term);

  // Create the linear solver context.
  KSP ksp;
  PetscErrorCode ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRXX(ierr);
  // This uses collision again as the preconditioning matrix.
  // TODO - is there a better choice?
  ierr = KSPSetOperators(ksp, collision, collision);CHKERRXX(ierr);
  // Could use KSPSetFromOptions here. In this case, prefer to keep options
  // hard-coded to have identical output from each test run.

  const unsigned int deriv_approx_order = 2;
  anomtrans::DerivStencil<1> stencil(anomtrans::DerivApproxType::central, deriv_approx_order);
  auto d_dk_Cart = anomtrans::make_d_dk_Cartesian(D, kmb, stencil);

  // Maximum number of elements expected for sum of Cartesian derivatives.
  PetscInt Ehat_grad_expected_elems_per_row = stencil.approx_order*k_dim*k_dim*k_dim;

  Mat Ehat_dot_grad_k = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ehat),
      d_dk_Cart, Ehat_grad_expected_elems_per_row);
  auto DH0_cross_Bhat = anomtrans::make_DH0_cross_Bhat(kmb, H, Bhat);

  // TODO - what is a good way to choose broadening for Berry connection?
  double berry_broadening = 1e-4;

  auto R = anomtrans::make_berry_connection(kmb, H, berry_broadening);
  auto Ehat_dot_R = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ehat), R, kmb.Nbands);

  auto Omega = anomtrans::make_berry_curvature(kmb, H, berry_broadening);
  auto Bhat_dot_Omega = anomtrans::Vec_from_sum_const(anomtrans::make_complex_array(Bhat), Omega);

  PetscInt Ekm_min_index, Ekm_max_index;
  PetscReal Ekm_min, Ekm_max;
  ierr = VecMin(Ekm, &Ekm_min_index, &Ekm_min);CHKERRXX(ierr);
  ierr = VecMax(Ekm, &Ekm_max_index, &Ekm_max);CHKERRXX(ierr);

  double mu_factor = 0.45;
  double mu_min = (1 - mu_factor) * Ekm_min + mu_factor * Ekm_max;
  double mu_max = mu_factor * Ekm_min + (1 - mu_factor) * Ekm_max;
  auto mus = anomtrans::linspace(mu_min, mu_max, num_mus);

  std::vector<PetscReal> all_sigma_xi_to_xi, all_sigma_xi_to_S_int, all_sigma_xi_to_S_ext;
  std::vector<PetscReal> all_sigma_S_int_to_xi, all_sigma_S_int_to_S_int, all_sigma_S_int_to_S_ext;
  std::vector<PetscReal> all_sigma_S_ext_to_xi, all_sigma_S_ext_to_S_int, all_sigma_S_ext_to_S_ext;
  for (auto mu : mus) {
    auto dm_rho0 = anomtrans::make_eq_node<anomtrans::StaticDMGraphNode>(Ekm, beta, mu);
    Vec rho0_km;
    ierr = VecDuplicate(Ekm, &rho0_km);CHKERRXX(ierr);
    ierr = MatGetDiagonal(dm_rho0->rho, rho0_km);CHKERRXX(ierr);

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

    // <xi_B> branch: <rho_0> -> <xi_B> -> <n_{EB}> -> {<S_{EB^2}>, <xi_{EB^2}>}
    anomtrans::add_linear_response_magnetic(dm_rho0, kmb, DH0_cross_Bhat, d_dk_Cart, R,
        Bhat_dot_Omega, H, berry_broadening);
    auto dm_xi_B = dm_rho0->children[anomtrans::StaticDMDerivedBy::B_dot_Omega];

    anomtrans::add_linear_response_electric(dm_xi_B, kmb, Ehat_dot_grad_k, Ehat_dot_R, ksp,
        H, sigma, disorder_term_od, berry_broadening);
    auto dm_xi_B_n_EB = dm_xi_B->children[anomtrans::StaticDMDerivedBy::Kdd_inv_DE];

    anomtrans::add_next_order_magnetic(dm_xi_B_n_EB, kmb, DH0_cross_Bhat, d_dk_Cart, R, ksp,
        Bhat_dot_Omega, H, sigma, disorder_term_od, berry_broadening);
    auto dm_xi_to_xi = dm_xi_B_n_EB->children[anomtrans::StaticDMDerivedBy::B_dot_Omega];
    auto dm_xi_to_S_int = dm_xi_B_n_EB->children[anomtrans::StaticDMDerivedBy::P_inv_DB];
    auto dm_xi_to_S_ext = dm_xi_B_n_EB->children[anomtrans::StaticDMDerivedBy::Kdd_inv_DB]
        ->children[anomtrans::StaticDMDerivedBy::P_inv_Kod];

    PetscScalar sigma_xi_to_xi = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_xi_to_xi->rho).at(2);
    PetscScalar sigma_xi_to_S_int = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_xi_to_S_int->rho).at(2);
    PetscScalar sigma_xi_to_S_ext = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_xi_to_S_ext->rho).at(2);

    all_sigma_xi_to_xi.push_back(sigma_xi_to_xi.real());
    all_sigma_xi_to_S_int.push_back(sigma_xi_to_S_int.real());
    all_sigma_xi_to_S_ext.push_back(sigma_xi_to_S_ext.real());

    // <S_E> branches
    anomtrans::add_linear_response_electric(dm_rho0, kmb, Ehat_dot_grad_k, Ehat_dot_R, ksp,
        H, sigma, disorder_term_od, berry_broadening);
    auto dm_n_E = dm_rho0->children[anomtrans::StaticDMDerivedBy::Kdd_inv_DE];

    // <S_E> intrinsic branch:
    auto dm_S_E_int = dm_rho0->children[anomtrans::StaticDMDerivedBy::P_inv_DE];

    anomtrans::add_next_order_magnetic(dm_S_E_int, kmb, DH0_cross_Bhat, d_dk_Cart, R, ksp,
        Bhat_dot_Omega, H, sigma, disorder_term_od, berry_broadening);
    auto dm_S_E_int_n_EB = dm_S_E_int->children[anomtrans::StaticDMDerivedBy::Kdd_inv_DB];

    anomtrans::add_next_order_magnetic(dm_S_E_int_n_EB, kmb, DH0_cross_Bhat, d_dk_Cart, R, ksp,
        Bhat_dot_Omega, H, sigma, disorder_term_od, berry_broadening);
    auto dm_S_int_to_xi = dm_S_E_int_n_EB->children[anomtrans::StaticDMDerivedBy::B_dot_Omega];
    auto dm_S_int_to_S_int = dm_S_E_int_n_EB->children[anomtrans::StaticDMDerivedBy::P_inv_DB];
    auto dm_S_int_to_S_ext = dm_S_E_int_n_EB->children[anomtrans::StaticDMDerivedBy::Kdd_inv_DB]
        ->children[anomtrans::StaticDMDerivedBy::P_inv_Kod];

    PetscScalar sigma_S_int_to_xi = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_S_int_to_xi->rho).at(2);
    PetscScalar sigma_S_int_to_S_int = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_S_int_to_S_int->rho).at(2);
    PetscScalar sigma_S_int_to_S_ext = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_S_int_to_S_ext->rho).at(2);

    all_sigma_S_int_to_xi.push_back(sigma_S_int_to_xi.real());
    all_sigma_S_int_to_S_int.push_back(sigma_S_int_to_S_int.real());
    all_sigma_S_int_to_S_ext.push_back(sigma_S_int_to_S_ext.real());

    // <S_E> extrinsic branch:
    auto dm_S_E_ext = dm_n_E->children[anomtrans::StaticDMDerivedBy::P_inv_Kod];

    anomtrans::add_next_order_magnetic(dm_S_E_ext, kmb, DH0_cross_Bhat, d_dk_Cart, R, ksp,
        Bhat_dot_Omega, H, sigma, disorder_term_od, berry_broadening);
    auto dm_S_E_ext_n_EB = dm_S_E_ext->children[anomtrans::StaticDMDerivedBy::Kdd_inv_DB];

    anomtrans::add_next_order_magnetic(dm_S_E_ext_n_EB, kmb, DH0_cross_Bhat, d_dk_Cart, R, ksp,
        Bhat_dot_Omega, H, sigma, disorder_term_od, berry_broadening);
    auto dm_S_ext_to_xi = dm_S_E_ext_n_EB->children[anomtrans::StaticDMDerivedBy::B_dot_Omega];
    auto dm_S_ext_to_S_int = dm_S_E_ext_n_EB->children[anomtrans::StaticDMDerivedBy::P_inv_DB];
    auto dm_S_ext_to_S_ext = dm_S_E_ext_n_EB->children[anomtrans::StaticDMDerivedBy::Kdd_inv_DB]
        ->children[anomtrans::StaticDMDerivedBy::P_inv_Kod];

    PetscScalar sigma_S_ext_to_xi = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_S_ext_to_xi->rho).at(2);
    PetscScalar sigma_S_ext_to_S_int = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_S_ext_to_S_int->rho).at(2);
    PetscScalar sigma_S_ext_to_S_ext = kmb.k_volume() * anomtrans::calculate_current_ev(kmb, v_op, dm_S_ext_to_S_ext->rho).at(2);

    all_sigma_S_ext_to_xi.push_back(sigma_S_ext_to_xi.real());
    all_sigma_S_ext_to_S_int.push_back(sigma_S_ext_to_S_int.real());
    all_sigma_S_ext_to_S_ext.push_back(sigma_S_ext_to_S_ext.real());

    ierr = MatNullSpaceDestroy(&nullspace);CHKERRXX(ierr);
    ierr = VecDestroy(&rho0_normalized);CHKERRXX(ierr);
    ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);
  }

  auto collected_Ekm = anomtrans::split_scalars(anomtrans::collect_contents(Ekm)).first;

  // Done with PETSc data.
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    ierr = MatDestroy(&(d_dk_Cart.at(dc)));CHKERRXX(ierr);
    ierr = MatDestroy(&(R.at(dc)));CHKERRXX(ierr);
    ierr = MatDestroy(&(v_op.at(dc)));CHKERRXX(ierr);
  }

  ierr = MatDestroy(&Ehat_dot_R);CHKERRXX(ierr);
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
      {"_series_sigma_xi_to_xi", all_sigma_xi_to_xi},
      {"_series_sigma_xi_to_S_int", all_sigma_xi_to_S_int},
      {"_series_sigma_xi_to_S_ext", all_sigma_xi_to_S_ext},
      {"_series_sigma_S_int_to_xi", all_sigma_S_int_to_xi},
      {"_series_sigma_S_int_to_S_int", all_sigma_S_int_to_S_int},
      {"_series_sigma_S_int_to_S_ext", all_sigma_S_int_to_S_ext},
      {"_series_sigma_S_ext_to_xi", all_sigma_S_ext_to_xi},
      {"_series_sigma_S_ext_to_S_int", all_sigma_S_ext_to_S_int},
      {"_series_sigma_S_ext_to_S_ext", all_sigma_S_ext_to_S_ext},
    };

    std::stringstream outpath;
    outpath << "wsm_continuum_quadratic_magnetoconductivity_test_out.json";

    std::ofstream fp_out(outpath.str());
    fp_out << j_out.dump();
    fp_out.close();

    // Check for changes from saved old result.
    boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
    if (not test_data_dir) {
      throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
    }

    std::stringstream known_path;
    known_path << *test_data_dir << "/wsm_continuum_quadratic_magnetoconductivity_test_out.json";

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
          j_known["k_comps"].get<std::vector<std::vector<unsigned int>>>(), -1.0, -1.0) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["ms"].get<std::vector<unsigned int>>(),
        j_known["ms"].get<std::vector<unsigned int>>(), -1.0, -1.0) );

    auto macheps = std::numeric_limits<PetscReal>::epsilon();
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["Ekm"].get<std::vector<PetscReal>>(),
        j_known["Ekm"].get<std::vector<PetscReal>>(),
        100.0*macheps, 10.0*macheps) );

    // TODO - what are appropriate scales for conductivities? Absolute error depends on disorder scale.
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_series_sigma_xi_to_xi"].get<std::vector<PetscReal>>(),
        j_known["_series_sigma_xi_to_xi"].get<std::vector<PetscReal>>(),
        100.0*macheps, 100.0*macheps) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_series_sigma_xi_to_S_int"].get<std::vector<PetscReal>>(),
        j_known["_series_sigma_xi_to_S_int"].get<std::vector<PetscReal>>(),
        100.0*macheps, 100.0*macheps) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_series_sigma_S_int_to_xi"].get<std::vector<PetscReal>>(),
        j_known["_series_sigma_S_int_to_xi"].get<std::vector<PetscReal>>(),
        100.0*macheps, 100.0*macheps) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_series_sigma_S_int_to_S_int"].get<std::vector<PetscReal>>(),
        j_known["_series_sigma_S_int_to_S_int"].get<std::vector<PetscReal>>(),
        100.0*macheps, 100.0*macheps) );
  }
}
