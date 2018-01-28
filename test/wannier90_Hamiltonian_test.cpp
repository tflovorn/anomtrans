#include <cstddef>
#include <limits>
#include <complex>
#include <tuple>
#include <exception>
#include <fstream>
#include <sstream>
#include <map>
#include <boost/optional.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include <json.hpp>
#include "util/MPIPrettyUnitTestResultPrinter.h"
#include "util/util.h"
#include "util/lattice.h"
#include "grid_basis.h"
#include "dyn_dm_graph.h"
#include "berry.h"
#include "driving.h"
#include "models/wannier90_Hamiltonian.h"
#include "observables/energy.h"
#include "observables/rho0.h"
#include "observables/current.h"
#include "disorder/disorder.h"
#include "disorder/collision.h"

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

/** @brief Check that extraction of Wannier90 hr.dat file is performed correctly.
 */
TEST( Wannier90_hr_load, Wannier90_hr_load ) {
  boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
  if (not test_data_dir) {
    throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
  }

  std::stringstream hr_path;
  hr_path << *test_data_dir << "/WSe2/wannier/WSe2_hr.dat";

  const std::size_t k_dim = 2;
  anomtrans::kComps<k_dim> Nk = {8, 8};
  anomtrans::DimMatrix<k_dim> D = {{{1.659521, 1.659521},
                                   {-2.874374, 2.874374}}};

  auto tb = anomtrans::extract_Wannier90_Hamiltonian(hr_path.str(), Nk, D);

  // Check that Nbands and Nr are correct.
  std::size_t Nbands_expected = 22;
  std::size_t Nr_expected = 91;

  ASSERT_EQ( tb.kmb.Nbands, Nbands_expected );
  ASSERT_EQ( tb.Hrs.size(), Nr_expected );

  // Check some values to ensure they loaded correctly.
  std::map<anomtrans::LatVec<k_dim>, unsigned int> expected_degens {
    {{-6, -3}, 3}, {{-5, -4}, 2}
  };

  using HrElem = std::tuple<anomtrans::LatVec<k_dim>, std::size_t, std::size_t>;

  std::map<HrElem, std::complex<double>> expected_values {
    {HrElem{{-6, -3}, 0, 0}, std::complex<double>(0.001540, 0.0)},
    {HrElem{{-6, -3}, 5, 0}, std::complex<double>(0.0, -0.000101)},
    {HrElem{{-5, -4}, 0, 0}, std::complex<double>(0.000448, 0.0)}
  };

  auto macheps = std::numeric_limits<double>::epsilon();
  for (auto it = expected_values.begin(); it != expected_values.end(); ++it) {
    anomtrans::LatVec<k_dim> r;
    std::size_t ip, i;
    std::tie(r, ip, i) = it->first;
    unsigned int degen = expected_degens[r];
    std::complex<double> expected_val = it->second / static_cast<double>(degen);

    std::complex<double> elem = tb.Hrs.at(r)(ip, i);

    ASSERT_TRUE( anomtrans::scalars_approx_equal(elem, expected_val, 10.0*macheps, 10.0*macheps) );
  }

  // Check that H(R) = H(-R)^{\dagger}, as required for Hermiticity.
  for (auto it = tb.Hrs.begin(); it != tb.Hrs.end(); ++it) {
    anomtrans::LatVec<k_dim> r = it->first;
    anomtrans::LatVec<k_dim> minus_r = {-r.at(0), -r.at(1)};

    ASSERT_TRUE( it->second.isApprox(tb.Hrs.at(minus_r).adjoint(), 10.0*macheps) );
  }
}

/** @brief Dynamic electric response to second order with WSe2 model from Wannier90 hr.dat.
 */
TEST( Wannier90_WSe2_dynamic, DISABLED_Wannier90_WSe2_dynamic ) {
  boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
  if (not test_data_dir) {
    throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
  }

  std::stringstream hr_path;
  hr_path << *test_data_dir << "/WSe2/wannier/WSe2_hr.dat";

  const std::size_t k_dim = 2;
  anomtrans::kComps<k_dim> Nk = {4, 4};
  anomtrans::DimMatrix<k_dim> D = {{{1.659521, 1.659521}, // Angstrom
                                   {-2.874374, 2.874374}}};

  auto H = anomtrans::extract_Wannier90_Hamiltonian(hr_path.str(), Nk, D);
  const auto& kmb = H.kmb;
  const auto Nbands = H.kmb.Nbands;

  // Choose berry_broadening ~ optical broadening.
  // What is appropriate value?
  double berry_broadening = 1e-3; // eV

  PetscReal max_energy_difference = anomtrans::find_max_energy_difference(kmb, H);
  double beta_max = anomtrans::get_beta_max(max_energy_difference);
  double beta = beta_max / 2.0;

  if (beta > beta_max) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: beta > beta_max: beta = %e ; beta_max = %e\n", beta, beta_max);
  }

  double sigma_min = anomtrans::get_sigma_min(max_energy_difference);
  double sigma = 2.0 * sigma_min;

  if (sigma < sigma_min) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: sigma < sigma_min: sigma = %e ; sigma_min = %e\n", sigma, sigma_min);
  }

  // U0 = how far can bands be driven from their average energy?
  double U0 = 1e-3; // eV

  auto Ekm = anomtrans::get_energies(kmb, H);

  auto v_op = anomtrans::calculate_velocity(kmb, H);
  auto spin_op = anomtrans::calculate_spin_operator(kmb, H);

  std::size_t Nk_tot = anomtrans::get_Nk_total(Nk);
  for (std::size_t d = 0; d < k_dim; d++) {
    Nk_tot *= kmb.Nk.at(d);
  }
  double U0_sq = U0*U0;
  double disorder_coeff = U0_sq / Nk_tot;

  auto disorder_term = [Nbands, H, disorder_coeff](PetscInt ikm1, PetscInt ikm2)->double {
    return disorder_coeff*anomtrans::on_site_diagonal_disorder_band_preserved(Nbands, H,
        ikm1, ikm2);
  };
  auto disorder_term_od = [Nbands, H, disorder_coeff](PetscInt ikm1, PetscInt ikm2,
      PetscInt ikm3)->std::complex<double> {
    return disorder_coeff*anomtrans::on_site_diagonal_disorder(Nbands, H, ikm1, ikm2, ikm3);
  };

  auto collision = anomtrans::make_collision(kmb, H, sigma, disorder_term);

  // Create the linear solver context.
  KSP ksp;
  PetscErrorCode ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRXX(ierr);
  // This uses collision again as the preconditioning matrix.
  // TODO - is there a better choice?
  ierr = KSPSetOperators(ksp, collision.M, collision.M);CHKERRXX(ierr);
  // Could use KSPSetFromOptions here. In this case, prefer to keep options
  // hard-coded to have identical output from each test run.

  const unsigned int deriv_approx_order = 2;
  anomtrans::DerivStencil<1> stencil(anomtrans::DerivApproxType::central, deriv_approx_order);
  auto d_dk_Cart = anomtrans::make_d_dk_Cartesian(D, kmb, stencil);

  // E in x and y directions
  std::array<double, k_dim> Ex = {1.0, 0.0};
  std::array<double, k_dim> Ey = {0.0, 1.0};

  // Maximum number of elements expected for sum of Cartesian derivatives.
  PetscInt Ehat_grad_expected_elems_per_row = stencil.approx_order*k_dim*k_dim*k_dim;

  auto Ex_dot_grad_k = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ex),
      anomtrans::unowned(d_dk_Cart), Ehat_grad_expected_elems_per_row);
  auto Ey_dot_grad_k = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ey),
      anomtrans::unowned(d_dk_Cart), Ehat_grad_expected_elems_per_row);

  auto R = anomtrans::make_berry_connection(kmb, H, berry_broadening);
  auto Ex_dot_R = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ex),
      anomtrans::unowned(R), kmb.Nbands);
  auto Ey_dot_R = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ey),
      anomtrans::unowned(R), kmb.Nbands);

  // Chemical potential in the gap.
  // Valence band maximum at K: -0.504 eV, Gamma: -1.005 eV.
  // VBM S-O splitting at K: -0.807 eV - (-1.276 eV) = 0.469 eV.
  // Conduction band minimum at K: 0.762 eV, Q: 0.798 eV.
  // CBM S-O splitting at K: 0.484 eV - 0.444 eV = 0.04 eV.
  double mu = 0.0;

  // Gap = 1.266 eV.
  // Choose omega \approx gap so that we produce resonant excitations from VBM to CBM.
  // TODO - interested in what happens when omega is below the gap.
  double omega = 1.266 + 0.004;

  // Equilibrium density matrix <rho_{0,0}>. Make two since we will have two DM graphs,
  // one for E ~ \hat{x} cos(\omega t) and one for E ~ \hat{y} sin(\omega t).
  auto dm_rho0_Ex_cos = anomtrans::make_eq_node<anomtrans::DynDMGraphNode>(Ekm.v, beta, mu);
  auto dm_rho0_Ey_sin = anomtrans::make_eq_node<anomtrans::DynDMGraphNode>(Ekm.v, beta, mu);
  Vec rho0_km;
  ierr = VecDuplicate(Ekm.v, &rho0_km);CHKERRXX(ierr);
  ierr = MatGetDiagonal(dm_rho0_Ex_cos->rho.M, rho0_km);CHKERRXX(ierr);

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
  ierr = MatSetNullSpace(collision.M, nullspace);CHKERRXX(ierr);
  // NOTE rho0_normalized must not be modified after this call until we are done with nullspace.

  // Add <rho_{1,1}> and <rho_{-1,1}> for (Ex, cos) and (Ey, sin) cases.
  anomtrans::add_dynamic_electric_n_nonzero(dm_rho0_Ex_cos, boost::none, 1, omega, kmb, Ex_dot_grad_k.M,
      Ex_dot_R.M, H, berry_broadening, anomtrans::DynVariation::cos);
  anomtrans::add_dynamic_electric_n_nonzero(boost::none, dm_rho0_Ex_cos, -1, omega, kmb, Ex_dot_grad_k.M,
      Ex_dot_R.M, H, berry_broadening, anomtrans::DynVariation::cos);
  auto dm_rho_1_1_Ex_cos = dm_rho0_Ex_cos->children[anomtrans::DynDMDerivedBy::omega_inv_DE_up];
  auto dm_rho_m1_1_Ex_cos = dm_rho0_Ex_cos->children[anomtrans::DynDMDerivedBy::omega_inv_DE_down];

  anomtrans::add_dynamic_electric_n_nonzero(dm_rho0_Ey_sin, boost::none, 1, omega, kmb, Ey_dot_grad_k.M,
      Ey_dot_R.M, H, berry_broadening, anomtrans::DynVariation::sin);
  anomtrans::add_dynamic_electric_n_nonzero(boost::none, dm_rho0_Ey_sin, -1, omega, kmb, Ey_dot_grad_k.M,
      Ey_dot_R.M, H, berry_broadening, anomtrans::DynVariation::sin);
  auto dm_rho_1_1_Ey_sin = dm_rho0_Ey_sin->children[anomtrans::DynDMDerivedBy::omega_inv_DE_up];
  auto dm_rho_m1_1_Ey_sin = dm_rho0_Ey_sin->children[anomtrans::DynDMDerivedBy::omega_inv_DE_down];

  // Add <rho_{0, 2}> for (Ex, cos) and (Ey, sin) cases.
  anomtrans::add_dynamic_electric_n_zero(dm_rho_m1_1_Ex_cos, dm_rho_1_1_Ex_cos, omega, kmb, Ex_dot_grad_k.M,
      Ex_dot_R.M, ksp, H, sigma, disorder_term_od, berry_broadening, anomtrans::DynVariation::cos);
  anomtrans::add_dynamic_electric_n_zero(dm_rho_m1_1_Ey_sin, dm_rho_1_1_Ey_sin, omega, kmb, Ey_dot_grad_k.M,
      Ey_dot_R.M, ksp, H, sigma, disorder_term_od, berry_broadening, anomtrans::DynVariation::sin);

  // Valley (anomalous) hall effect at second order in electric field
  // from <rho_{0, 2}>. Intrinsic contribution. Current in y, electric field in x.
  auto dm_n_0_2_Ex_cos = dm_rho_m1_1_Ex_cos->children[anomtrans::DynDMDerivedBy::Kdd_inv_DE_up];
  auto dm_S_0_2_intrinsic_Ex_cos = dm_rho_m1_1_Ex_cos->children[anomtrans::DynDMDerivedBy::P_inv_DE_up];
  auto dm_S_0_2_extrinsic_Ex_cos = dm_rho_m1_1_Ex_cos->children[anomtrans::DynDMDerivedBy::P_inv_Kod_up];

  auto dm_n_0_2_Ey_sin = dm_rho_m1_1_Ey_sin->children[anomtrans::DynDMDerivedBy::Kdd_inv_DE_up];
  auto dm_S_0_2_intrinsic_Ey_sin = dm_rho_m1_1_Ey_sin->children[anomtrans::DynDMDerivedBy::P_inv_DE_up];
  auto dm_S_0_2_extrinsic_Ey_sin = dm_rho_m1_1_Ey_sin->children[anomtrans::DynDMDerivedBy::P_inv_Kod_up];

  Mat n_0_2_plus, S_0_2_intrinsic_plus, S_0_2_extrinsic_plus;
  ierr = MatDuplicate(dm_n_0_2_Ex_cos->rho.M, MAT_COPY_VALUES,
      &n_0_2_plus);CHKERRXX(ierr);
  ierr = MatAXPY(n_0_2_plus, 1.0, dm_n_0_2_Ey_sin->rho.M,
      DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
  ierr = MatDuplicate(dm_S_0_2_intrinsic_Ex_cos->rho.M, MAT_COPY_VALUES,
      &S_0_2_intrinsic_plus);CHKERRXX(ierr);
  ierr = MatAXPY(S_0_2_intrinsic_plus, 1.0, dm_S_0_2_intrinsic_Ey_sin->rho.M,
      DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
  ierr = MatDuplicate(dm_S_0_2_extrinsic_Ex_cos->rho.M, MAT_COPY_VALUES,
      &S_0_2_extrinsic_plus);CHKERRXX(ierr);
  ierr = MatAXPY(S_0_2_extrinsic_plus, 1.0, dm_S_0_2_extrinsic_Ey_sin->rho.M,
      DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);

  bool ret_Mat = false;
  auto sigma_n_0_2_plus = anomtrans::calculate_current_ev(kmb, v_op, n_0_2_plus, ret_Mat);
  auto sigma_S_0_2_intrinsic_plus = anomtrans::calculate_current_ev(kmb, v_op,
      S_0_2_intrinsic_plus, ret_Mat);
  auto sigma_S_0_2_extrinsic_plus = anomtrans::calculate_current_ev(kmb, v_op,
      S_0_2_extrinsic_plus, ret_Mat);
  PetscReal sigma_n_0_2_xx_plus = sigma_n_0_2_plus.at(0).first.real();
  PetscReal sigma_n_0_2_xy_plus = sigma_n_0_2_plus.at(1).first.real();
  PetscReal sigma_S_0_2_intrinsic_xx_plus = sigma_S_0_2_intrinsic_plus.at(0).first.real();
  PetscReal sigma_S_0_2_intrinsic_xy_plus = sigma_S_0_2_intrinsic_plus.at(1).first.real();
  PetscReal sigma_S_0_2_extrinsic_xx_plus = sigma_S_0_2_extrinsic_plus.at(0).first.real();
  PetscReal sigma_S_0_2_extrinsic_xy_plus = sigma_S_0_2_extrinsic_plus.at(1).first.real();

  auto spin_n_0_2_plus = anomtrans::calculate_spin_ev(kmb, spin_op, n_0_2_plus, ret_Mat);
  auto spin_S_0_2_intrinsic_plus = anomtrans::calculate_spin_ev(kmb, spin_op,
      S_0_2_intrinsic_plus, ret_Mat);
  auto spin_S_0_2_extrinsic_plus = anomtrans::calculate_spin_ev(kmb, spin_op,
      S_0_2_extrinsic_plus, ret_Mat);
  PetscReal spin_n_0_2_z_plus = spin_n_0_2_plus.at(2).first.real();
  PetscReal spin_S_0_2_intrinsic_z_plus = spin_S_0_2_intrinsic_plus.at(2).first.real();
  PetscReal spin_S_0_2_extrinsic_z_plus = spin_S_0_2_extrinsic_plus.at(2).first.real();

  auto js_n_0_2_plus = anomtrans::calculate_spin_current_ev(kmb, spin_op, v_op, n_0_2_plus,
      ret_Mat);
  auto js_S_0_2_intrinsic_plus = anomtrans::calculate_spin_current_ev(kmb, spin_op, v_op,
      S_0_2_intrinsic_plus, ret_Mat);
  auto js_S_0_2_extrinsic_plus = anomtrans::calculate_spin_current_ev(kmb, spin_op, v_op,
      S_0_2_extrinsic_plus, ret_Mat);
  PetscReal js_n_0_2_sz_vy_plus = js_n_0_2_plus.at(2).at(1).first.real();
  PetscReal js_S_0_2_intrinsic_sz_vy_plus = js_S_0_2_intrinsic_plus.at(2).at(1).first.real();
  PetscReal js_S_0_2_extrinsic_sz_vy_plus = js_S_0_2_extrinsic_plus.at(2).at(1).first.real();

  // Done with PETSc data.
  ierr = MatDestroy(&n_0_2_plus);CHKERRXX(ierr);
  ierr = MatDestroy(&S_0_2_intrinsic_plus);CHKERRXX(ierr);
  ierr = MatDestroy(&S_0_2_extrinsic_plus);CHKERRXX(ierr);

  ierr = MatNullSpaceDestroy(&nullspace);CHKERRXX(ierr);
  ierr = VecDestroy(&rho0_normalized);CHKERRXX(ierr);
  ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);

  ierr = KSPDestroy(&ksp);CHKERRXX(ierr);

  json j_out = {
    {"sigma_n_0_2_xx_plus", sigma_n_0_2_xx_plus},
    {"sigma_n_0_2_xy_plus", sigma_n_0_2_xy_plus},
    {"sigma_S_0_2_intrinsic_xx_plus", sigma_S_0_2_intrinsic_xx_plus},
    {"sigma_S_0_2_intrinsic_xy_plus", sigma_S_0_2_intrinsic_xy_plus},
    {"sigma_S_0_2_extrinsic_xx_plus", sigma_S_0_2_extrinsic_xx_plus},
    {"sigma_S_0_2_extrinsic_xy_plus", sigma_S_0_2_extrinsic_xy_plus},
    {"spin_n_0_2_z_plus", spin_n_0_2_z_plus},
    {"spin_S_0_2_intrinsic_z_plus", spin_S_0_2_intrinsic_z_plus},
    {"spin_S_0_2_extrinsic_z_plus", spin_S_0_2_extrinsic_z_plus},
    {"js_n_0_2_sz_vy_plus", js_n_0_2_sz_vy_plus},
    {"js_S_0_2_intrinsic_sz_vy_plus", js_S_0_2_intrinsic_sz_vy_plus},
    {"js_S_0_2_extrinsic_sz_vy_plus", js_S_0_2_extrinsic_sz_vy_plus}
  };

  std::stringstream outpath;
  outpath << "wannier90_Hamiltonian_WSe2_test_out.json";

  std::ofstream fp_out(outpath.str());
  fp_out << j_out.dump();
  fp_out.close();
}
