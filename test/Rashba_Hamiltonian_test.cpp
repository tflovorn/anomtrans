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
#include "models/Rashba_Hamiltonian.h"
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

/** @brief Check that the Berry connection for the Rashba model has the expected form.
 */
TEST( Rashba_electric, berry_connection ) {
  const std::size_t k_dim = 2;

  double t = 1.0;
  double tr = 0.2;
  assert(tr > 0.0);

  std::array<unsigned int, k_dim> Nk = {8, 8};

  // Choose berry_broadening small enough that
  // (E_k^- - E_k^+)^2 + berry_broadening^2
  // always rounds to (E_k^- - E_k^+)^2 except in the case of a degeneracy.
  auto macheps = std::numeric_limits<PetscReal>::epsilon();
  double berry_broadening = macheps * 1e-3 * tr;

  unsigned int Nbands = 2;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands);

  anomtrans::Rashba_Hamiltonian H(t, tr, kmb);
  std::array<Mat, k_dim> R = anomtrans::make_berry_connection(kmb, H, berry_broadening);

  auto Rpm_expected = [](anomtrans::kVals<k_dim> k)->std::array<PetscScalar, k_dim> {
    // Assumes a = 1 and k-points where denom = 0 are avoided.
    double kx_a = 2.0*anomtrans::pi*k.at(0);
    double ky_a = 2.0*anomtrans::pi*k.at(1);

    double sx = std::sin(kx_a);
    double cx = std::cos(kx_a);
    double sy = std::sin(ky_a);
    double cy = std::cos(ky_a);

    double denom = 2.0 * (std::pow(sx, 2.0) + std::pow(sy, 2.0));

    return {-cx * sy / denom, cy * sx / denom};
  };

  auto ediff_expected = [tr](anomtrans::kVals<k_dim> k)->double {
    double kx_a = 2.0*anomtrans::pi*k.at(0);
    double ky_a = 2.0*anomtrans::pi*k.at(1);

    double sx = std::sin(kx_a);
    double sy = std::sin(ky_a);

    return -4.0 * tr * std::sqrt(std::pow(sx, 2.0) + std::pow(sy, 2.0));
  };

  auto grad_expected = [t, tr](anomtrans::kVals<k_dim> k)->std::array<PetscScalar, k_dim> {
    double kx_a = 2.0*anomtrans::pi*k.at(0);
    double ky_a = 2.0*anomtrans::pi*k.at(1);

    double sx = std::sin(kx_a);
    double cx = std::cos(kx_a);
    double sy = std::sin(ky_a);
    double cy = std::cos(ky_a);

    double denom = std::sqrt(std::pow(sx, 2.0) + std::pow(sy, 2.0));

    double num_im_x = -2.0 * tr * cx * sy;
    double num_im_y = 2.0 * tr * cy * sx;

    return {std::complex<double>(0.0, num_im_x / denom),
        std::complex<double>(0.0, num_im_y / denom)};
  };

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(R.at(0), &begin, &end);CHKERRXX(ierr);

  for (PetscInt ikm = begin; ikm < end; ikm++) {
    anomtrans::kmComps<k_dim> km = kmb.decompose(ikm);
    unsigned int m = std::get<1>(km);

    if (m == 1) {
      // Interested only in R^{+,-}. Skip R^{-, m'}.
      continue;
    }

    auto k_val = std::get<0>(kmb.km_at(km));
    if ((k_val.at(0) == 0.0 or k_val.at(0) == 0.5)
        and (k_val.at(1) == 0.0 or k_val.at(1) == 0.5)) {
      // Avoid points where the chosen eigenbasis becomes singular.
      continue;
    }

    auto Rk_pm = Rpm_expected(k_val);

    for (std::size_t dc = 0; dc < k_dim; dc++) {
      PetscInt ncols;
      const PetscInt *cols;
      const PetscScalar *vals;
      ierr = MatGetRow(R.at(dc), ikm, &ncols, &cols, &vals);CHKERRXX(ierr);

      for (PetscInt col_index = 0; col_index < ncols; col_index++) {
        PetscInt col = cols[col_index];
        PetscScalar val = vals[col_index];

        anomtrans::kmComps<k_dim> kmp = kmb.decompose(col);
        unsigned int mp = std::get<1>(kmp);

        // R should be k-diagonal: entries for k' != k should not be present.
        ASSERT_TRUE(std::get<0>(kmp) == std::get<0>(km));

        if (mp == 0) {
          // Interested only in R^{+, -}. Skip R^{+, +}.
          continue;
        }

        double ediff = H.energy(kmp) - H.energy(km);
        ASSERT_TRUE(anomtrans::scalars_approx_equal(ediff_expected(k_val), ediff, 10.0*macheps*tr, 10.0*macheps));

        auto grad_k_expect = grad_expected(k_val).at(dc);
        auto grad = H.gradient(km, mp).at(dc);

        ASSERT_TRUE(anomtrans::scalars_approx_equal(grad_k_expect, grad, 100.0*macheps*tr, 100.0*macheps));
        ASSERT_TRUE(anomtrans::scalars_approx_equal(Rk_pm.at(dc), val, 100.0*macheps, 100.0*macheps));
      }

      ierr = MatRestoreRow(R.at(dc), ikm, &ncols, &cols, &vals);CHKERRXX(ierr);
    }
  }
}

/** @brief Check that the electric-field response n_E and S_E of the Rashba
 *         model has the expected form.
 */
TEST( Rashba_electric, Rashba_electric ) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  const std::size_t k_dim = 2;

  double t = 1.0;
  double tr = 0.2;

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

  unsigned int Nbands = 2;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands);

  anomtrans::Rashba_Hamiltonian H(t, tr, kmb);

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
  double U0 = 1.0*t;

  double sigma_min = anomtrans::get_sigma_min(max_energy_difference);

  if (sigma < sigma_min) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: sigma < sigma_min: sigma = %e ; sigma_min = %e\n", sigma, sigma_min);
  }

  std::array<double, k_dim> Ehat = {1.0, 0.0};

  Vec Ekm = anomtrans::get_energies(kmb, H);

  std::array<Mat, k_dim> v_op = anomtrans::calculate_velocity(kmb, H);
  std::array<Mat, 3> spin_op = anomtrans::calculate_spin_operator(kmb, H);

  PetscInt Ekm_min_index, Ekm_max_index;
  PetscReal Ekm_min, Ekm_max;
  PetscErrorCode ierr = VecMin(Ekm, &Ekm_min_index, &Ekm_min);CHKERRXX(ierr);
  ierr = VecMax(Ekm, &Ekm_max_index, &Ekm_max);CHKERRXX(ierr);

  std::size_t Nk_tot = anomtrans::get_Nk_total(Nk);
  double U0_sq = U0*U0;
  double disorder_coeff = U0_sq / Nk_tot;
  /*
  auto disorder_term = [Nbands, H, disorder_coeff](PetscInt ikm1, PetscInt ikm2)->double {
    return disorder_coeff*anomtrans::on_site_diagonal_disorder_band_preserved(Nbands, H,
        ikm1, ikm2);
  };
  */
  double Lambda = 1e-12;
  anomtrans::SpatialDisorderCorrelation<k_dim> ULambda(kmb, D, Lambda);
  auto disorder_term = [Nbands, H, ULambda, disorder_coeff](PetscInt ikm1, PetscInt ikm2)->double {
    return disorder_coeff*anomtrans::spatially_correlated_diagonal_disorder_band_preserved(Nbands,
        H, ULambda, ikm1, ikm2);
  };
  auto disorder_term_od = [Nbands, H, ULambda, disorder_coeff](PetscInt ikm1, PetscInt ikm2,
      PetscInt ikm3)->std::complex<double> {
    return disorder_coeff*anomtrans::spatially_correlated_diagonal_disorder(Nbands,
        H, ULambda, ikm1, ikm2, ikm3);
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
  double berry_broadening = 1e-8;
  auto R = anomtrans::make_berry_connection(kmb, H, berry_broadening);
  auto Ehat_dot_R = anomtrans::Mat_from_sum_const(anomtrans::make_complex_array(Ehat), R, kmb.Nbands);

  auto Omega = anomtrans::make_berry_curvature(kmb, H, berry_broadening);

  auto mus = anomtrans::linspace(Ekm_min, Ekm_max, num_mus);

  std::vector<std::vector<PetscReal>> all_rho0;
  std::vector<std::vector<PetscReal>> all_n_E;
  std::vector<std::vector<PetscReal>> all_S_E_pm_real;
  std::vector<std::vector<PetscReal>> all_S_E_pm_imag;
  std::vector<std::vector<PetscReal>> all_S_E_pm_int_real;
  std::vector<std::vector<PetscReal>> all_S_E_pm_int_imag;
  std::vector<std::vector<PetscReal>> all_S_E_pm_ext_real;
  std::vector<std::vector<PetscReal>> all_S_E_pm_ext_imag;
  std::vector<PetscReal> all_sigma_xxs;
  std::vector<PetscReal> all_sys;
  std::vector<PetscReal> all_js_sz_vys_intrinsic;
  std::vector<PetscReal> all_js_sz_vys_extrinsic;
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
    // sigma_xx = -e Tr[v_x <rho_{E_x}>] / E_y
    PetscScalar sigma_xx = anomtrans::calculate_current_ev(kmb, v_op, dm_n_E->rho).at(0);
    all_sigma_xxs.push_back(sigma_xx.real());

    PetscScalar sy = anomtrans::calculate_spin_ev(kmb, spin_op, dm_n_E->rho).at(1);
    all_sys.push_back(sy.real());

    auto collected_rho0 = anomtrans::split_scalars(anomtrans::collect_contents(rho0_km)).first;
    all_rho0.push_back(collected_rho0);
    auto collected_n_E = anomtrans::split_scalars(anomtrans::collect_contents(n_E)).first;
    all_n_E.push_back(collected_n_E);

    auto dm_S_E_intrinsic = dm_rho0->children[anomtrans::StaticDMDerivedBy::P_inv_DE];
    auto dm_S_E_extrinsic = dm_n_E->children[anomtrans::StaticDMDerivedBy::P_inv_Kod];

    PetscScalar js_sz_vy_intrinsic = anomtrans::calculate_spin_current_ev(kmb, spin_op, v_op,
        dm_S_E_intrinsic->rho).at(2).at(1);
    all_js_sz_vys_intrinsic.push_back(js_sz_vy_intrinsic.real());

    PetscScalar js_sz_vy_extrinsic = anomtrans::calculate_spin_current_ev(kmb, spin_op, v_op,
        dm_S_E_extrinsic->rho).at(2).at(1);
    all_js_sz_vys_extrinsic.push_back(js_sz_vy_extrinsic.real());

    auto collected_S_E_pm_int = anomtrans::split_scalars(anomtrans::collect_band_elem(kmb, dm_S_E_intrinsic->rho, 0, 1));
    all_S_E_pm_int_real.push_back(collected_S_E_pm_int.first);
    all_S_E_pm_int_imag.push_back(collected_S_E_pm_int.second);

    auto collected_S_E_pm_ext = anomtrans::split_scalars(anomtrans::collect_band_elem(kmb, dm_S_E_extrinsic->rho, 0, 1));
    all_S_E_pm_ext_real.push_back(collected_S_E_pm_ext.first);
    all_S_E_pm_ext_imag.push_back(collected_S_E_pm_ext.second);

    Mat S_E_total;
    ierr = MatDuplicate(dm_S_E_intrinsic->rho, MAT_COPY_VALUES, &S_E_total);CHKERRXX(ierr);
    ierr = MatAXPY(S_E_total, 1.0, dm_S_E_extrinsic->rho, DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);

    auto collected_S_E_pm = anomtrans::split_scalars(anomtrans::collect_band_elem(kmb, S_E_total, 0, 1));
    all_S_E_pm_real.push_back(collected_S_E_pm.first);
    all_S_E_pm_imag.push_back(collected_S_E_pm.second);

    ierr = MatDestroy(&S_E_total);CHKERRXX(ierr);
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
    ierr = MatDestroy(&(spin_op.at(dc)));CHKERRXX(ierr);
    ierr = VecDestroy(&(Omega.at(dc)));CHKERRXX(ierr);
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
      {"_oneband_S_E_pm_real", all_S_E_pm_real},
      {"_oneband_S_E_pm_imag", all_S_E_pm_imag},
      {"_oneband_S_E_pm_int_real", all_S_E_pm_int_real},
      {"_oneband_S_E_pm_int_imag", all_S_E_pm_int_imag},
      {"_oneband_S_E_pm_ext_real", all_S_E_pm_ext_real},
      {"_oneband_S_E_pm_ext_imag", all_S_E_pm_ext_imag},
      {"_series_sigma_xx", all_sigma_xxs},
      {"_series_sy", all_sys},
      {"_series_js_sz_vy_intrinsic", all_js_sz_vys_intrinsic},
      {"_series_js_sz_vy_extrinsic", all_js_sz_vys_extrinsic}
    };

    std::stringstream outpath;
    outpath << "Rashba_Hamiltonian_test_out.json";

    std::ofstream fp_out(outpath.str());
    fp_out << j_out.dump();
    fp_out.close();

    // Check for changes from saved old result.
    boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
    if (not test_data_dir) {
      throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
    }

    std::stringstream known_path;
    known_path << *test_data_dir << "/Rashba_Hamiltonian_test_out.json";

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

    // t is an appropriate scale for E.
    auto macheps = std::numeric_limits<PetscReal>::epsilon();
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["Ekm"].get<std::vector<PetscReal>>(),
        j_known["Ekm"].get<std::vector<PetscReal>>(),
        100.0*t*macheps, 10.0*macheps) );

    // TODO - what are appropriate scales for conductivities? Absolute error depends on disorder scale.
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_series_sigma_xx"].get<std::vector<PetscReal>>(),
        j_known["_series_sigma_xx"].get<std::vector<PetscReal>>(),
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
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_oneband_S_E_pm_real"].get<std::vector<std::vector<PetscReal>>>(),
        j_known["_oneband_S_E_pm_real"].get<std::vector<std::vector<PetscReal>>>(),
        100.0*macheps, 10.0*macheps) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["_oneband_S_E_pm_imag"].get<std::vector<std::vector<PetscReal>>>(),
        j_known["_oneband_S_E_pm_imag"].get<std::vector<std::vector<PetscReal>>>(),
        100.0*macheps, 10.0*macheps) );
  }
}
