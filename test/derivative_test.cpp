#include <cstddef>
#include <cmath>
#include <limits>
#include <tuple>
#include <array>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <tuple>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include <json.hpp>
#include "util/MPIPrettyUnitTestResultPrinter.h"
#include "util/constants.h"
#include "util/util.h"
#include "grid_basis.h"
#include "models/square_tb_spectrum.h"
#include "observables/energy.h"
#include "util/vec.h"
#include "observables/rho0.h"
#include "derivative.h"

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

TEST( derivative, linear ) {
  const std::size_t k_dim = 3;
  std::array<unsigned int, k_dim> Nk = {8, 8, 8};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands);

  std::array<PetscScalar, k_dim> coeffs = {1.0, 2.0, 3.0};

  // f(k) = \sum_d c_d k_d
  // The derivative df/dk_d = c_d is given exactly by the first-order
  // forward finite difference and second-order central difference.
  auto f_linear = [&kmb, &coeffs](PetscInt ikm)->PetscScalar {
    auto k = std::get<0>(anomtrans::km_at(kmb.Nk, kmb.decompose(ikm)));

    PetscScalar result = 0.0;
    for (std::size_t d = 0; d < k_dim; d++) {
      result += coeffs.at(d) * k.at(d);
    }
    return result;
  };

  Vec f_vals = anomtrans::vector_index_apply(kmb.end_ikm, f_linear);

  auto check_deriv = [&kmb, &coeffs, f_vals](anomtrans::DerivStencil<1> stencil) {
    auto d_dk = anomtrans::make_d_dk_recip(kmb, stencil);

    Vec df_dk;
    PetscErrorCode ierr = VecDuplicate(f_vals, &df_dk);CHKERRXX(ierr);

    for (std::size_t d = 0; d < k_dim; d++) {
      ierr = MatMult(d_dk.at(d), f_vals, df_dk);CHKERRXX(ierr);

      std::vector<PetscInt> df_dk_ikms;
      std::vector<PetscScalar> df_dk_vals;
      std::tie(df_dk_ikms, df_dk_vals) = anomtrans::get_local_contents(df_dk);

      auto macheps = std::numeric_limits<PetscReal>::epsilon();
      auto eps_abs = 2.0 * macheps * std::abs(coeffs.at(d));
      auto eps_rel = 2.0 * macheps;

      auto expected = coeffs.at(d);

      for (std::size_t i = 0; i < df_dk_ikms.size(); i++) {
        auto ikm = df_dk_ikms.at(i);
        auto v = df_dk_vals.at(i);

        auto k_comps = std::get<0>(kmb.decompose(ikm));
        if (k_comps.at(d) == 0 or k_comps.at(d) == kmb.Nk.at(d) - 1) {
          // Our function f_linear is not periodic, so the derivative is incorrect
          // at the boundary of the Brillouin zone.
          continue;
        }

        ASSERT_TRUE(anomtrans::scalars_approx_equal(v, expected, eps_abs, eps_rel));
      }
    }

    ierr = VecDestroy(&df_dk);CHKERRXX(ierr);
  };

  anomtrans::DerivStencil<1> stencil_forward(anomtrans::DerivApproxType::forward, 1);
  anomtrans::DerivStencil<1> stencil_central(anomtrans::DerivApproxType::central, 2);

  check_deriv(stencil_forward);
  check_deriv(stencil_central);

  PetscErrorCode ierr = VecDestroy(&f_vals);CHKERRXX(ierr);
}

TEST( derivative, quadratic ) {
  const std::size_t k_dim = 3;
  std::array<unsigned int, k_dim> Nk = {8, 8, 8};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands);

  std::array<PetscScalar, k_dim> coeffs = {1.0, 2.0, 3.0};

  // f(k) = \sum_d c_d (k_d)^2
  // The derivative df/dk_d = 2 * c_d * k_d is given exactly by
  // the second-order central difference.
  auto f_quad = [&kmb, &coeffs](PetscInt ikm)->PetscScalar {
    auto k = std::get<0>(anomtrans::km_at(kmb.Nk, kmb.decompose(ikm)));

    PetscScalar result = 0.0;
    for (std::size_t d = 0; d < k_dim; d++) {
      result += coeffs.at(d) * std::pow(k.at(d), 2.0);
    }
    return result;
  };

  Vec f_vals = anomtrans::vector_index_apply(kmb.end_ikm, f_quad);

  auto check_deriv = [&kmb, &coeffs, f_vals](anomtrans::DerivStencil<1> stencil) {
    auto d_dk = anomtrans::make_d_dk_recip(kmb, stencil);

    Vec df_dk;
    PetscErrorCode ierr = VecDuplicate(f_vals, &df_dk);CHKERRXX(ierr);

    for (std::size_t d = 0; d < k_dim; d++) {
      ierr = MatMult(d_dk.at(d), f_vals, df_dk);CHKERRXX(ierr);

      std::vector<PetscInt> df_dk_ikms;
      std::vector<PetscScalar> df_dk_vals;
      std::tie(df_dk_ikms, df_dk_vals) = anomtrans::get_local_contents(df_dk);

      auto macheps = std::numeric_limits<PetscReal>::epsilon();
      auto eps_abs = 2.0 * macheps * std::abs(coeffs.at(d));
      auto eps_rel = 2.0 * macheps;

      for (std::size_t i = 0; i < df_dk_ikms.size(); i++) {
        auto ikm = df_dk_ikms.at(i);
        auto v = df_dk_vals.at(i);

        auto k_comps = std::get<0>(kmb.decompose(ikm));
        if (k_comps.at(d) == 0 or k_comps.at(d) == kmb.Nk.at(d) - 1) {
          // Our function f_quad is not periodic, so the derivative is incorrect
          // at the boundary of the Brillouin zone.
          continue;
        }

        auto k = std::get<0>(anomtrans::km_at(kmb.Nk, kmb.decompose(ikm)));
        auto expected = 2.0 * coeffs.at(d) * k.at(d);

        ASSERT_TRUE(anomtrans::scalars_approx_equal(v, expected, eps_abs, eps_rel));
      }
    }

    ierr = VecDestroy(&df_dk);CHKERRXX(ierr);
  };

  anomtrans::DerivStencil<1> stencil_central(anomtrans::DerivApproxType::central, 2);

  check_deriv(stencil_central);

  PetscErrorCode ierr = VecDestroy(&f_vals);CHKERRXX(ierr);
}

TEST( derivative, square_TB_fermi_surface ) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  const std::size_t k_dim = 2;

  // This Nk value is set to keep regression test data compact.
  // For adequate Fermi surface images, set Nk = {128, 128}.
  std::array<unsigned int, k_dim> Nk = {8, 8};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands);

  double t = 1.0;
  double tp = -0.3;
  anomtrans::square_tb_Hamiltonian H(t, tp, Nk);

  std::array<double, k_dim> a1 = {1.0, 0.0};
  std::array<double, k_dim> a2 = {0.0, 1.0};
  anomtrans::DimMatrix<k_dim> D = {a1, a2};

  double beta = 10.0/t;

  Vec Ekm = anomtrans::get_energies(kmb, H);

  PetscInt Ekm_min_index, Ekm_max_index;
  PetscReal Ekm_min, Ekm_max;
  PetscErrorCode ierr = VecMin(Ekm, &Ekm_min_index, &Ekm_min);CHKERRXX(ierr);
  ierr = VecMax(Ekm, &Ekm_max_index, &Ekm_max);CHKERRXX(ierr);

  std::vector<PetscInt> local_rows = std::get<0>(anomtrans::get_local_contents(Ekm));

  const unsigned int deriv_approx_order = 2;
  anomtrans::DerivStencil<1> stencil(anomtrans::DerivApproxType::central, deriv_approx_order);
  auto d_dk = anomtrans::make_d_dk_recip(kmb, stencil);
  auto d_dk_Cart = anomtrans::make_d_dk_Cartesian(D, kmb, stencil);

  // Since a1 = \hat{x}, a2 = \hat{y}, we should have d_dk == 2*pi*d_dk_Cart.
  for (std::size_t d = 0; d < k_dim; d++) {
    Mat d_dk_Cart_d_2pi;
    MatDuplicate(d_dk_Cart.at(d), MAT_COPY_VALUES, &d_dk_Cart_d_2pi);
    MatScale(d_dk_Cart_d_2pi, 2*anomtrans::pi);

    // Tried to use PETSc MatEqual() for this, got confusing errors about unequal
    // matrix dimensions. Checking with MatGetSize() and MatGetOwnershipRange()
    // showed that d_dk.at(d) and d_dk_Cart.at(d) had equal sizes and local row
    // distributions. Not sure what the source of the error was.
    // The appropriate scale for floating-point comparison here is 1 (or more
    // generally would be a, if we had a1 = {a, 0}, a2 = {0, a}).
    double tol = 2*std::numeric_limits<double>::epsilon();
    ASSERT_TRUE( anomtrans::check_Mat_equal(d_dk.at(d), d_dk_Cart_d_2pi, tol) );

    ierr = MatDestroy(&d_dk_Cart_d_2pi);CHKERRXX(ierr);
  }

  unsigned int num_mus = 40;
  auto mus = anomtrans::linspace(Ekm_min, Ekm_max, num_mus);

  std::vector<std::vector<PetscReal>> all_rho0;
  std::vector<std::vector<PetscReal>> all_norm_d_rho0_dk;
  for (auto mu : mus) {
    Vec rho0_km = anomtrans::make_rho0(Ekm, beta, mu);

    std::array<Vec, k_dim> d_rho0_dk;
    for (std::size_t d = 0; d < k_dim; d++) {
      Vec d_rho0_dk_d;
      PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm, &d_rho0_dk_d);CHKERRXX(ierr);
      // d_rho0_dk(d) = [d_dk]_{d} * rho0
      ierr = MatMult(d_dk.at(d), rho0_km, d_rho0_dk_d);CHKERRXX(ierr);
      
      ierr = VecAssemblyBegin(d_rho0_dk_d);CHKERRXX(ierr);
      ierr = VecAssemblyEnd(d_rho0_dk_d);CHKERRXX(ierr);

      d_rho0_dk.at(d) = d_rho0_dk_d;
    }

    // TODO could factor out the block below into a function that takes a std::vector of
    // Vecs and applies a function to corresponding elements, yi = f(x1i, x2i, ...)
    // in the same manner as vector_elem_apply.
    Vec norm_d_rho0_dk;
    ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm, &norm_d_rho0_dk);CHKERRXX(ierr);

    PetscInt begin, end;
    ierr = VecGetOwnershipRange(norm_d_rho0_dk, &begin, &end);CHKERRXX(ierr);
    PetscInt num_local_rows = end - begin;

    std::array<anomtrans::IndexValPairs, k_dim> local_d_rho0_dk;
    for (std::size_t d = 0; d < k_dim; d++) {
      local_d_rho0_dk.at(d) = anomtrans::get_local_contents(d_rho0_dk.at(d));
      auto d_rho0_dk_local_rows = std::get<0>(local_d_rho0_dk.at(d));
      ASSERT_EQ( begin, d_rho0_dk_local_rows.at(0) );
      ASSERT_EQ( end, d_rho0_dk_local_rows.at(num_local_rows - 1) + 1 );
    }

    std::vector<PetscScalar> norm_vals;
    norm_vals.reserve(num_local_rows);

    for (PetscInt i = 0; i < num_local_rows; i++) {
      double norm2 = 0;
      for (std::size_t d = 0; d < k_dim; d++) {
        double component = std::get<1>(local_d_rho0_dk.at(d)).at(i).real();
        norm2 += component*component;
      }
      double norm = std::sqrt(norm2);
      norm_vals.push_back(norm);
    }

    ierr = VecSetValues(norm_d_rho0_dk, local_rows.size(), local_rows.data(),
        norm_vals.data(), INSERT_VALUES);CHKERRXX(ierr);

    ierr = VecAssemblyBegin(norm_d_rho0_dk);CHKERRXX(ierr);
    ierr = VecAssemblyEnd(norm_d_rho0_dk);CHKERRXX(ierr);

    auto collected_rho0 = anomtrans::split_scalars(anomtrans::collect_contents(rho0_km));
    auto collected_norm_d_rho0_dk = anomtrans::split_scalars(anomtrans::collect_contents(norm_d_rho0_dk));

    all_rho0.push_back(collected_rho0.first);
    all_norm_d_rho0_dk.push_back(collected_norm_d_rho0_dk.first);

    ierr = VecDestroy(&norm_d_rho0_dk);CHKERRXX(ierr);
    ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);
    for (std::size_t d = 0; d < k_dim; d++) {
      ierr = VecDestroy(&(d_rho0_dk.at(d)));CHKERRXX(ierr);
    }
  }

  auto collected_Ekm = anomtrans::split_scalars(anomtrans::collect_contents(Ekm)).first;

  // Done with PETSc data.
  ierr = VecDestroy(&Ekm);CHKERRXX(ierr);
  for (std::size_t d = 0; d < k_dim; d++) {
    ierr = MatDestroy(&(d_dk.at(d)));CHKERRXX(ierr);
    ierr = MatDestroy(&(d_dk_Cart.at(d)));CHKERRXX(ierr);
  }

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
      {"norm_d_rho0_dk", all_norm_d_rho0_dk}
    };

    std::stringstream outpath;
    outpath << "derivative_test_out.json";

    std::ofstream fp_out(outpath.str());
    fp_out << j_out.dump();
    fp_out.close();

    // Check for changes from saved old result.
    boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
    if (not test_data_dir) {
      throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
    }

    // TODO could use boost::filesystem here to build path.
    // Tried, had issue with 'undefined reference to operator/='.
    std::stringstream known_path;
    known_path << *test_data_dir << "/derivative_test_out.json";

    json j_known;
    std::ifstream fp_k(known_path.str());
    if (not fp_k.good()) {
      throw std::runtime_error("could not open file in check_json_equal");
    }
    fp_k >> j_known;
    fp_k.close();

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

    // 1 is an appropriate scale for rho: elements range from 0 to 1.
    // TODO using 1 as scale for norm_d_rho0_dk also. Is this appropriate?
    // The k here is has scale 1 (k_recip values from 0 to 1).
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["rho0"].get<std::vector<std::vector<PetscReal>>>(),
        j_known["rho0"].get<std::vector<std::vector<PetscReal>>>(),
        100.0*macheps, 10.0*macheps) );
    ASSERT_TRUE( anomtrans::check_equal_within(j_out["norm_d_rho0_dk"].get<std::vector<std::vector<PetscReal>>>(),
        j_known["norm_d_rho0_dk"].get<std::vector<std::vector<PetscReal>>>(),
        100.0*macheps, 10.0*macheps) );
  }
}
