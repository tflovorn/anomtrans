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
#include "MPIPrettyUnitTestResultPrinter.h"
#include "constants.h"
#include "util.h"
#include "grid_basis.h"
#include "square_tb_spectrum.h"
#include "energy.h"
#include "vec.h"
#include "rho0.h"
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

TEST( Derivative, square_TB_fermi_surface ) {
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

  const unsigned int deriv_order = 2;
  auto d_dk = anomtrans::make_d_dk_recip(kmb, deriv_order);
  auto d_dk_Cart = anomtrans::make_d_dk_Cartesian(D, kmb, deriv_order);

  // Since a1 = \hat{x}, a2 = \hat{y}, we should have d_dk == 2*pi*d_dk_Cart.
  for (std::size_t d = 0; d < k_dim; d++) {
    Mat d_dk_Cart_d_2pi;
    MatDuplicate(d_dk_Cart.at(d), MAT_COPY_VALUES, &d_dk_Cart_d_2pi);
    MatScale(d_dk_Cart_d_2pi, 2*anomtrans::pi);

    // Tried to use PETSc MatEqual() for this, got confusing errors about unequal
    // matrix dimensions. Checking with MatGetSize() and MatGetOwnershipRange()
    // showed that d_dk.at(d) and d_dk_Cart.at(d) had equal sizes and local row
    // distributions. Not sure what the source of the error was.
    double tol = 2*std::numeric_limits<double>::epsilon();
    ASSERT_TRUE( anomtrans::check_Mat_equal(d_dk.at(d), d_dk_Cart_d_2pi, tol) );

    ierr = MatDestroy(&d_dk_Cart_d_2pi);CHKERRXX(ierr);
  }

  unsigned int num_mus = 40;
  auto mus = anomtrans::linspace(Ekm_min, Ekm_max, num_mus);

  std::vector<std::vector<PetscScalar>> all_rho0;
  std::vector<std::vector<PetscScalar>> all_norm_d_rho0_dk;
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
        double component = std::get<1>(local_d_rho0_dk.at(d)).at(i);
        norm2 += component*component;
      }
      double norm = std::sqrt(norm2);
      norm_vals.push_back(norm);
    }

    ierr = VecSetValues(norm_d_rho0_dk, local_rows.size(), local_rows.data(),
        norm_vals.data(), INSERT_VALUES);CHKERRXX(ierr);

    ierr = VecAssemblyBegin(norm_d_rho0_dk);CHKERRXX(ierr);
    ierr = VecAssemblyEnd(norm_d_rho0_dk);CHKERRXX(ierr);

    auto collected_rho0 = anomtrans::collect_contents(rho0_km);
    auto collected_norm_d_rho0_dk = anomtrans::collect_contents(norm_d_rho0_dk);

    all_rho0.push_back(collected_rho0);
    all_norm_d_rho0_dk.push_back(collected_norm_d_rho0_dk);

    ierr = VecDestroy(&norm_d_rho0_dk);CHKERRXX(ierr);
    ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);
    for (std::size_t d = 0; d < k_dim; d++) {
      ierr = VecDestroy(&(d_rho0_dk.at(d)));CHKERRXX(ierr);
    }
  }

  auto collected_Ekm = anomtrans::collect_contents(Ekm);

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
    std::stringstream known_data;
    known_data << *test_data_dir << "/derivative_test_out.json";

    ASSERT_TRUE( anomtrans::check_json_equal(outpath.str(), known_data.str()) );
  }
}
