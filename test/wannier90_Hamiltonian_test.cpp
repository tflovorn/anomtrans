#include <cstddef>
#include <limits>
#include <complex>
#include <tuple>
#include <exception>
#include <sstream>
#include <map>
#include <boost/optional.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "util/MPIPrettyUnitTestResultPrinter.h"
#include "util/util.h"
#include "util/lattice.h"
#include "grid_basis.h"
#include "models/wannier90_Hamiltonian.h"

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
TEST( Wannier90_hr, Wannier90_hr ) {
  boost::optional<std::string> test_data_dir = anomtrans::getenv_optional("ANOMTRANS_TEST_DATA_DIR");
  if (not test_data_dir) {
    throw std::runtime_error("Could not get ANOMTRANS_TEST_DATA_DIR environment variable for regression test data");
  }

  std::stringstream hr_path;
  hr_path << *test_data_dir << "/WSe2/wannier/WSe2_hr.dat";

  const std::size_t k_dim = 2;
  anomtrans::kComps<k_dim> Nk = {8, 8};

  auto tb = anomtrans::extract_Wannier90_Hamiltonian(hr_path.str(), Nk);

  std::size_t Nbands_expected = 22;
  std::size_t Nr_expected = 91;

  ASSERT_EQ( tb.kmb.Nbands, Nbands_expected );
  ASSERT_EQ( tb.Hrs.size(), Nr_expected );

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
}
