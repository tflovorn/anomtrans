#include <cstddef>
#include <array>
#include <tuple>
#include <gtest/gtest.h>
#include <mpi.h>
#include <petscksp.h>
#include "util/MPIPrettyUnitTestResultPrinter.h"
#include "grid_basis.h"

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

TEST( GridBasis, GridBasis ) {
  const std::size_t ncomp = 3;
  std::array<unsigned int, ncomp> sizes = {8, 6, 4};
  anomtrans::GridBasis<ncomp> gb(sizes);
  ASSERT_EQ( gb.end_iall, static_cast<PetscInt>(sizes.at(0)*sizes.at(1)*sizes.at(2)) );

  PetscInt iall = 0;
  for (unsigned int i2 = 0; i2 < sizes.at(2); i2++) {
    for (unsigned int i1 = 0; i1 < sizes.at(1); i1++) {
      for (unsigned int i0 = 0; i0 < sizes.at(0); i0++) {
        std::array<unsigned int, ncomp> comps = {i0, i1, i2};
        ASSERT_EQ( gb.decompose(iall), comps );
        ASSERT_EQ( gb.compose(comps), iall );

        for (int p2 = -sizes.at(2); p2 <= static_cast<int>(sizes.at(2)); p2++) {
          for (int p1 = -sizes.at(1); p1 <= static_cast<int>(sizes.at(1)); p1++) {
            for (int p0 = -sizes.at(0); p0 <= static_cast<int>(sizes.at(0)); p0++) {
              std::array<int, ncomp> p = {p0, p1, p2};
              std::array<unsigned int, ncomp> expect = {(i0 + p0) % sizes.at(0),
                  (i1 + p1) % sizes.at(1), (i2 + p2) % sizes.at(2)};
              ASSERT_EQ( gb.decompose(gb.add(gb.compose(comps), p)), expect );
            }
          }
        }

        iall++;
      }
    }
  }
}

TEST( GridBasis, kmBasis ) {
  const std::size_t dim = 2;
  using kComps = anomtrans::kComps<dim>;
  using dkComps = anomtrans::dkComps<dim>;
  using kmComps = anomtrans::kmComps<dim>;
  using kVals = anomtrans::kVals<dim>;
  using kmVals = anomtrans::kmVals<dim>;

  std::array<unsigned int, dim> Nk = {8, 6};
  unsigned int Nbands = 2;
  anomtrans::kmBasis<dim> kmb(Nk, Nbands);
  ASSERT_EQ( kmb.end_ikm, static_cast<PetscInt>(Nk.at(0)*Nk.at(1)*Nbands) );

  PetscInt iall = 0;
  for (unsigned int m = 0; m < Nbands; m++) {
    for (unsigned int ik1 = 0; ik1 < Nk.at(1); ik1++) {
      for (unsigned int ik0 = 0; ik0 < Nk.at(0); ik0++) {
        kComps ik_comps = {ik0, ik1};
        kmComps ikm_comps(ik_comps, m);
        kVals k_at_comps = {ik0/static_cast<double>(Nk.at(0)), ik1/static_cast<double>(Nk.at(1))};
        kmVals km_at_comps(k_at_comps, m);

        ASSERT_EQ( kmb.decompose(iall), ikm_comps );
        ASSERT_EQ( kmb.compose(ikm_comps), iall );
        ASSERT_EQ( anomtrans::km_at(Nk, ikm_comps), km_at_comps );

        for (int p1 = -Nk.at(1); p1 <= static_cast<int>(Nk.at(1)); p1++) {
          for (int p0 = -Nk.at(0); p0 <= static_cast<int>(Nk.at(0)); p0++) {
            dkComps Delta_k = {p0, p1};
            kComps k_expect = {(ik0 + p0) % Nk.at(0), (ik1 + p1) % Nk.at(1)};
            kmComps kmp_expect(k_expect, m);
            ASSERT_EQ( kmb.decompose(kmb.add(kmb.compose(ikm_comps), Delta_k)), kmp_expect );
          }
        }

        iall++;
      }
    }
  }
}
