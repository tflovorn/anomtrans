#include <cstddef>
#include <array>
#include <tuple>
#include <Teuchos_UnitTestHarness.hpp>
#include "mpi.h"
#include "dist_vec.h"
#include "grid_basis.h"

namespace {

TEUCHOS_UNIT_TEST( GridBasis, GridBasis ) {
  anomtrans::MPIComm comm = anomtrans::get_comm();

  const std::size_t ncomp = 3;
  std::array<unsigned int, ncomp> sizes = {4, 4, 4};
  anomtrans::GridBasis<ncomp> gb(sizes, comm);
  TEST_ASSERT( gb.end_iall == 4*4*4 );

  anomtrans::GO iall = 0;
  for (unsigned int i2 = 0; i2 < sizes.at(2); i2++) {
    for (unsigned int i1 = 0; i1 < sizes.at(1); i1++) {
      for (unsigned int i0 = 0; i0 < sizes.at(0); i0++) {
        std::array<unsigned int, ncomp> comps = {i0, i1, i2};
        TEST_ASSERT( gb.decompose(iall) == comps );
        TEST_ASSERT( gb.compose(comps) == iall );

        for (int p2 = -sizes.at(2); p2 <= sizes.at(2); p2++) {
          for (int p1 = -sizes.at(1); p1 <= sizes.at(1); p1++) {
            for (int p0 = -sizes.at(0); p0 <= sizes.at(0); p0++) {
              std::array<int, ncomp> p = {p0, p1, p2};
              std::array<unsigned int, ncomp> expect = {(i0 + p0) % sizes.at(0),
                  (i1 + p1) % sizes.at(1), (i2 + p2) % sizes.at(2)};
              TEST_ASSERT( gb.decompose(gb.add(gb.compose(comps), p)) == expect );
            }
          }
        }

        iall++;
      }
    }
  }
}

TEUCHOS_UNIT_TEST( GridBasis, kmBasis ) {
  anomtrans::MPIComm comm = anomtrans::get_comm();

  const std::size_t dim = 2;
  using kComps = anomtrans::kComps<dim>;
  using dkComps = anomtrans::dkComps<dim>;
  using kmComps = anomtrans::kmComps<dim>;
  using kVals = anomtrans::kVals<dim>;
  using kmVals = anomtrans::kmVals<dim>;

  std::array<unsigned int, dim> Nk = {2, 2};
  unsigned int Nbands = 2;
  anomtrans::kmBasis<dim> kmb(Nk, Nbands, comm);
  TEST_ASSERT( kmb.end_ikm == 2*2*2 );

  anomtrans::GO iall = 0;
  for (unsigned int m = 0; m < Nbands; m++) {
    for (unsigned int ik1 = 0; ik1 < Nk.at(1); ik1++) {
      for (unsigned int ik0 = 0; ik0 < Nk.at(0); ik0++) {
        kComps ik_comps = {ik0, ik1};
        kmComps ikm_comps(ik_comps, m);
        kVals k_at_comps = {ik0/static_cast<double>(Nk.at(0)), ik1/static_cast<double>(Nk.at(1))};
        kmVals km_at_comps(k_at_comps, m);

        TEST_ASSERT( kmb.decompose(iall) == ikm_comps );
        TEST_ASSERT( kmb.compose(ikm_comps) == iall );
        TEST_ASSERT( anomtrans::km_at(Nk, ikm_comps) == km_at_comps );

        for (int p1 = -Nk.at(1); p1 <= Nk.at(1); p1++) {
          for (int p0 = -Nk.at(0); p0 <= Nk.at(0); p0++) {
            dkComps Delta_k = {p0, p1};
            kComps k_expect = {(ik0 + p0) % Nk.at(0), (ik1 + p1) % Nk.at(1)};
            kmComps kmp_expect(k_expect, m);
            TEST_ASSERT( kmb.decompose(kmb.add(kmb.compose(ikm_comps), Delta_k)) == kmp_expect );
          }
        }

        iall++;
      }
    }
  }
}

} // namespace
