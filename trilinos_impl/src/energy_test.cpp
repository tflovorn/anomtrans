#include <Teuchos_UnitTestHarness.hpp>
#include "mpi.h"
#include "dist_vec.h"
#include "grid_basis.h"
#include "square_tb_spectrum.h"
#include "energy.h"

namespace {

TEUCHOS_UNIT_TEST( Energy, Square_TB_Energy ) {
  using GO = anomtrans::GO;
  using LO = anomtrans::LO;

  anomtrans::MPIComm comm = anomtrans::get_comm();
  const std::size_t dim = 2;

  std::array<unsigned int, dim> Nk = {8, 8};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<dim> kmb(Nk, Nbands, comm);
  auto kmb_map = kmb.get_map();

  double t = 1.0;
  double tp = -0.3;
  anomtrans::square_tb_Hamiltonian H(t, tp, Nk);

  anomtrans::DistVec<double> Ekm = anomtrans::get_energies(kmb, H);

  Ekm.sync<Kokkos::HostSpace>();
  auto Ekm_2d = Ekm.getLocalView<Kokkos::HostSpace>();
  auto Ekm_1d = Kokkos::subview(Ekm_2d, Kokkos::ALL(), 0);

  const LO num_local_elements = static_cast<LO>(kmb_map->getNodeNumElements());

  for (LO local_row = 0; local_row < num_local_elements; local_row++) {
    const GO global_row = kmb_map->getGlobalElement(local_row);
    auto ikm_comps = kmb.decompose(global_row);
    double energy = H.energy(ikm_comps);
    TEST_ASSERT( Ekm_1d(local_row) == energy );
  }
}

} // namespace
