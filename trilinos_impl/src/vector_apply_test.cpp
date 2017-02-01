#include <functional>
#include <Teuchos_UnitTestHarness.hpp>
#include "mpi.h"
#include "dist_vec.h"
#include "grid_basis.h"
#include "square_tb_spectrum.h"
#include "energy.h"
#include "special_functions.h"
#include "vector_apply.h"

namespace {

TEUCHOS_UNIT_TEST( Vector_Apply, Square_TB_Energy_Fermi ) {
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
  double beta = 100.0;

  auto fd = [beta](double E)->double {
    return anomtrans::fermi_dirac(beta, E);
  };

  anomtrans::DistVec<double> Ekm = anomtrans::get_energies(kmb, H);
  auto rho0_km = anomtrans::vector_elem_apply<double>(kmb, Ekm, fd);

  Ekm.sync<Kokkos::HostSpace>();
  auto rho0_km_2d = rho0_km.getLocalView<Kokkos::HostSpace>();
  auto rho0_km_1d = Kokkos::subview(rho0_km_2d, Kokkos::ALL(), 0);

  const LO num_local_elements = static_cast<LO>(kmb_map->getNodeNumElements());

  for (LO local_row = 0; local_row < num_local_elements; local_row++) {
    const GO global_row = kmb_map->getGlobalElement(local_row);
    auto ikm_comps = kmb.decompose(global_row);
    double energy = H.energy(ikm_comps);
    TEST_ASSERT( rho0_km_1d(local_row) == fd(energy) );
  }
}

} // namespace
