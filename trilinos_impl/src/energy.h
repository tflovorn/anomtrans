#ifndef ANOMTRANS_ENERGY_H
#define ANOMTRANS_ENERGY_H

#include <cstddef>
#include <Tpetra_DefaultPlatform.hpp>
#include "dist_vec.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief Construct a vector of energies using the given k-space discretization
 *         and Hamiltonian.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class giving the Hamiltonian of the system. Should have the method
 *           energy(kmComps<dim>).
 */
template <std::size_t k_dim, typename Hamiltonian>
DistVec<double> get_energies(kmBasis<k_dim> kmb, Hamiltonian H) {
  auto kmb_map = kmb.get_map();
  DistVec<double> Ekm(kmb_map);

  Ekm.sync<Kokkos::HostSpace>();
  auto Ekm_2d = Ekm.getLocalView<Kokkos::HostSpace>();
  auto Ekm_1d = Kokkos::subview(Ekm_2d, Kokkos::ALL(), 0);
  Ekm.modify<Kokkos::HostSpace>();

  const LO num_local_elements = static_cast<LO>(kmb_map->getNodeNumElements());

  for (LO local_row = 0; local_row < num_local_elements; local_row++) {
    const GO global_row = kmb_map->getGlobalElement(local_row);
    auto ikm_comps = kmb.decompose(global_row);
    double energy = H.energy(ikm_comps);
    Ekm_1d(local_row) = energy;
  }

  Ekm.sync<DistVecMemorySpace<double>>();

  return Ekm;
}

} // namespace anomtrans

#endif // ANOMTRANS_ENERGY_H
