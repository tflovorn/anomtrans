#ifndef ANOMTRANS_RHO0_H
#define ANOMTRANS_RHO0_H

#include <petscksp.h>
#include "grid_basis.h"
#include "vec.h"
#include "energy.h"
#include "special_functions.h"

namespace anomtrans {

/** @brief Construct a vector rho0 of equilibrium occupations (i.e. external E=0, B=0)
 *         using the given k-space discretization and energies.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param energies Vector of E_m(k) values.
 *  @param beta Inverse temperature 1/(k_B T).
 *  @param E_F Fermi energy.
 */
template <std::size_t k_dim>
Vec make_rho0(kmBasis<k_dim> kmb, Vec energies,
    double beta, double E_F) {
  auto fd = [beta, E_F](double E)->double {
    return fermi_dirac(beta, E - E_F);
  };

  Vec rho0 = vector_elem_apply(kmb, energies, fd);
  return rho0;
}

} // namespace anomtrans

#endif // ANOMTRANS_RHO0_H
