#ifndef ANOMTRANS_RHO0_H
#define ANOMTRANS_RHO0_H

#include "dist_vec.h"
#include "grid_basis.h"
#include "energy.h"
#include "vector_apply.h"
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
DistVec<double> make_rho0(kmBasis<k_dim> kmb, DistVec<double> energies,
    double beta, double E_F) {
  auto fd = [beta, E_F](double E)->double {
    return fermi_dirac(beta, E - E_F);
  };

  auto rho0 = vector_elem_apply<double>(kmb, energies, fd);
  return rho0;
}

} // namespace anomtrans

#endif // ANOMTRANS_RHO0_H
