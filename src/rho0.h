#ifndef ANOMTRANS_RHO0_H
#define ANOMTRANS_RHO0_H

#include <petscksp.h>
#include "vec.h"
#include "special_functions.h"

namespace anomtrans {

/** @brief Construct a vector rho0 of equilibrium occupations (i.e. external E=0, B=0)
 *         using the given k-space discretization and energies.
 *  @param energies Vector of E_m(k) values.
 *  @param beta Inverse temperature 1/(k_B T).
 *  @param mu Chemical potential.
 */
Vec make_rho0(Vec energies, double beta, double mu);

} // namespace anomtrans

#endif // ANOMTRANS_RHO0_H
