#ifndef ANOMTRANS_RHO0_H
#define ANOMTRANS_RHO0_H

#include <petscksp.h>
#include "util/vec.h"
#include "util/special_functions.h"

namespace anomtrans {

/** @brief Construct a vector rho0 of equilibrium occupations (i.e. external E=0, B=0)
 *         using the given k-space discretization and energies.
 *  @param energies Vector of E_m(k) values.
 *  @param beta Inverse temperature 1/(k_B T).
 *  @param mu Chemical potential.
 */
Vec make_rho0(Vec energies, double beta, double mu);

/** @brief Gives the maximum beta value that can be expected to allow an
 *         adequate sampling of the Fermi surface.
 *  @note This is obtained by enforcing the condition that
 *        |f_{FD}(E_{k,m}, beta) - f_{FD}(E_{k+dk_i,m}, beta)| < 1/2
 *        for all km (where dk_i is the minimum step in the i'th reciprocal
 *        lattice coordinate direction); i.e. that it takes at least two
 *        dk steps to go from f_{FD}(E_{km} - mu, beta) = 1 to
 *        f_{FD}(E_{k'm} - mu, beta) = 0. Under this condiction, the 'Fermi
 *        surface' of transition from fully-occupied to unoccupied states
 *        contains at least one k-space unit.
 *        The condition is obtained by expanding f_{FD} to leading order in
 *        E - mu.
 */
double get_beta_max(PetscReal max_energy_difference);

} // namespace anomtrans

#endif // ANOMTRANS_RHO0_H
