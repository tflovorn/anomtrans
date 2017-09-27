#ifndef ANOMTRANS_RASHBA_HAMILTONIAN_H
#define ANOMTRANS_RASHBA_HAMILTONIAN_H

#include <cassert>
#include <cstddef>
#include <cmath>
#include <tuple>
#include <complex>
#include <petscksp.h>
#include "constants.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief k-space Hamiltonian for square lattice (or higher-dimensional version of square
 *         lattice) tight-binding model with Rashba coupling.
 *  @note The lattice spacing a is set to 1.
 */
class Rashba_Hamiltonian {
public:
  /** @brief Nearest-neighbor orbital-preserving hopping amplitude.
   */
  const double t0;

  /** @brief Nearest-neighbor Rashba hopping amplitude.
   */
  const double tr;

  /** @brief Discretization of (k, m) basis on which this Hamiltonian is defined.
   *  @todo Would prefer to avoid copying kmb: it may be large.
   *        Replace kmb copy with a shared pointer?
   */
  const kmBasis<2> kmb;

  Rashba_Hamiltonian(double _t0, double _tr, kmBasis<2> _kmb);

  /** @brief Energy at (k,m): E_{km}.
   */
  double energy(kmComps<2> ikm_comps) const;

  /** @brief Gradient of the Hamiltonian, evaluated in the eigenbasis;
   *         equal to the covariant derivative of the Hamiltonain.
   *         gradient(ikm, mp) = <k, m|grad_k H|k, mp>.
   */
  std::array<std::complex<double>, 2> gradient(kmComps<2> ikm_comps, unsigned int mp) const;

  /** @brief Velocity at (k, m): v_{km} = dE_{km}/dk|_{k}.
   *  @note By Hellmann-Feynman theorem, this is equal to gradient(km, m)
   *         = <k, m|grad_k H|k, m>.
   */
  std::array<double, 2> velocity(kmComps<2> ikm_comps) const;

  /** @brief Value of U_{im}(k), where U is the unitary matrix which diagonalizes
   *         H(k), m is the eigenvalue index, and i is the component of the
   *         initial basis (pseudo-atomic orbital or otherwise).
   */
  std::complex<double> basis_component(PetscInt ikm, unsigned int i) const;
};

} // namespace anomtrans

#endif // ANOMTRANS_RASHBA_HAMILTONIAN_H
