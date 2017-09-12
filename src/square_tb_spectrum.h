#ifndef ANOMTRANS_SQUARE_TB_SPECTRUM_H
#define ANOMTRANS_SQUARE_TB_SPECTRUM_H

#include <cstddef>
#include <cmath>
#include <tuple>
#include <complex>
#include <petscksp.h>
#include "constants.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief k-space Hamiltonian for square lattice (or higher-dimensional version of square
 *         lattice) tight-binding model.
 *  @note For Hamiltonian initialized from TB model by projection (i.e. without
 *        analytic k-> energy formula) can store diagonalized H(k) privately
 *        (for all k).
 */
class square_tb_Hamiltonian {
public:
  /** @brief Nearest-neighbor hopping amplitude.
   */
  const double t;
  /** @brief Next-nearest-neighbor hopping amplitude.
   */
  const double tp;
  /** @brief Number of k-points in each direction to sample.
   */
  const kComps<2> Nk;
  /** @brief Number of bands.
   */
  const unsigned int Nbands;

  square_tb_Hamiltonian(double _t, double _tp, kComps<2> _Nk);

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

#endif // ANOMTRANS_SQUARE_TB_SPECTRUM_H
