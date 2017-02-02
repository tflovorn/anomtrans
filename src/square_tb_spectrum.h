#ifndef ANOMTRANS_SQUARE_TB_SPECTRUM_H
#define ANOMTRANS_SQUARE_TB_SPECTRUM_H

#include <cstddef>
#include <cmath>
#include <tuple>
#include <complex>
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

  square_tb_Hamiltonian(double _t, double _tp, kComps<2> _Nk);

  /** @brief Energy at (k,m).
   */
  double energy(kmComps<2> ikm_comps);

  /** @brief Value of U_{im}(k), where U is the unitary matrix which diagonalizes
   *         H(k), m is the eigenvalue index, and i is the component of the
   *         initial basis (pseudo-atomic orbital or otherwise).
   */
  std::complex<double> basis_component(kmComps<2> ikm_comps, unsigned int i);
};

} // namespace anomtrans

#endif // ANOMTRANS_SQUARE_TB_SPECTRUM_H
