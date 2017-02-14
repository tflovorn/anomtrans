#ifndef ANOMTRANS_DRIVING_H
#define ANOMTRANS_DRIVING_H

#include <cstddef>
#include <stdexcept>
#include <petscksp.h>
#include "grid_basis.h"
#include "util.h"

namespace anomtrans {

/** @brief Electric field driving term hbar/e * Dbar_E.
 *  @param D Matrix with elements [D]_{ci} giving the c'th Cartesian component
 *           (in order x, y, z) of the i'th lattice vector.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param order Order of approximation to use for finite difference calculation
 *               of derivative.
 *  @param Ehat Unit vector giving the direction of the electric field in
 *              the Cartesian basis.
 *  @todo Implement correct calculation for multi-band case (superoperator
 *        representation?).
 *  @todo Is it appropriate for E to have the same dimension as k?
 *        Term E dot d<rho>/dk has vanishing contributions from E components
 *        where there are no corresponding k components.
 *        Possibly some interesting situations where E is perpendicular to a
 *        surface, though.
 *  @todo Pass in derivative/energies to avoid calculating calculating them
 *        here?
 */
template <std::size_t k_dim>
Mat driving_electric(DimMatrix<k_dim> D, kmBasis<k_dim> kmb, unsigned int order,
    std::array<double, k_dim> Ehat) {
  // Maximum number of elements expected for sum of Cartesian derivatives.
  PetscInt expected_elems_per_row = order*k_dim*k_dim*k_dim;

  auto d_dk_Cart = make_d_dk_Cartesian(D, kmb, order);
  Mat Dbar_E = Mat_from_sum_const(Ehat, d_dk_Cart, expected_elems_per_row);

  return Dbar_E;
}

/** @brief Magnetic field driving term hbar^2/e * Dbar_B.
 *  @param D Matrix with elements [D]_{ci} giving the c'th Cartesian component
 *           (in order x, y, z) of the i'th lattice vector.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param order Order of approximation to use for finite difference calculation
 *               of derivative.
 *  @param H Class instance giving the Hamiltonian of the system. Should have
 *           the method std::array<double, k_dim> velocity(kmComps<k_dim>) which
 *           gives the group velocity in the Cartesian basis.
 *  @param Bhat Unit vector giving the direction of the magnetic field in
 *              the Cartesian basis.
 *  @todo What is the correct way to handle x, y components of Bfield when k_dim is 2?
 *        Should the number of components of Bfield be fixed differently?
 *  @todo Implement correct calculation for multi-band case (superoperator
 *        representation?). Need to understand calculation of velocity for
 *        general case (multi-band, possible degeneracies).
 *  @todo Pass in derivative/energies to avoid calculating calculating them
 *        here?
 *  @todo Could use constexpr if here.
 */
template <std::size_t k_dim, typename Hamiltonian>
Mat driving_magnetic(DimMatrix<k_dim> D, kmBasis<k_dim> kmb, unsigned int order,
    Hamiltonian H, std::array<double, 3> Bhat) {
  std::vector<std::array<PetscScalar, 3>> coeffs;
  for (PetscInt ikm = 0; ikm < kmb.end_ikm; ikm++) {
    // Need a function Mat_from_sum(coeff_fn, Bs, expected_elems_per_for)
    // where coeff_fn gives a coefficient which is constant along a given row
    // but may vary between rows.
    std::array<PetscScalar, k_dim> v = H.velocity(ikm);
    coeffs.push_back(cross(v, B));
  }

  auto coeff_fn = [coeffs](std::size_t d, PetscInt row, PetscInt col)->PetscScalar {
    return coeffs.at(row).at(d);
  };

  // Maximum number of elements expected for sum of Cartesian derivatives.
  PetscInt expected_elems_per_row = order*k_dim*k_dim*k_dim;

  auto d_dk_Cart = make_d_dk_Cartesian(D, kmb, order);
  Mat Dbar_b = Mat_from_sum_fn(coeff_fn, Bs, expected_elems_per_row);

  return Dbar_b;
}

} // namespace anomtrans

#endif // ANOMTRANS_DRIVING_H
