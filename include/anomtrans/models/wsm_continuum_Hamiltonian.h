#ifndef ANOMTRANS_MODELS_WSM_CONTINUUM_HAMILTONIAN_H
#define ANOMTRANS_MODELS_WSM_CONTINUUM_HAMILTONIAN_H

#include <stdexcept>
#include <array>
#include <cmath>
#include <complex>
#include <petscksp.h>
#include <Eigen/Core>
#include "grid_basis.h"
#include "observables/spin.h"

namespace anomtrans {

/** @brief Hamiltonian for continuum model of a Weyl semimetal.
 */
class WsmContinuumHamiltonian {
  static std::array<Eigen::Matrix4cd, 3> get_spin_matrices();

  double m_t(double kz, int t) const;

  double eps_st(kVals<3> k, int s, int t) const;

  std::complex<double> u_component_upper(kVals<3> k, int s, int t) const;

  std::complex<double> u_component_lower(kVals<3> k, int s, int t) const;

  Eigen::Matrix4cd evecs(kVals<3> k) const;

  std::array<Eigen::Matrix4cd, 3> grad_H(kVals<3> k) const;

  const std::array<Eigen::Matrix4cd, 3> spin_matrices;
public:
  /** @brief Magnetic parameter $b$, in units of $\Delta$ (the Dirac mass parameter).
   */
  const double b;

  /** @brief Discretization of (k, m) basis on which this Hamiltonian is defined.
   *  @todo Would prefer to avoid copying kmb: it may be large.
   *        Replace kmb copy with a shared pointer?
   */
  const kmBasis<3> kmb;

  /** @brief Hamiltonian for continuum model of a Weyl semimetal.
   *  @param _b Magnetic parameter $b$ in units of $\Delta$, the Dirac mass parameter.
   *  @note $k$ values are taken to be unitless: here we use $\tilde{k}$, scaled
   *        as follows: $\tilde{k} = \frac{\hbar v_F}{\Delta} k$.
   *  @note All energies are given in units of $\Delta$.
   */
  WsmContinuumHamiltonian(double _b, kmBasis<3> _kmb);

  /** @brief Energy at (k,m): E_{km}.
   */
  double energy(kmComps<3> ikm_comps) const;

  /** @brief Value of U_{im}(k), where U is the unitary matrix which diagonalizes
   *         H(k), m is the eigenvalue index, and i is the component of the
   *         initial basis (pseudo-atomic orbital or otherwise).
   */
  std::complex<double> basis_component(PetscInt ikm, unsigned int i) const;

  /** @brief Gradient of the Hamiltonian, evaluated in the eigenbasis;
   *         equal to the covariant derivative of the Hamiltonain.
   *         gradient(ikm, mp) = <k, m|grad_k H|k, mp>.
   */
  std::array<std::complex<double>, 3> gradient(kmComps<3> ikm_comps, unsigned int mp) const;

  /** @brief Spin, evaluated in the eigenbasis (units of hbar):
   *         spin(ikm, mp)[a] = <km|S_a|km'>
   */
  std::array<std::complex<double>, 3> spin(PetscInt ikm, unsigned int mp) const;
};

} // namespace anomtrans

#endif // ANOMTRANS_MODELS_WSM_CONTINUUM_HAMILTONIAN_H
