#ifndef ANOMTRANS_MODELS_TMD_NN_HAMILTONIAN_H
#define ANOMTRANS_MODELS_TMD_NN_HAMILTONIAN_H

#include <cstddef>
#include <complex>
#include <map>
#include <Eigen/Core>
#include "util/lattice.h"
#include "models/wannier90_Hamiltonian.h"

namespace anomtrans {

namespace internal {

/** @brief Given a matrix H in the {z^2, xy, x^2 - y^2} basis, return the corresponding
 *         matrix expanded to include SOC:
 *         H -> [H + lambda/2 L_z, 0]
 *              [0, H - lambda/2 L_z]
 */
Eigen::MatrixXcd expand_soc(Eigen::MatrixXcd H, double lambda);

} // namespace internal

/** @brief Parameters describing the nearest-neighbor TMD tight-binding model
 *         of Liu et al., PRB 88, 085433 (2013).
 */
struct TMD_NN_Params {
  /** @brief Lattice constant.
   */
  double a;
  /* @brief On-site energies.
   */
  double epsilon1, epsilon2;
  /** @brief Nearest-neighbor hopping parameters for R = R1.
   */
  double t0, t1, t2, t11, t12, t22;
  /** @brief Spin-orbit coupling strength.
   */
  double lambda;
};

/** @brief Parameters for WSe2 (GGA) from Table II and IV of Liu et al., PRB 88, 085433 (2013).
 */
TMD_NN_Params get_TMD_NN_Params_WSe2();

/** @brief Given an appropriate set of parameters, create a `TBHamiltonian` which
 *         encodes the TMD tight-binding model of Liu et al., PRB 88, 085433 (2013).
 */
TBHamiltonian<2> make_TMD_NN_Hamiltonian(TMD_NN_Params p, kComps<2> _Nk);

} // namespace anomtrans

#endif // ANOMTRANS_MODELS_TMD_NN_HAMILTONIAN_H
