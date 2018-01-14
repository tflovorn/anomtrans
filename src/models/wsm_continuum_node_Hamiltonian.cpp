#include "models/wsm_continuum_node_Hamiltonian.h"

namespace anomtrans {

WsmContinuumNodeHamiltonian::WsmContinuumNodeHamiltonian(int _nu, kmBasis<3> _kmb)
    : spin_matrices(get_spin_matrices()), nu(_nu), kmb(_kmb) {
  if (kmb.Nbands != 2) {
    throw std::invalid_argument("must supply kmb with 2 bands to WsmContinuumNodeHamiltonian");
  }

  if (nu != -1 and nu != 1) {
    throw std::invalid_argument("must have nu = +/- 1 for Weyl node");
  }
}

Eigen::Matrix2cd WsmContinuumNodeHamiltonian::H(kVals<3> k) const {
  auto pauli = pauli_matrices();

  Eigen::Matrix2cd H = k.at(0) * pauli.at(0) + k.at(1) * pauli.at(1) + nu * k.at(2) * pauli.at(2);

  return H;
}

std::array<Eigen::Matrix2cd, 3> WsmContinuumNodeHamiltonian::grad_H(kVals<3> k) const {
  auto pauli = pauli_matrices();

  Eigen::Matrix2cd dH_dkx = pauli.at(0);
  Eigen::Matrix2cd dH_dky = pauli.at(1);
  Eigen::Matrix2cd dH_dkz = nu * pauli.at(2);

  return {dH_dkx, dH_dky, dH_dkz};
}

double WsmContinuumNodeHamiltonian::energy(kmComps<3> ikm_comps) const {
  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1) {
    throw std::invalid_argument("WsmContinuumNodeHamiltonian is not defined unless Nbands == 2");
  }

  auto Hk = H(k);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> eigensolver(Hk);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("eigensolver failed");
  }

  return eigensolver.eigenvalues()(m);
}

Eigen::Matrix2cd WsmContinuumNodeHamiltonian::evecs(kVals<3> k) const {
  auto Hk = H(k);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> eigensolver(Hk);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("eigensolver failed");
  }

  return eigensolver.eigenvectors();
}

std::complex<double> WsmContinuumNodeHamiltonian::basis_component(PetscInt ikm, unsigned int i) const {
  auto ikm_comps = kmb.decompose(ikm);

  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or i > 1) {
    throw std::invalid_argument("WsmContinuumNodeHamiltonian is not defined unless Nbands == 2");
  }

  auto U = evecs(k);
  return U(i, m);
}

std::array<std::complex<double>, 3> WsmContinuumNodeHamiltonian::gradient(kmComps<3> ikm_comps,
    unsigned int mp) const {
  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("WsmContinuumNodeHamiltonian is not defined unless Nbands == 2");
  }

  auto U = evecs(k);
  auto grad = grad_H(k);

  return {(U.adjoint() * grad.at(0) * U)(m, mp),
      (U.adjoint() * grad.at(1) * U)(m, mp),
      (U.adjoint() * grad.at(2) * U)(m, mp)};
}

/** @brief Spin matrices for S = 1/2 given in units of hbar.
 */
std::array<Eigen::Matrix2cd, 3> WsmContinuumNodeHamiltonian::get_spin_matrices() {
  auto pauli = pauli_matrices();
  std::array<Eigen::Matrix2cd, 3> result;

  for (std::size_t d = 0; d < 3; d++) {
    result.at(d) = 0.5 * pauli.at(d);
  }

  return result;
}

std::array<std::complex<double>, 3> WsmContinuumNodeHamiltonian::spin(PetscInt ikm, unsigned int mp) const {
  auto km = kmb.decompose(ikm);
  unsigned int m = std::get<1>(km);
  auto k_vals = std::get<0>(kmb.km_at(km));

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("WsmContinuumNodeHamiltonian is not defined unless Nbands == 2");
  }

  auto U = evecs(k_vals);

  return {(U.adjoint() * spin_matrices.at(0) * U)(m, mp),
      (U.adjoint() * spin_matrices.at(1) * U)(m, mp),
      (U.adjoint() * spin_matrices.at(2) * U)(m, mp)};
}

} // namespace anomtrans
