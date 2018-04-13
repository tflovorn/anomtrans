#include "models/wsm_continuum_mu5_Hamiltonian.h"

namespace anomtrans {

WsmContinuumMu5Hamiltonian::WsmContinuumMu5Hamiltonian(double _b, double _mu5, kmBasis<3> _kmb)
    : sigma_matrices(get_sigma_matrices()), tau_matrices(get_tau_matrices()),
      b(_b), mu5(_mu5), kmb(_kmb) {
  if (kmb.Nbands != 4) {
    throw std::invalid_argument("must supply kmb with 4 bands to WsmContinuumMu5Hamiltonian");
  }

  if (std::abs(_b) <= 1.0) {
    throw std::invalid_argument("must have |b/Delta| > 1 for Weyl semimetal");    
  }
}

std::array<Eigen::Matrix4cd, 3> WsmContinuumMu5Hamiltonian::get_sigma_matrices() {
  auto pauli = pauli_matrices();
  std::array<Eigen::Matrix4cd, 3> result;

  for (std::size_t d = 0; d < 3; d++) {
    result.at(d) = Eigen::Matrix4cd::Zero();
    result.at(d).block(0, 0, 2, 2) = pauli.at(d);
    result.at(d).block(2, 2, 2, 2) = pauli.at(d);
  }

  return result;
}

std::array<Eigen::Matrix4cd, 3> WsmContinuumMu5Hamiltonian::get_tau_matrices() {
  auto pauli = pauli_matrices();
  Eigen::Matrix2cd id = Eigen::Matrix2cd::Identity();
  std::array<Eigen::Matrix4cd, 3> result;

  for (std::size_t d = 0; d < 3; d++) {
    result.at(d) = Eigen::Matrix4cd::Zero();
    result.at(d).block(0, 0, 2, 2) = pauli.at(d)(0, 0) * id;
    result.at(d).block(0, 2, 2, 2) = pauli.at(d)(0, 1) * id;
    result.at(d).block(2, 0, 2, 2) = pauli.at(d)(1, 0) * id;
    result.at(d).block(2, 2, 2, 2) = pauli.at(d)(1, 1) * id;
  }

  return result;
}

Eigen::Matrix4cd WsmContinuumMu5Hamiltonian::H(kVals<3> k) const {
  Eigen::Matrix4cd H = tau_matrices.at(2) * (k.at(0) * sigma_matrices.at(0)
      + k.at(1) * sigma_matrices.at(1) + k.at(2) * sigma_matrices.at(2))
      + tau_matrices.at(0) + b * sigma_matrices.at(2) - mu5 * tau_matrices.at(2);

  return H;
}

std::array<Eigen::Matrix4cd, 3> WsmContinuumMu5Hamiltonian::grad_H(kVals<3> k) const {
  Eigen::Matrix4cd dH_dkx = tau_matrices.at(2) * sigma_matrices.at(0);
  Eigen::Matrix4cd dH_dky = tau_matrices.at(2) * sigma_matrices.at(1);
  Eigen::Matrix4cd dH_dkz = tau_matrices.at(2) * sigma_matrices.at(2);

  return {dH_dkx, dH_dky, dH_dkz};
}

double WsmContinuumMu5Hamiltonian::energy(kmComps<3> ikm_comps) const {
  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 3) {
    throw std::invalid_argument("WsmContinuumMu5Hamiltonian is not defined unless Nbands == 4");
  }

  auto Hk = H(k);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cd> eigensolver(Hk);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("eigensolver failed");
  }

  return eigensolver.eigenvalues()(m);
}

Eigen::Matrix4cd WsmContinuumMu5Hamiltonian::evecs(kVals<3> k) const {
  auto Hk = H(k);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cd> eigensolver(Hk);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("eigensolver failed");
  }

  return eigensolver.eigenvectors();
}

std::complex<double> WsmContinuumMu5Hamiltonian::basis_component(PetscInt ikm, unsigned int i) const {
  auto ikm_comps = kmb.decompose(ikm);

  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 3 or i > 3) {
    throw std::invalid_argument("WsmContinuumMu5Hamiltonian is not defined unless Nbands == 4");
  }

  auto U = evecs(k);
  return U(i, m);
}

std::array<std::complex<double>, 3> WsmContinuumMu5Hamiltonian::gradient(kmComps<3> ikm_comps,
    unsigned int mp) const {
  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 3 or mp > 3) {
    throw std::invalid_argument("WsmContinuumMu5Hamiltonian is not defined unless Nbands == 4");
  }

  auto U = evecs(k);
  auto grad = grad_H(k);

  return {(U.adjoint() * grad.at(0) * U)(m, mp),
      (U.adjoint() * grad.at(1) * U)(m, mp),
      (U.adjoint() * grad.at(2) * U)(m, mp)};
}

std::array<std::complex<double>, 3> WsmContinuumMu5Hamiltonian::spin(PetscInt ikm, unsigned int mp) const {
  auto km = kmb.decompose(ikm);
  unsigned int m = std::get<1>(km);
  auto k_vals = std::get<0>(kmb.km_at(km));

  if (m > 3 or mp > 3) {
    throw std::invalid_argument("WsmContinuumMu5Hamiltonian is not defined unless Nbands == 4");
  }

  auto U = evecs(k_vals);

  return {(U.adjoint() * (0.5 * sigma_matrices.at(0)) * U)(m, mp),
      (U.adjoint() * (0.5 * sigma_matrices.at(1)) * U)(m, mp),
      (U.adjoint() * (0.5 * sigma_matrices.at(2)) * U)(m, mp)};
}

} // namespace anomtrans
