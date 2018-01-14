#include "models/wsm_continuum_Hamiltonian.h"

namespace anomtrans {

WsmContinuumHamiltonian::WsmContinuumHamiltonian(double _b, kmBasis<3> _kmb)
    : spin_matrices(get_spin_matrices()), b(_b), kmb(_kmb) {
  if (kmb.Nbands != 4) {
    throw std::invalid_argument("must supply kmb with 4 bands to WsmContinuumHamiltonian");
  }

  if (std::abs(_b) <= 1.0) {
    throw std::invalid_argument("must have |b/Delta| > 1 for Weyl semimetal");    
  }
}

namespace {

int get_s(unsigned int m) {
  if (m == 0 or m == 2) {
    return 1;
  } else {
    return -1;
  }
}

int get_t(unsigned int m) {
  if (m == 0 or m == 1) {
    return 1;
  } else {
    return -1;
  }
}

}

double WsmContinuumHamiltonian::m_t(double kz, int t) const {
  return b + t * std::sqrt(std::pow(kz, 2.0) + 1.0);
}

double WsmContinuumHamiltonian::eps_st(kVals<3> k, int s, int t) const {
  return s * std::sqrt(std::pow(k.at(0), 2.0) + std::pow(k.at(1), 2.0)
      + std::pow(m_t(k.at(2), t), 2.0));
}

double WsmContinuumHamiltonian::energy(kmComps<3> ikm_comps) const {
  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 3) {
    throw std::invalid_argument("WsmContinuumHamiltonian is not defined unless Nbands == 4");
  }

  int s = get_s(m);
  int t = get_t(m);

  return eps_st(k, s, t);
}

namespace {

std::complex<double> get_phase(kVals<3> k) {
  double eps = 1e-9;

  if (std::abs(k.at(0)) < eps and std::abs(k.at(1)) < eps) {
    // Special case: here theta is not well defined.
    // Choose e^{i theta} = 1 in this case (approaching along ky = 0).
    return 1.0;
  } else {
    double k_perp = std::sqrt(std::pow(k.at(0), 2.0) + std::pow(k.at(1), 2.0));
    return std::complex<double>{k.at(0) / k_perp, k.at(1) / k_perp};
  }
}

}

std::complex<double> WsmContinuumHamiltonian::u_component_upper(kVals<3> k, int s, int t) const {
  double k0 = std::sqrt(std::pow(b, 2.0) - 1.0);
  double eps = 1e-9;

  if (t == -1 and std::abs(k.at(0)) < eps and std::abs(k.at(1)) < eps
      and (std::abs(k.at(2) - k0) < eps or std::abs(k.at(2) + k0) < eps)) {
    // At Weyl points, H_- = 0. Choose the eigenvectors
    // |u^s_k> = (1/sqrt(2)) * [1, s]
    // in this case, obtained by approaching the Weyl nodes along a path
    // with kz = +/- k0 but kx != 0 or ky != 0.
    return 1.0 / std::sqrt(2.0);
  }

  return std::sqrt(1.0 + (s * m_t(k.at(2), t) / eps_st(k, 1, t))) / std::sqrt(2.0);
}

std::complex<double> WsmContinuumHamiltonian::u_component_lower(kVals<3> k, int s, int t) const {
  double k0 = std::sqrt(std::pow(b, 2.0) - 1.0);
  double eps = 1e-9;

  if (t == -1 and std::abs(k.at(0)) < eps and std::abs(k.at(1)) < eps
      and (std::abs(k.at(2) - k0) < eps or std::abs(k.at(2) + k0) < eps)) {
    return static_cast<double>(s) / std::sqrt(2.0);
  }

  std::complex<double> phase = get_phase(k);

  return static_cast<double>(s) * phase * std::sqrt(1.0 - (s * m_t(k.at(2), t) / eps_st(k, 1, t))) / std::sqrt(2.0);
}

Eigen::Matrix4cd WsmContinuumHamiltonian::evecs(kVals<3> k) const {
  Eigen::Matrix4cd U = Eigen::Matrix4cd::Zero();

  U(0, 0) = u_component_upper(k, 1, 1);
  U(1, 0) = u_component_lower(k, 1, 1);

  U(0, 1) = u_component_upper(k, -1, 1);
  U(1, 1) = u_component_lower(k, -1, 1);

  U(2, 2) = u_component_upper(k, 1, -1);
  U(3, 2) = u_component_lower(k, 1, -1);

  U(2, 3) = u_component_upper(k, -1, -1);
  U(3, 3) = u_component_lower(k, -1, -1);

  return U;
}

std::complex<double> WsmContinuumHamiltonian::basis_component(PetscInt ikm, unsigned int i) const {
  auto ikm_comps = kmb.decompose(ikm);

  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 3 or i > 3) {
    throw std::invalid_argument("WsmContinuumHamiltonian is not defined unless Nbands == 4");
  }

  auto U = evecs(k);
  return U(i, m);
}

std::array<Eigen::Matrix4cd, 3> WsmContinuumHamiltonian::grad_H(kVals<3> k) const {
  auto pauli = pauli_matrices();

  Eigen::Matrix4cd dH_dkx = Eigen::Matrix4cd::Zero();
  Eigen::Matrix4cd dH_dky = Eigen::Matrix4cd::Zero();
  Eigen::Matrix4cd dH_dkz = Eigen::Matrix4cd::Zero();

  // dHt_dkx = sigma_x
  dH_dkx.block(0, 0, 2, 2) = pauli.at(0);
  dH_dkx.block(2, 2, 2, 2) = pauli.at(0);
  // dHt_dky = sigma_y
  dH_dky.block(0, 0, 2, 2) = pauli.at(1);
  dH_dky.block(2, 2, 2, 2) = pauli.at(1);
  // dHt_dkz = dm_t/dkz * sigma_z
  dH_dkz.block(0, 0, 2, 2) = (k.at(2) / std::sqrt(std::pow(k.at(2), 2.0) + 1.0)) * pauli.at(2);
  dH_dkz.block(2, 2, 2, 2) = (-k.at(2) / std::sqrt(std::pow(k.at(2), 2.0) + 1.0)) * pauli.at(2);

  return {dH_dkx, dH_dky, dH_dkz};
}

std::array<std::complex<double>, 3> WsmContinuumHamiltonian::gradient(kmComps<3> ikm_comps,
    unsigned int mp) const {
  kmVals<3> km = kmb.km_at(ikm_comps);
  kVals<3> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 3 or mp > 3) {
    throw std::invalid_argument("WsmContinuumHamiltonian is not defined unless Nbands == 4");
  }

  auto U = evecs(k);
  auto grad = grad_H(k);

  return {(U.adjoint() * grad.at(0) * U)(m, mp),
      (U.adjoint() * grad.at(1) * U)(m, mp),
      (U.adjoint() * grad.at(2) * U)(m, mp)};
}

/** @brief Spin matrices for S = 1/2 in basis {(t+, s+), (t+, s-), (t-, s+), (t-, s-)},
 *         given in units of hbar.
 */
std::array<Eigen::Matrix4cd, 3> WsmContinuumHamiltonian::get_spin_matrices() {
  auto pauli = pauli_matrices();
  std::array<Eigen::Matrix4cd, 3> result;

  for (std::size_t d = 0; d < 3; d++) {
    result.at(d) = Eigen::Matrix4cd::Zero();
    result.at(d).block(0, 0, 2, 2) = 0.5 * pauli.at(d);
    result.at(d).block(2, 2, 2, 2) = 0.5 * pauli.at(d);
  }

  return result;
}

std::array<std::complex<double>, 3> WsmContinuumHamiltonian::spin(PetscInt ikm, unsigned int mp) const {
  auto km = kmb.decompose(ikm);
  unsigned int m = std::get<1>(km);
  auto k_vals = std::get<0>(kmb.km_at(km));

  if (m > 3 or mp > 3) {
    throw std::invalid_argument("WsmContinuumHamiltonian is not defined unless Nbands == 4");
  }

  auto U = evecs(k_vals);

  return {(U.adjoint() * spin_matrices.at(0) * U)(m, mp),
      (U.adjoint() * spin_matrices.at(1) * U)(m, mp),
      (U.adjoint() * spin_matrices.at(2) * U)(m, mp)};
}

} // namespace anomtrans
