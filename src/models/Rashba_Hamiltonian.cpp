#include "models/Rashba_Hamiltonian.h"

namespace anomtrans {

Rashba_Hamiltonian::Rashba_Hamiltonian(double _t0, double _tr, kmBasis<2> _kmb)
  : t0(_t0), tr(_tr), kmb(_kmb) {
  if (kmb.Nbands != 2) {
    throw std::invalid_argument("must supply kmb with 2 bands to Rashba_Hamiltonian");
  }
}

double Rashba_Hamiltonian::energy(kmComps<2> ikm_comps) const {
  kmVals<2> km = kmb.km_at(ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1) {
    throw std::invalid_argument("Rashba_Hamiltonian is not defined for Nbands > 2");
  }

  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  double diag_term = 2.0 * t0 * (2.0 - std::cos(kx_a) - std::cos(ky_a));

  double rashba_term = 2.0 * std::abs(tr) * std::sqrt(std::pow(std::sin(kx_a), 2.0) +
      std::pow(std::sin(ky_a), 2.0));

  if (m == 0) {
    return diag_term + rashba_term;
  } else {
    return diag_term - rashba_term;
  }
}

std::complex<double> Rashba_Hamiltonian::u_component(kVals<2> k) const {
  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  if ((k.at(0) == 0.0 or k.at(0) == 0.5)
      and (k.at(1) == 0.0 or k.at(1) == 0.5)) {
    // At k = (0, 0) and k = (1/2, 1/2), bands are degenerate
    // and the u component is not well-defined.
    // Choose the u value obtained by approaching k = 0 along
    // (kx > 0, ky = 0).
    // Here we assume that k in reciprocal lattice coordinates
    // is restricted to [0, 1) x [0, 1).
    // Exact comparison should be OK since 0.5 is exactly representable
    // and generated in km_at by division, not repeated addition.
    return std::complex<double>(1.0/std::sqrt(2.0));
  }

  double denom = std::sqrt(2.0 * (std::pow(std::sin(kx_a), 2.0) +
      std::pow(std::sin(ky_a), 2.0)));

  return std::complex<double>(std::sin(kx_a) / denom, -std::sin(ky_a) / denom);
}

Eigen::Matrix2cd Rashba_Hamiltonian::evecs(kVals<2> k) const {
  std::complex<double> plus_up = u_component(k);
  std::complex<double> plus_down(0.0, -1.0/std::sqrt(2.0));
  std::complex<double> minus_up = plus_up;
  std::complex<double> minus_down = -plus_down;

  Eigen::Matrix2cd U;
  U << plus_up, minus_up,
      plus_down, minus_down;

  return U;
}

namespace {

/** @note Implementation of gradient is separated since it is the same for Rashba
 *        and Rashba_magnetic.
 */
std::array<Eigen::Matrix2cd, 2> grad_H_impl(kVals<2> k, double t0, double tr) {
  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  std::complex<double> dH0_dx(2.0 * t0 * std::sin(kx_a), 0.0);
  std::complex<double> dH0_dy(2.0 * t0 * std::sin(ky_a), 0.0);

  std::complex<double> dHr_dx(0.0, 2.0 * tr * std::cos(kx_a));
  std::complex<double> dHr_dy(2.0 * tr * std::cos(ky_a), 0.0);

  Eigen::Matrix2cd dH_dx;
  dH_dx << dH0_dx, dHr_dx,
      std::conj(dHr_dx), dH0_dx;

  Eigen::Matrix2cd dH_dy;
  dH_dy << dH0_dy, dHr_dy,
      std::conj(dHr_dy), dH0_dy;

  return {dH_dx, dH_dy};
}

} // namespace

std::array<Eigen::Matrix2cd, 2> Rashba_Hamiltonian::grad_H(kVals<2> k) const {
  return grad_H_impl(k, t0, tr);
}

std::complex<double> Rashba_Hamiltonian::basis_component(const PetscInt ikm, const unsigned int i) const {
  auto ikm_comps = kmb.decompose(ikm);

  kmVals<2> km = kmb.km_at(ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or i > 1) {
    throw std::invalid_argument("Rashba_Hamiltonian is not defined for Nbands > 2");
  }

  auto U = evecs(k);
  return U(i, m);
}

std::array<std::complex<double>, 2> Rashba_Hamiltonian::gradient(kmComps<2> ikm_comps, unsigned int mp) const {
  kmVals<2> km = kmb.km_at(ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("Rashba_Hamiltonian is not defined for Nbands > 2");
  }

  auto U = evecs(k);
  auto grad = grad_H(k);

  return {(U.adjoint() * grad.at(0) * U)(m, mp),
      (U.adjoint() * grad.at(1) * U)(m, mp)};
}

namespace {

std::array<Eigen::Matrix2cd, 3> spin_impl(Eigen::Matrix2cd U) {
  auto pauli = pauli_matrices();
  std::array<Eigen::Matrix2cd, 3> result;

  // <km|S_a|km'> = \sum_{l,sigma,sigma'} [U_k]_{l,sigma;m} [U_k]^*_{l,sigma';m'} <sigma|S_a|sigma'>
  // For Rashba, only one orbital index (l value).
  for (std::size_t d = 0; d < 3; d++) {
    // Small, fixed number of elements (4) contributing to each <km|S_a|km'> -
    // don't bother with Kahan sum here.
    // For general H, this number is 4*(Nbands/2), providing motivation
    // for Kahan summation over bands (l's).
    result.at(d) = 0.5 * U.adjoint() * pauli.at(d) * U;
  }

  return result;
}

} // namespace

std::array<std::complex<double>, 3> Rashba_Hamiltonian::spin(PetscInt ikm, unsigned int mp) const {
  auto km = kmb.decompose(ikm);
  unsigned int m = std::get<1>(km);
  auto k_vals = std::get<0>(kmb.km_at(km));

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("Rashba_magnetic_Hamiltonian is not defined for Nbands > 2");
  }

  auto U = evecs(k_vals);
  auto S = spin_impl(U);

  return {S.at(0)(m, mp), S.at(1)(m, mp), S.at(2)(m, mp)};
}

Rashba_magnetic_Hamiltonian::Rashba_magnetic_Hamiltonian(double _t0, double _tr, double _M, kmBasis<2> _kmb)
  : t0(_t0), tr(_tr), M(_M), kmb(_kmb) {
  if (kmb.Nbands != 2) {
    throw std::invalid_argument("must supply kmb with 2 bands to Rashba_magnetic_Hamiltonian");
  }
}

std::complex<double> Rashba_magnetic_Hamiltonian::Hr(double kx_a, double ky_a) const {
  return 2.0 * tr * (std::complex<double>(0.0, std::sin(kx_a))
      + std::complex<double>(std::sin(ky_a), 0.0));
}

double Rashba_magnetic_Hamiltonian::lambda(double kx_a, double ky_a) const {
  return std::sqrt(std::norm(Hr(kx_a, ky_a)) + std::pow(M, 2.0));
}

double Rashba_magnetic_Hamiltonian::energy(kmComps<2> ikm_comps) const {
  kmVals<2> km = kmb.km_at(ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1) {
    throw std::invalid_argument("Rashba_magnetic_Hamiltonian is not defined for Nbands > 2");
  }

  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  double diag_term = 2.0 * t0 * (2.0 - std::cos(kx_a) - std::cos(ky_a));

  if (m == 0) {
    return diag_term + lambda(kx_a, ky_a);
  } else {
    return diag_term - lambda(kx_a, ky_a);
  }
}

namespace {

Eigen::Matrix2cd magnetic_component_special(double M) {
  std::complex<double> ilit(0.0, 1.0);
  if (M > 0.0) {
    // M - lambda = 0 case.
    Eigen::Matrix2cd U;
    U << -ilit, 0.0,
        0.0, ilit;

    return U;
  } else {
    // M + lambda = 0 case.
    Eigen::Matrix2cd U;
    U << 0.0, -ilit,
        -ilit, 0.0;

    return U;
  }
}

} // namespace

Eigen::Matrix2cd Rashba_magnetic_Hamiltonian::evecs(kVals<2> k) const {
  if ((k.at(0) == 0.0 or k.at(0) == 0.5)
      and (k.at(1) == 0.0 or k.at(1) == 0.5)) {
    // At k = (0, 0) and k = (1/2, 1/2), the eigenvector components are not well-defined.
    // Choose the values obtained by approaching along the path such that Hr/|Hr| = 1 
    // with M != 0.
    // Here we assume that k in reciprocal lattice coordinates
    // is restricted to [0, 1) x [0, 1).
    // Exact comparison should be OK since 0.5 is exactly representable
    // and generated in km_at by division, not repeated addition.
    return magnetic_component_special(M);
  }

  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  std::complex<double> ilit(0.0, 1.0);
  double l = lambda(kx_a, ky_a);

  std::complex<double> plus_up = -ilit * Hr(kx_a, ky_a) / std::sqrt(2.0 * l * (l - M));
  std::complex<double> plus_down = -ilit * std::sqrt(l - M) / std::sqrt(2.0 * l);
  std::complex<double> minus_up = -ilit * Hr(kx_a, ky_a) / std::sqrt(2.0 * l * (l + M));
  std::complex<double> minus_down = ilit * std::sqrt(l + M) / std::sqrt(2.0 * l);

  Eigen::Matrix2cd U;
  U << plus_up, minus_up,
      plus_down, minus_down;

  return U;
}

std::complex<double> Rashba_magnetic_Hamiltonian::basis_component(const PetscInt ikm, const unsigned int i) const {
  auto ikm_comps = kmb.decompose(ikm);

  kmVals<2> km = kmb.km_at(ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or i > 1) {
    throw std::invalid_argument("Rashba_magnetic_Hamiltonian is not defined for Nbands > 2");
  }

  auto U = evecs(k);
  return U(i, m);
}

std::array<Eigen::Matrix2cd, 2> Rashba_magnetic_Hamiltonian::grad_H(kVals<2> k) const {
  return grad_H_impl(k, t0, tr);
}

std::array<std::complex<double>, 2> Rashba_magnetic_Hamiltonian::gradient(kmComps<2> ikm_comps,
    unsigned int mp) const {
  kmVals<2> km = kmb.km_at(ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("Rashba_Hamiltonian is not defined for Nbands > 2");
  }

  auto U = evecs(k);
  auto grad = grad_H(k);

  return {(U.adjoint() * grad.at(0) * U)(m, mp),
      (U.adjoint() * grad.at(1) * U)(m, mp)};
}

std::array<std::complex<double>, 3> Rashba_magnetic_Hamiltonian::spin(PetscInt ikm, unsigned int mp) const {
  auto km = kmb.decompose(ikm);
  unsigned int m = std::get<1>(km);
  auto k_vals = std::get<0>(kmb.km_at(km));

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("Rashba_magnetic_Hamiltonian is not defined for Nbands > 2");
  }

  auto U = evecs(k_vals);
  auto S = spin_impl(U);

  return {S.at(0)(m, mp), S.at(1)(m, mp), S.at(2)(m, mp)};
}

} // namespace anomtrans
