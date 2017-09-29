#include "Rashba_Hamiltonian.h"

namespace anomtrans {

Rashba_Hamiltonian::Rashba_Hamiltonian(double _t0, double _tr, kmBasis<2> _kmb)
  : t0(_t0), tr(_tr), kmb(_kmb) {
    assert(kmb.Nbands == 2);
}

double Rashba_Hamiltonian::energy(kmComps<2> ikm_comps) const {
  kmVals<2> km = km_at(kmb.Nk, ikm_comps);
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

namespace {

std::complex<double> u_component(kVals<2> k) {
  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  if ((k.at(0) == 0.0 and k.at(1) == 0.0)
      or (k.at(0) == 0.5 and k.at(1) == 0.5)) {
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

}

std::complex<double> Rashba_Hamiltonian::basis_component(const PetscInt ikm, const unsigned int i) const {
  auto ikm_comps = kmb.decompose(ikm);

  kmVals<2> km = km_at(kmb.Nk, ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or i > 1) {
    throw std::invalid_argument("Rashba_Hamiltonian is not defined for Nbands > 2");
  }

  if (m == 0) {
    if (i == 0) {
      // u_+
      return u_component(k);
    } else {
      // v_+
      return std::complex<double>(0.0, -1.0/std::sqrt(2.0));
    }
  } else {
    if (i == 0) {
      // u_- = u_+
      return u_component(k);
    } else {
      // v_-
      return std::complex<double>(0.0, 1.0/std::sqrt(2.0));
    }
  }
}

namespace {

std::array<std::complex<double>, 2> gradient_impl(double kx_a, double ky_a,
    double t0, double tr, std::complex<double> um, std::complex<double> vm,
    std::complex<double> ump, std::complex<double> vmp) {
  // Would prefer to use std::complex_literals for this, but not available in C++11.
  // Not sure if this C++14 feature is supported in icpc16 - not given on feature page.
  std::complex<double> i2(0.0, 2.0);

  std::complex<double> vx = std::conj(um) * (2.0 * t0 * std::sin(kx_a) * ump + i2 * tr * std::cos(kx_a) * vmp)
      + std::conj(vm) * (-i2 * tr * std::cos(kx_a) * ump + 2.0 * t0 * std::sin(kx_a) * vmp);

  std::complex<double> vy = std::conj(um) * (2.0 * t0 * std::sin(ky_a) * ump + 2.0 * tr * std::cos(ky_a) * vmp)
      + std::conj(vm) * (2.0 * tr * std::cos(ky_a) * ump + 2.0 * t0 * std::sin(ky_a) * vmp);

  return {vx, vy};
}

} // namespace

std::array<std::complex<double>, 2> Rashba_Hamiltonian::gradient(kmComps<2> ikm_comps, unsigned int mp) const {
  kmVals<2> km = km_at(kmb.Nk, ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("Rashba_Hamiltonian is not defined for Nbands > 2");
  }

  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  PetscInt ikm = kmb.compose(ikm_comps);
  auto ikmp_comps = std::make_tuple(std::get<0>(ikm_comps), mp);
  PetscInt ikmp = kmb.compose(ikmp_comps);

  std::complex<double> um = basis_component(ikm, 0);
  std::complex<double> vm = basis_component(ikm, 1);
  std::complex<double> ump = basis_component(ikmp, 0);
  std::complex<double> vmp = basis_component(ikmp, 1);

  return gradient_impl(kx_a, ky_a, t0, tr, um, vm, ump, vmp);
}

namespace {

std::array<std::complex<double>, 3> spin_impl(std::array<std::complex<double>, 2> um,
    std::array<std::complex<double>, 2> ump) {
  auto pauli = pauli_matrices();
  std::array<std::complex<double>, 3> result = {0.0, 0.0, 0.0};

  // <km|S_a|km'> = \sum_{l,sigma,sigma'} [U_k]_{l,sigma;m} [U_k]^*_{l,sigma';m'} <sigma|S_a|sigma'>
  // For Rashba, only one orbital index (l value).
  for (unsigned int s = 0; s < 2; s++) {
    for (unsigned int sp = 0; sp < 2; sp++) {
      for (std::size_t d = 0; d < 3; d++) {
        // Small, fixed number of elements (4) contributing to each <km|S_a|km'> -
        // don't bother with Kahan sum here.
        // For general H, this number is 4*(Nbands/2), providing motivation
        // for Kahan summation.
        result.at(d) += 0.5 * um.at(s) * std::conj(ump.at(sp)) * pauli.at(d)(s, sp);
      }
    }
  }

  return result;
}

}

std::array<std::complex<double>, 3> Rashba_Hamiltonian::spin(PetscInt ikm, unsigned int mp) const {
  kComps<2> k;
  unsigned int m;
  std::tie(k, m) = kmb.decompose(ikm);

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("Rashba_magnetic_Hamiltonian is not defined for Nbands > 2");
  }

  PetscInt ikmp = kmb.compose(std::make_tuple(k, mp));

  std::array<std::complex<double>, 2> um = {basis_component(ikm, 0), basis_component(ikm, 1)};
  std::array<std::complex<double>, 2> ump = {basis_component(ikmp, 0), basis_component(ikmp, 1)};

  return spin_impl(um, ump);
}

Rashba_magnetic_Hamiltonian::Rashba_magnetic_Hamiltonian(double _t0, double _tr, double _M, kmBasis<2> _kmb)
  : t0(_t0), tr(_tr), M(_M), kmb(_kmb) {
    assert(kmb.Nbands == 2);
}

std::complex<double> Rashba_magnetic_Hamiltonian::Hr(double kx_a, double ky_a) const {
  return 2.0 * tr * (std::complex<double>(0.0, std::sin(kx_a))
      + std::complex<double>(std::sin(ky_a), 0.0));
}

double Rashba_magnetic_Hamiltonian::lambda(double kx_a, double ky_a) const {
  return std::sqrt(std::norm(Hr(kx_a, ky_a)) + std::pow(M, 2.0));
}

double Rashba_magnetic_Hamiltonian::energy(kmComps<2> ikm_comps) const {
  kmVals<2> km = km_at(kmb.Nk, ikm_comps);
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

std::complex<double> magnetic_component_special(double M, unsigned int m, unsigned int i) {
  if (M > 0.0) {
    // M - lambda = 0 case.
    if (m == 0) {
      // |u_k^+>
      if (i == 0) {
        return std::complex<double>(0.0, -1.0);
      } else {
        return std::complex<double>(0.0, 0.0);
      }
    } else {
      // |u_k^->
      if (i == 0) {
        return std::complex<double>(0.0, 0.0);
      } else {
        return std::complex<double>(0.0, 1.0);
      }
    }
  } else {
    // M + lambda = 0 case.
    if (m == 0) {
      // |u_k^+>
      if (i == 0) {
        return std::complex<double>(0.0, 0.0);
      } else {
        return std::complex<double>(0.0, -1.0);
      }
    } else {
      // |u_k^->
      if (i == 0) {
        return std::complex<double>(0.0, -1.0);
      } else {
        return std::complex<double>(0.0, 0.0);
      }
    }
  }
}

} // namespace

std::complex<double> Rashba_magnetic_Hamiltonian::basis_component(const PetscInt ikm, const unsigned int i) const {
  auto ikm_comps = kmb.decompose(ikm);

  kmVals<2> km = km_at(kmb.Nk, ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or i > 1) {
    throw std::invalid_argument("Rashba_magnetic_Hamiltonian is not defined for Nbands > 2");
  }

  if ((k.at(0) == 0.0 and k.at(1) == 0.0)
      or (k.at(0) == 0.5 and k.at(1) == 0.5)) {
    // At k = (0, 0) and k = (1/2, 1/2), the eigenvector components are not well-defined.
    // Choose the values obtained by approaching along the path such that Hr/|Hr| = 1 
    // with M != 0.
    // Here we assume that k in reciprocal lattice coordinates
    // is restricted to [0, 1) x [0, 1).
    // Exact comparison should be OK since 0.5 is exactly representable
    // and generated in km_at by division, not repeated addition.
    return magnetic_component_special(M, m, i);
  }

  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  std::complex<double> ilit(0.0, 1.0);
  double l = lambda(kx_a, ky_a);

  if (m == 0) {
    if (i == 0) {
      // u_+
      return -ilit * Hr(kx_a, ky_a) / std::sqrt(2.0 * l * (l - M));
    } else {
      // v_+
      return -ilit * std::sqrt(l - M) / std::sqrt(2.0 * l);
    }
  } else {
    if (i == 0) {
      // u_-
      return -ilit * Hr(kx_a, ky_a) / std::sqrt(2.0 * l * (l + M));
    } else {
      // v_-
      return ilit * std::sqrt(l + M) / std::sqrt(2.0 * l);
    }
  }
}

std::array<std::complex<double>, 2> Rashba_magnetic_Hamiltonian::gradient(kmComps<2> ikm_comps, unsigned int mp) const {

  kmVals<2> km = km_at(kmb.Nk, ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("Rashba_magnetic_Hamiltonian is not defined for Nbands > 2");
  }

  double kx_a = 2.0*pi*k.at(0);
  double ky_a = 2.0*pi*k.at(1);

  PetscInt ikm = kmb.compose(ikm_comps);
  auto ikmp_comps = std::make_tuple(std::get<0>(ikm_comps), mp);
  PetscInt ikmp = kmb.compose(ikmp_comps);

  // Gradient takes the same form as Rashba_Hamiltonian, since the M term added to
  // the Hamiltonian is k-invariant.
  // Need to use magnetic basis components, though.
  std::complex<double> um = basis_component(ikm, 0);
  std::complex<double> vm = basis_component(ikm, 1);
  std::complex<double> ump = basis_component(ikmp, 0);
  std::complex<double> vmp = basis_component(ikmp, 1);

  return gradient_impl(kx_a, ky_a, t0, tr, um, vm, ump, vmp);
}

std::array<std::complex<double>, 3> Rashba_magnetic_Hamiltonian::spin(PetscInt ikm, unsigned int mp) const {
  kComps<2> k;
  unsigned int m;
  std::tie(k, m) = kmb.decompose(ikm);

  if (m > 1 or mp > 1) {
    throw std::invalid_argument("Rashba_magnetic_Hamiltonian is not defined for Nbands > 2");
  }

  PetscInt ikmp = kmb.compose(std::make_tuple(k, mp));

  std::array<std::complex<double>, 2> um = {basis_component(ikm, 0), basis_component(ikm, 1)};
  std::array<std::complex<double>, 2> ump = {basis_component(ikmp, 0), basis_component(ikmp, 1)};

  return spin_impl(um, ump);
}

} // namespace anomtrans
