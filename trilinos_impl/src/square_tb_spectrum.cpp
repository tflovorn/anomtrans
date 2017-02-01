#include "square_tb_spectrum.h"

namespace anomtrans {

square_tb_Hamiltonian::square_tb_Hamiltonian(double _t, double _tp, kComps<2> _Nk)
      : t(_t), tp(_tp), Nk(_Nk) {}

double square_tb_Hamiltonian::energy(kmComps<2> ikm_comps) {
  kmVals<2> km = km_at(Nk, ikm_comps);
  kVals<2> k = std::get<0>(km);
  unsigned int m = std::get<1>(km);

  if (m != 0) {
    throw std::invalid_argument("square_tb_Hamiltonian is not defined for Nbands > 1");
  }

  double kx = 2*pi*k.at(0);
  double ky = 2*pi*k.at(1);

  return -2*t*(std::cos(kx) + std::cos(ky)) + 4*tp*std::cos(kx)*std::cos(ky);
}

std::complex<double> basis_component(kmComps<2> ikm_comps, unsigned int i) {
  return std::complex<double>(1.0, 0.0);
}

} // namespace anomtrans
