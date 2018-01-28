#include "observables/rho0.h"

namespace anomtrans {

OwnedVec make_rho0(Vec energies, double beta, double mu) {
  auto fd = [beta, mu](std::complex<double> E)->std::complex<double> {
    return std::complex<double>(fermi_dirac(beta, E.real() - mu), 0.0);
  };

  auto rho0 = vector_elem_apply(energies, fd);
  return rho0;
}

double get_beta_max(PetscReal max_energy_difference) {
  return 2.0 / max_energy_difference;
}

} // namespace anomtrans
