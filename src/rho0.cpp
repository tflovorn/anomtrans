#include "rho0.h"

namespace anomtrans {

Vec make_rho0(Vec energies, double beta, double mu) {
  auto fd = [beta, mu](double E)->double {
    return fermi_dirac(beta, E - mu);
  };

  Vec rho0 = vector_elem_apply(energies, fd);
  return rho0;
}

} // namespace anomtrans
