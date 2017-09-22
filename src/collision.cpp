#include "collision.h"

namespace anomtrans {

double delta_Gaussian(double sigma, double x) {
  double exp_arg = -x*x/(2*sigma*sigma);
  if (exp_arg < LN_DBL_MIN) {
    return 0.0;
  }
  double coeff = 1/(std::sqrt(2*pi)*sigma);
  return coeff * std::exp(exp_arg);
}

double get_sigma_min(PetscReal max_energy_difference) {
  double coeff = 1.0 / std::pow(-2*LN_DBL_EPS, 0.5);
  return coeff * max_energy_difference;
}

bool on_fermi_surface(const double sigma, const SortResult &sorted_Ekm,
    const std::vector<PetscInt> &ikm_to_sorted, const PetscReal threshold,
    PetscReal E_km, PetscInt sorted_ikpmp_index) {
  PetscReal E_kpmp = sorted_Ekm.at(sorted_ikpmp_index).first;

  double delta_fac = delta_Gaussian(sigma, E_km - E_kpmp);

  return std::abs(delta_fac) > threshold;
}

} // namespace anomtrans
