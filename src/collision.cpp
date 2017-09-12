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

bool collision_count_nonzeros_elem(const double sigma,
    const std::vector<std::pair<PetscReal, PetscInt>> &sorted_Ekm,
    const PetscReal threshold, const PetscInt begin, const PetscInt end,
    const PetscReal E_row, const PetscInt sorted_col_index,
    PetscInt &row_diag, PetscInt &row_od) {
  PetscReal E_col = sorted_Ekm.at(sorted_col_index).first;
  PetscInt column = sorted_Ekm.at(sorted_col_index).second;

  double delta_fac = delta_Gaussian(sigma, E_row - E_col);

  // If this element is over threshold, we will store it.
  // TODO could assume delta_fac is always positive.
  // For Gaussian delta, this is true.
  if (std::abs(delta_fac) > threshold) {
    if (begin <= column and column < end) {
      row_diag++;
    } else {
      row_od++;
    }
    return true;
  } else {
    // All elements farther away than this have a greater energy
    // difference. If this element is below threshold, the rest of them
    // will be too.
    return false;
  }
}

} // namespace anomtrans
