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

bool collision_count_nonzeros_elem(const double spread,
    const std::vector<std::pair<PetscScalar, PetscInt>> &sorted_Ekm,
    const PetscReal threshold, const PetscInt begin, const PetscInt end,
    const PetscScalar E_row, const PetscInt sorted_col_index,
    PetscInt &row_diag, PetscInt &row_od) {
  PetscScalar E_col = sorted_Ekm.at(sorted_col_index).first;
  PetscInt column = sorted_Ekm.at(sorted_col_index).second;

  double delta_fac = delta_Gaussian(spread, E_row - E_col);

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
