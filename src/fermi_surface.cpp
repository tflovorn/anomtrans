#include "fermi_surface.h"

namespace anomtrans {

PetscReal get_n_threshold(const std::vector<PetscScalar> &n_all) {
  double n_max = 0.0;
  for (PetscScalar n : n_all) {
    double n_abs = std::abs(n);
    if (n_abs > n_max) {
      n_max = n_abs;
    }
  }

  return std::numeric_limits<double>::epsilon() * n_max;
}

PetscReal DeltaBin::get_width(const SortResult &sorted_Ekm, unsigned int num_fs) {
  PetscReal E_min = sorted_Ekm.at(0).first;
  PetscReal E_max = sorted_Ekm.at(sorted_Ekm.size() - 1).first;

  return (E_max - E_min) / num_fs;
}

PetscInt DeltaBin::get_bin(PetscReal E) const {
  if (E < E_min or E > E_max) {
    throw std::invalid_argument("must have E within [E_min, E_max] range");
  }

  PetscInt fs_index = static_cast<PetscInt>(std::floor((E - E_min) / width));

  // For `E = E_max`, we will get `fs_index == num_fs` if there is no rounding error.
  // Assign this to the last bin by hand (second clause). Perform similar operation for
  // `E = E_min`, in case rounding error yields `fs_index == -1` (first clause).
  if (fs_index == -1) {
    fs_index = 0;
  } else if (fs_index == static_cast<PetscInt>(num_fs)) {
    fs_index = num_fs - 1;
  }

  return fs_index;
}

PetscReal DeltaBin::operator()(PetscReal E1, PetscReal E2) const {
  PetscInt bin1 = get_bin(E1);
  PetscInt bin2 = get_bin(E2);

  if (bin1 == bin2) {
    return 1.0 / width;
  } else {
    return 0.0;
  }
}

PetscReal DeltaGaussian::get_threshold(PetscReal sigma) {
  PetscReal gaussian_max = 1.0 / (std::sqrt(2.0 * pi) * sigma);
  return gaussian_max * std::numeric_limits<PetscReal>::epsilon();
}

PetscReal DeltaGaussian::get_sigma_min(PetscReal max_energy_difference) {
  PetscReal coeff = 1.0 / std::pow(-2*LN_DBL_EPS, 0.5);
  return coeff * max_energy_difference;
}

PetscReal DeltaGaussian::operator()(PetscReal E1, PetscReal E2) const {
  PetscReal x = E1 - E2;
  PetscReal exp_arg = -x*x/(2*sigma*sigma);
  if (exp_arg < LN_DBL_MIN) {
    return 0.0;
  }
  PetscReal coeff = 1.0 / (std::sqrt(2.0 * pi) * sigma);
  return coeff * std::exp(exp_arg);
}

} // namespace anomtrans
