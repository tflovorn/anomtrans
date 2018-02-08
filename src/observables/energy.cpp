#include "observables/energy.h"

namespace anomtrans {

PetscReal FermiSurfaceMap::get_width(const SortResult &sorted_Ekm, unsigned int num_fs) {
  PetscReal E_min = sorted_Ekm.at(0).first;
  PetscReal E_max = sorted_Ekm.at(sorted_Ekm.size() - 1).first;

  return (E_max - E_min) / num_fs;
}

PetscReal FermiSurfaceMap::get_gaussian_sigma_factor(PetscReal width, PetscReal gaussian_width_fraction) {
  return -1.0 / (2.0 * std::pow(width, 2.0) * std::pow(gaussian_width_fraction, 2.0));
}

PetscReal FermiSurfaceMap::gaussian_norm_correction(PetscReal gaussian_width_fraction) {
  if (gaussian_width_fraction <= 0.0) {
    throw std::invalid_argument("must give Gaussian width > 0");
  }

  // For Gaussian delta function representation,
  // `\int_{-w/2}^{w/2} dx \delta_{f * w}(x) = erf(1 / (2 * sqrt(2) * f))`.
  return boost::math::erf(1.0 / (2.0 * std::sqrt(2.0) * gaussian_width_fraction));
}

PetscReal FermiSurfaceMap::get_gaussian_coeff(PetscReal width, PetscReal gaussian_width_fraction) {
  PetscReal delta_coeff = gaussian_width_fraction / (std::sqrt(2.0 * pi) * width);
  return gaussian_norm_correction(gaussian_width_fraction) * delta_coeff;
}

FsMapPair FermiSurfaceMap::make_fs_map(const SortResult &sorted_Ekm,
    unsigned int num_fs) {
  auto w = get_width(sorted_Ekm, num_fs);
  PetscReal E_min = sorted_Ekm.at(0).first;

  std::vector<std::size_t> all_fs_index(sorted_Ekm.size());
  std::vector<std::vector<PetscInt>> fs_members(num_fs);

  PetscInt end_ikm = sorted_Ekm.size();

  for (PetscInt sorted_index = 0; sorted_index < end_ikm; sorted_index++) {
    PetscReal E;
    PetscInt ikm;
    std::tie(E, ikm) = sorted_Ekm.at(sorted_index);

    PetscInt fs_index = static_cast<PetscInt>(std::floor((E - E_min) / w));

    // For `E = E_max`, we will get `fs_index == num_fs` if there is no rounding error.
    // Assign this to the last bin by hand (second clause). Perform similar operation for
    // `E = E_min`, in case rounding error yields `fs_index == -1` (first clause).
    if (fs_index == -1) {
      fs_index = 0;
    } else if (fs_index == static_cast<PetscInt>(num_fs)) {
      fs_index = num_fs - 1;
    }
    
    if (fs_index < 0 or fs_index > static_cast<PetscInt>(num_fs)) {
      throw std::runtime_error("got Fermi surface index outside of expected range");
    }

    all_fs_index.at(ikm) = static_cast<std::size_t>(fs_index);
    fs_members.at(fs_index).push_back(ikm);
  }

  return std::make_pair(all_fs_index, fs_members);
}

std::size_t FermiSurfaceMap::fs_index(PetscInt ikm) const {
  return fs_map.first.at(ikm);
}

const std::vector<PetscInt>& FermiSurfaceMap::fs_partners(PetscInt ikm) const {
  return fs_map.second.at(fs_index(ikm));
}

PetscReal FermiSurfaceMap::gaussian(PetscReal x) const {
  PetscReal exp_term = std::exp(gaussian_sigma_factor * x*x);
  return gaussian_coeff * exp_term;
}

} // namespace anomtrans
