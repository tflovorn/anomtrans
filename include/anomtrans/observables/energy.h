#ifndef ANOMTRANS_OBSERVABLES_ENERGY_H
#define ANOMTRANS_OBSERVABLES_ENERGY_H

#include <cstddef>
#include <cmath>
#include <exception>
#include <vector>
#include <algorithm>
#include <boost/math/special_functions/erf.hpp>
#include <petscksp.h>
#include "util/vec.h"
#include "util/util.h"
#include "grid_basis.h"
#include "derivative.h"

namespace anomtrans {

/** @brief Construct a vector of energies using the given k-space discretization
 *         and Hamiltonian.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have
 *           the method energy(kmComps<dim>).
 */
template <std::size_t k_dim, typename Hamiltonian>
OwnedVec get_energies(const kmBasis<k_dim> &kmb, const Hamiltonian &H) {
  // TODO may want to just pass ikm to H: look up precomputed v without
  // conversion to ikm_comps and back.
  auto E_elem = [kmb, H](PetscInt ikm)->PetscScalar {
    auto ikm_comps = kmb.decompose(ikm);
    return H.energy(ikm_comps);
  };

  return vector_index_apply(kmb.end_ikm, E_elem);
}

/** @brief Find the maximum value of |E_{k+dk_i, m} - E_{k, m}| where dk_i is
 *         a step in one direction in reciprocal space, with the step distance
 *         given by the minimum step in that direction (1/kmb.Nk.at(i)).
 */
template <std::size_t k_dim, typename Hamiltonian>
PetscReal find_max_energy_difference(const kmBasis<k_dim> &kmb, const Hamiltonian &H) {
  auto Ekm = get_energies(kmb, H);

  // First-order approximant for foward difference first derivative
  // = (E_{k+dk_i, m} - E_{k,m})/h_i.
  DerivStencil<1> stencil(DerivApproxType::forward, 1);

  auto d_dk = make_d_dk_recip(kmb, stencil);
  auto dE_dk = make_Vec_with_structure(Ekm.v);

  PetscReal ediff_max = 0.0;
  for (std::size_t d = 0; d < k_dim; d++) {
    PetscErrorCode ierr = MatMult(d_dk.at(d).M, Ekm.v, dE_dk.v);CHKERRXX(ierr);

    PetscReal ediff_d_max = get_Vec_MaxAbs(dE_dk.v) * kmb.k_step(d);

    if (ediff_d_max > ediff_max) {
      ediff_max = ediff_d_max;
    }
  }

  return ediff_max;
}

using SortResult = std::vector<std::pair<PetscReal, PetscInt>>;

/** @brief Sort the values of Ekm_all and return the sorted values along with
 *         lists giving the preimage and image of the sort.
 *  @returns A pair of values: first, a vector of (sorted energy, original index) pairs;
 *           second, a vector whose elements (in the unsorted basis) give the index of the
 *           corresponding element of the sorted vector.
 */
template <std::size_t k_dim>
std::pair<SortResult, std::vector<PetscInt>> sort_energies(const kmBasis<k_dim> &kmb,
    Vec Ekm_all) {
  std::vector<PetscInt> all_rows;
  std::vector<PetscScalar> all_Ekm_vals;
  std::tie(all_rows, all_Ekm_vals) = get_local_contents(Ekm_all);
  assert(all_rows.at(0) == 0);
  assert(all_rows.at(all_rows.size() - 1) == kmb.end_ikm - 1);

  SortResult sorted_Ekm;
  for (std::size_t ikm = 0; ikm < static_cast<std::size_t>(kmb.end_ikm); ikm++) {
    sorted_Ekm.push_back(std::make_pair(all_Ekm_vals.at(ikm).real(), ikm));
  }
  std::sort(sorted_Ekm.begin(), sorted_Ekm.end());
  // Now sorted_Ekm.at(i).first is the i'th energy in ascending order and
  // sorted_Ekm.at(i).second is the corresponding ikm value of that
  // energy.
  std::vector<PetscInt> ikm_to_sorted = invert_vals_indices(sorted_Ekm);
  assert(sorted_Ekm.size() == ikm_to_sorted.size());

  return std::make_pair(sorted_Ekm, ikm_to_sorted);
}

using FsMapPair = std::pair<std::vector<size_t>, std::vector<std::vector<PetscInt>>>;

class FermiSurfaceMap {
  static PetscReal get_width(const SortResult &sorted_Ekm, unsigned int num_fs);

  static PetscReal get_gaussian_sigma_factor(PetscReal width, PetscReal gaussian_width_fraction);

  static PetscReal gaussian_norm_correction(PetscReal gaussian_width_fraction);

  static PetscReal get_gaussian_coeff(PetscReal width, PetscReal gaussian_width_fraction);

  static FsMapPair make_fs_map(const SortResult &sorted_Ekm, unsigned int num_fs);

public:
  /** @brief The number of Fermi surface bins.
   */
  const std::size_t num_fs;

  /** @brief The width in energy of each Fermi surface bin.
   */
  const PetscReal width;

  /** @brief Within each Fermi surface bin, the delta function is represented as a Gaussian
   *         with standard deviation `sigma = width * gaussian_width_fraction`.
   *         Due to the restriction to the bin, this Gaussian would integrate to 1 within the bin;
   *         the normalization is corrected for this. This correction is very minor for
   *         `gaussian_width_fraction` < ~1/8.
   */
  const PetscReal gaussian_width_fraction;

  /** @brief Create a mapping from each point `ikm` to its Fermi surface partners.
   *         Points are divided based on energy into `num_fs` equally-sized Fermi surfaces.
   */
  FermiSurfaceMap(const SortResult &sorted_Ekm, unsigned int _num_fs,
      PetscReal _gaussian_width_fraction)
      : num_fs(_num_fs), width(get_width(sorted_Ekm, _num_fs)),
        gaussian_width_fraction(_gaussian_width_fraction),
        fs_map(make_fs_map(sorted_Ekm, _num_fs)),
        gaussian_sigma_factor(get_gaussian_sigma_factor(width, gaussian_width_fraction)),
        gaussian_coeff(get_gaussian_coeff(width, gaussian_width_fraction)) {}

  /** @brief Return the index of the Fermi surface associated with the representative
   *         point `ikm`.
   */
  std::size_t fs_index(PetscInt ikm) const;

  /** @brief Return the members of the Fermi surface associated with the representative
   *         point `ikm`.
   */
  const std::vector<PetscInt>& fs_partners(PetscInt ikm) const;

  /** @brief Gaussian delta function representation, with standard deviation
   *         `sigma = width * gaussian_width_fraction`.
   */
  PetscReal gaussian(PetscReal x) const;

private:
  /** @brief Each (k, m) index `ikm` is assiged a Fermi surface, with index
   *         `fs_map.first[ikm]`. Each Fermi surface `i` has members `ikm`
   *         given by `fs_map.second[i]`.
   */
  const FsMapPair fs_map;

  const PetscReal gaussian_sigma_factor;
  const PetscReal gaussian_coeff;
};

/** @brief Apply the precession term P^{-1} to the density matrix `rho` and return
 *         the resulting value
 *         [P^{-1} <rho>]_k^{mm'} = -i\hbar <rho>_k^{mm'} / (E_{km} - E_{km'}).
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have
 *           the method energy(kmComps<dim>).
 *  @note Degeneracies are treated by introducing a Lorentzian broadening:
 *        1 / (E_{km} - E_{km'}) is replaced by
 *        (E_{km} - E_{km'}) / ((E_{km} - E_{km'})^2 + broadening^2).
 *  @pre <rho> should be diagonal in k, i.e. <rho>_{km, k'm'} has no entries where k' != k.
 *  @todo Prefer to act on rho in-place?
 */
template <std::size_t k_dim, typename Hamiltonian>
OwnedMat apply_precession_term(const kmBasis<k_dim> &kmb, const Hamiltonian &H, Mat rho,
    double broadening) {
  return apply_precession_term_dynamic(kmb, H, rho, broadening, 0, 0.0);
}

/** @brief Apply the dynamic precession term P_n^{-1} to the density matrix `rho` and return
 *         the resulting value
 *         [P_n^{-1} <rho>]_k^{mm'} =
 *           -i\hbar <rho>_k^{mm'} / (\hbar * omega * n + E_{km} - E_{km'}).
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have
 *           the method energy(kmComps<dim>).
 *  @note Degeneracies are treated by introducing a Lorentzian broadening:
 *        1 / (E_{km} - E_{km'}) is replaced by
 *        (E_{km} - E_{km'}) / ((E_{km} - E_{km'})^2 + broadening^2).
 *  @pre <rho> should be diagonal in k, i.e. <rho>_{km, k'm'} has no entries where k' != k.
 *  @todo Prefer to act on rho in-place?
 */
template <std::size_t k_dim, typename Hamiltonian>
OwnedMat apply_precession_term_dynamic(const kmBasis<k_dim> &kmb, const Hamiltonian &H, Mat rho,
    double broadening, int n, double omega) {
  Mat result;
  PetscErrorCode ierr = MatDuplicate(rho, MAT_SHARE_NONZERO_PATTERN, &result);CHKERRXX(ierr);

  PetscInt begin, end;
  ierr = MatGetOwnershipRange(rho, &begin, &end);CHKERRXX(ierr);

  for (PetscInt ikm = begin; ikm < end; ikm++) {
    PetscInt ncols;
    const PetscInt *cols;
    const PetscScalar *vals;
    ierr = MatGetRow(rho, ikm, &ncols, &cols, &vals);CHKERRXX(ierr);

    auto km = kmb.decompose(ikm);

    std::vector<PetscInt> result_cols;
    std::vector<PetscScalar> result_row;

    for (PetscInt column_index = 0; column_index < ncols; column_index++) {
      PetscInt ikmp = cols[column_index];
      auto kmp = kmb.decompose(ikmp);
      // Make sure that <rho> is k-diagonal.
      // Prefer to check this with if/throw? Choosing assert() here since it's in inner loop.
      assert(std::get<0>(kmp) == std::get<0>(km));

      // P^{-1} is applied only to purely off-diagonal matrices. Skip diagonal elements.
      if (std::get<1>(km) == std::get<1>(kmp)) {
        continue;
      }

      double ediff = H.energy(km) - H.energy(kmp);
      double P_elem_imag = omega * n + ediff;
      PetscScalar coeff = std::complex<double>(0.0, -P_elem_imag/(std::pow(P_elem_imag, 2.0) + std::pow(broadening, 2.0)));
      PetscScalar elem = coeff * vals[column_index];

      result_cols.push_back(ikmp);
      result_row.push_back(elem);
    }

    assert(result_row.size() == result_cols.size());
    ierr = MatSetValues(result, 1, &ikm, result_cols.size(), result_cols.data(), result_row.data(), INSERT_VALUES);CHKERRXX(ierr);

    ierr = MatRestoreRow(rho, ikm, &ncols, &cols, &vals);CHKERRXX(ierr);
  }

  ierr = MatAssemblyBegin(result, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(result, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  return OwnedMat(result);
}

} // namespace anomtrans

#endif // ANOMTRANS_OBSERVABLES_ENERGY_H
