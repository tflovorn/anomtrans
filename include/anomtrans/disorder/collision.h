#ifndef ANOMTRANS_COLLISION_H
#define ANOMTRANS_COLLISION_H

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <cmath>
#include <vector>
#include <tuple>
#include <utility>
#include <petscksp.h>
#include "util/constants.h"
#include "grid_basis.h"
#include "util/vec.h"
#include "util/mat.h"
#include "observables/energy.h"

namespace anomtrans {

static_assert(std::is_same<PetscReal, double>::value,
    "The implementation of sigma_min assumes that PetscReal is the same as double.");

/** @brief Gaussian representation of the Dirac delta function.
 *  @param sigma Standard deviation of the Gaussian distribution.
 *  @param x Delta function argument.
 *  @todo Probably should use PetscReal here.
 */
double delta_Gaussian(double sigma, double x);

/** @brief Gives the minimum sigma value that can be expected to allow adequate
 *         sampling of energy differences.
 *  @note This is obtained by enforcing the condition that
 *        delta_{sigma}(E_{km} - E_{k+dk_i,m}) > delta_{sigma}(0)*DBL_EPSILON,
 *        i.e. that nearest-neighbor k-points with the same band index always
 *        have finite 'overlap' as measured by the delta function.
 *  @todo Probably should use output PetscReal and epsilon appropriate for
 *        PetscReal here.
 */
double get_sigma_min(PetscReal max_energy_difference);

/** @brief Calculate the threshold difference in energies beyond which km-points are
 *         considered to be part of difference Fermi surfaces.
 *  @note Take this to be any values where the delta function factor is
 *        below the threshold given by: delta(0) * DBL_EPSILON.
 * @todo Is this an appropriate scale?
 *       Is it sufficient to use delta function value to determine if above
 *       threshold, or should UU play a role in comparison and threshold?
 */
PetscReal get_fermi_surface_threshold(double sigma);

/** @brief Construct the collision matrix: hbar K.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have the methods
 *           double energy(kmComps<dim>)
 *           and
 *           std::complex<double> basis_component(ikm, i).
 *  @param sigma Standard deviation for Gaussian delta function representation.
 *  @param disorder_term A function with signature
 *                       double f(ikm1, ikm2)
 *                       giving the disorder-averaged term
 *                       U_{ikm1, ikm2} U_{ikm2, ikm1}.
 *  @todo Sure that Gaussian delta function is appropriate? Lorentzian is natural
 *        given the origin of the term but a poor fit for generating sparsity.
 *        Would cold smearing be better than Gaussian?
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
OwnedMat make_collision(const kmBasis<k_dim> &kmb, const Hamiltonian &H, const double sigma,
    const UU &disorder_term) {
  // TODO could make this an argument to avoid recomputing.
  auto Ekm = get_energies(kmb, H);
  // We need the full contents of Ekm to construct each row of K.
  // Bring them to each process.
  // TODO could add parameter to get_energies to just construct Ekm as local vector.
  auto Ekm_all = scatter_to_all(Ekm.v);

  // Need to sort energies and get their permutation index to avoid considering
  // all columns of K when building a row.
  SortResult sorted_Ekm;
  std::vector<PetscInt> ikm_to_sorted;
  std::tie(sorted_Ekm, ikm_to_sorted) = sort_energies(kmb, Ekm_all.v);

  // We need to know what values to regard as 'effectively 0' in K.
  // Choose those which are multiplied by a negligible delta function factor.
  PetscReal threshold = get_fermi_surface_threshold(sigma);

  // Assume Ekm has the same local distribution as K.
  // This should be true since K is N x N and Ekm is length N, and local distributions
  // are determined with PETSC_DECIDE.
  // TODO is there a better way to do this?
  PetscInt begin, end;
  PetscErrorCode ierr = VecGetOwnershipRange(Ekm.v, &begin, &end);CHKERRXX(ierr);

  // Count how many nonzeros are in each local row on the diagonal portion
  // (i.e. those elements (ikm1, ikm2) with begin <= ikm2 < end) and the
  // off-diagonal portion (the rest).
  std::vector<PetscInt> row_counts_diag;
  std::vector<PetscInt> row_counts_od;
  std::tie(row_counts_diag, row_counts_od) = collision_count_nonzeros(kmb, sigma,
      sorted_Ekm, ikm_to_sorted, threshold, begin, end);
  assert(static_cast<PetscInt>(row_counts_diag.size()) == end - begin);
  assert(static_cast<PetscInt>(row_counts_od.size()) == end - begin);

  Mat K;
  ierr = MatCreate(PETSC_COMM_WORLD, &K);CHKERRXX(ierr);
  ierr = MatSetType(K, MATMPIAIJ);CHKERRXX(ierr);
  ierr = MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, kmb.end_ikm, kmb.end_ikm);
  // Now that we know how many nonezeros there are, preallocate the memory
  // for K.
  // Note: -1 values here are placeholders for `d_nz` and `o_nz` arguments
  // (nonzero counts which are the same across all rows); these are ignored
  // since we give `d_nnz` and `o_nnz` (nonzero counts for each row).
  ierr = MatMPIAIJSetPreallocation(K, -1, row_counts_diag.data(),
      -1, row_counts_od.data());

  // Make sure the K has the row distribution we think it does.
  // TODO does PETSc guarantee that we don't need to make this check, given
  // that we used PETSC_DECIDE and the same size as the vector dimension?
  // TODO nicer way to clean up PETSc data here?
  PetscInt begin_K, end_K;
  ierr = MatGetOwnershipRange(K, &begin_K, &end_K);CHKERRXX(ierr);
  if (begin_K != begin or end_K != end) {
    ierr = MatDestroy(&K);CHKERRXX(ierr);

    throw std::runtime_error("Did not get expected row distribution in K");
  }

  // Set the values of K.
  for (PetscInt local_row = begin; local_row < end; local_row++) {
    PetscInt row_count = row_counts_diag.at(local_row - begin) + row_counts_od.at(local_row - begin);

    std::vector<PetscInt> column_ikms;
    std::vector<PetscScalar> column_vals;
    std::tie(column_ikms, column_vals) = collision_row(kmb, sigma, disorder_term,
        sorted_Ekm, ikm_to_sorted, threshold, row_count, local_row);

    ierr = MatSetValues(K, 1, &local_row, column_ikms.size(), column_ikms.data(),
        column_vals.data(), INSERT_VALUES);CHKERRXX(ierr);
  }

  ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  return OwnedMat(K);
}

/** @brief Return true iff the (k', m') point given by sorted_ikpmp_index is on
 *         the Fermi surface associated with the point (k, m) with energy E_km.
 */
bool on_fermi_surface(const double sigma, const SortResult &sorted_Ekm,
    const std::vector<PetscInt> &ikm_to_sorted, const PetscReal threshold,
    PetscReal E_km, PetscInt sorted_ikpmp_index);

/** @brief Iterate over the indices ikpmp of the (k', m') points which are on the
 *         Fermi surface associated with the point ikm and call the function fs_fn(ikpmp)
 *         for each of these points.
 *  @param fs_fn A function with the signature `void fs_fn(PetscInt ikpmp)`.
 *  @note Points (k', m') on the Fermi surface of point (k, m) are given by those with
 *        |\delta_{\sigma}(E_{km} - E_{k'm'})| > threshold.
 */
template <std::size_t k_dim, typename F>
void apply_on_fermi_surface(const kmBasis<k_dim> &kmb,
    const double sigma, const SortResult &sorted_Ekm, const std::vector<PetscInt> &ikm_to_sorted,
    const PetscReal threshold, PetscInt ikm, F fs_fn) {
  PetscInt sorted_ikm_index = ikm_to_sorted.at(ikm);
  PetscReal E_km = sorted_Ekm.at(sorted_ikm_index).first;

  // We will iterate through the list of sorted energies, moving up and down away from ikm.
  // If the highest or lowest energy is reached, the up and down iterations will
  // wrap around and continue.
  // `end_up` and `end_down` are 1 + the maximum number of elements which may be
  // encountered in this way.
  PetscInt end_up = static_cast<PetscInt>(std::floor(kmb.end_ikm/2.0) + 1);
  PetscInt end_down = static_cast<PetscInt>(std::ceil(kmb.end_ikm/2.0));

  // Iterate through columns in sorted order, moving up in energy from ikm.
  for (PetscInt di = 1; di < end_up; di++) {
    PetscInt sorted_ikpmp_index = wrap(sorted_ikm_index + di, kmb.end_ikm);

    // Are we still on the Fermi surface? If so, call fs_fn.
    if (on_fermi_surface(sigma, sorted_Ekm, ikm_to_sorted, threshold, E_km, sorted_ikpmp_index)) {
      PetscInt ikpmp = sorted_Ekm.at(sorted_ikpmp_index).second;
      fs_fn(ikpmp);
    } else {
      // No longer on Fermi surface: we can stop moving up in energy.
      break;
    }
  }

  // Iterate through columns in sorted order, moving down in energy from row.
  for (PetscInt di = 1; di < end_down; di++) {
    PetscInt sorted_ikpmp_index = wrap(sorted_ikm_index - di, kmb.end_ikm);

    // Are we still on the Fermi surface? If so, call fs_fn.
    if (on_fermi_surface(sigma, sorted_Ekm, ikm_to_sorted, threshold, E_km, sorted_ikpmp_index)) {
      PetscInt ikpmp = sorted_Ekm.at(sorted_ikpmp_index).second;
      fs_fn(ikpmp);
    } else {
      // No longer on Fermi surface: we can stop moving down in energy.
      break;
    }
  }
}

/** @brief Construct the row structure of the local part of the collision matrix:
 *         return the number of nonzeros in the diagonal and off-diagonal parts
 *         of the collision matrix for the rows belonging to this process.
 */
template <std::size_t k_dim>
std::pair<std::vector<PetscInt>, std::vector<PetscInt>> collision_count_nonzeros(const kmBasis<k_dim> &kmb,
    const double sigma, const SortResult &sorted_Ekm, const std::vector<PetscInt> &ikm_to_sorted,
    const PetscReal threshold, const PetscInt begin, const PetscInt end) {
  std::vector<PetscInt> row_counts_diag;
  row_counts_diag.reserve(end - begin);
  std::vector<PetscInt> row_counts_od;
  row_counts_od.reserve(end - begin);

  // Iterate through local rows in Kdd basis order.
  for (PetscInt row = begin; row < end; row++) {
    // Columns ikpmp on the Fermi surface of the row point ikm will have nonzero contribution
    // to Kdd. Count these points.
    PetscInt row_diag = 1; // Always include the diagonal element.
    PetscInt row_od = 0;

    auto update_count = [begin, end, &row_diag, &row_od](PetscInt col) {
      if (begin <= col and col < end) {
        row_diag++;
      } else {
        row_od++;
      }
    };

    apply_on_fermi_surface(kmb, sigma, sorted_Ekm, ikm_to_sorted, threshold, row, update_count);

    row_counts_diag.push_back(row_diag);
    row_counts_od.push_back(row_od);
  }

  return std::make_pair(row_counts_diag, row_counts_od);
}

/** @brief Construct the row of the collision matrix with the given row index.
 *  @param threshold Include only elements with absolute value greater than this.
 *  @todo Common infrastructure between this and collision_count_nonzeros?
 */
template <std::size_t k_dim, typename UU>
IndexValPairs collision_row(const kmBasis<k_dim> &kmb,
    const double sigma, const UU &disorder_term,
    const SortResult &sorted_Ekm, const std::vector<PetscInt> &ikm_to_sorted,
    const PetscReal threshold, const PetscInt row_count, const PetscInt row) {
  std::vector<PetscInt> column_ikms;
  column_ikms.reserve(row_count);
  std::vector<PetscScalar> column_vals;
  column_vals.reserve(row_count);

  PetscInt sorted_row_index = ikm_to_sorted.at(row);
  PetscReal E_row = sorted_Ekm.at(sorted_row_index).first;

  // Calculate the contributions of the column points ikpmp on this row ikm.
  auto add_elem = [E_row, row, sigma, &disorder_term, &sorted_Ekm, &ikm_to_sorted,
       &column_ikms, &column_vals](PetscInt col) {
    // TODO could just use col here if passed Ekm.
    PetscInt sorted_col_index = ikm_to_sorted.at(col);
    PetscReal E_col = sorted_Ekm.at(sorted_col_index).first;

    double delta_fac = delta_Gaussian(sigma, E_row - E_col);
    PetscScalar K_elem = -2*pi * delta_fac * disorder_term(row, col);

    column_ikms.push_back(col);
    column_vals.push_back(K_elem);
  };

  apply_on_fermi_surface(kmb, sigma, sorted_Ekm, ikm_to_sorted, threshold, row, add_elem);

  // Always include diagonal element.
  // Its value is given by (-1)*(sum of elements in this row).
  // May be many elements in row, so use Kahan summation.
  PetscScalar row_sum = 0.0;
  PetscScalar c = 0.0;
  for (auto elem : column_vals) {
    PetscScalar y = elem - c;
    PetscScalar t = row_sum + y;
    c = (t - row_sum) - y;
    row_sum = t;
  }
  PetscScalar diag_val = -row_sum;
  column_ikms.push_back(row);
  column_vals.push_back(diag_val);

  assert(static_cast<PetscInt>(column_ikms.size()) == row_count);
  assert(static_cast<PetscInt>(column_vals.size()) == row_count);

  return IndexValPairs(column_ikms, column_vals);
}

} // namespace anomtrans

#endif // ANOMTRANS_COLLISION_H
