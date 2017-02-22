#ifndef ANOMTRANS_COLLISION_H
#define ANOMTRANS_COLLISION_H

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <cmath>
#include <complex>
#include <vector>
#include <tuple>
#include <utility>
#include <petscksp.h>
#include "constants.h"
#include "grid_basis.h"
#include "vec.h"
#include "mat.h"
#include "energy.h"
#include "util.h"

namespace anomtrans {

static_assert(std::is_same<PetscScalar, PetscReal>::value,
    "The implementation of the collision matrix assumes that PetscScalar is a real-valued type.");
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

/** @brief Construct the collision matrix: hbar K.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have the methods
 *           double energy(kmComps<dim>)
 *           and
 *           std::complex<double> basis_component(ikm, i).
 *  @param sigma Standard deviation for Gaussian delta function representation.
 *  @param disorder_term A function with signature
 *                       double f(ikm1, ikm2, ikm3, ikm4)
 *                       giving the disorder-averaged term
 *                       U_{ikm1, ikm2} U_{ikm3, ikm4}.
 *  @todo Sure that Gaussian delta function is appropriate? Lorentzian is natural
 *        given the origin of the term but a poor fit for generating sparsity.
 *        Would cold smearing be better than Gaussian?
 *  @todo Is the expected signature of disorder_term appropriate? Could be more
 *        restrictive in the accepted ikm values (i.e. force ikm3 == ikm2,
 *        ikm4 == ikm1).
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
Mat make_collision(const kmBasis<k_dim> &kmb, const Hamiltonian &H, const double sigma,
    const UU &disorder_term) {
  // TODO could make this an argument to avoid recomputing.
  Vec Ekm = get_energies(kmb, H);
  // We need the full contents of Ekm to construct each row of K.
  // Bring them to each process.
  // TODO could add parameter to get_energies to just construct Ekm as local vector.
  VecScatter ctx;
  Vec Ekm_all;
  PetscErrorCode ierr = VecScatterCreateToAll(Ekm, &ctx, &Ekm_all);CHKERRXX(ierr);
  ierr = VecScatterBegin(ctx, Ekm, Ekm_all, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);
  ierr = VecScatterEnd(ctx, Ekm, Ekm_all, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);

  std::vector<PetscInt> all_rows;
  std::vector<PetscScalar> all_Ekm_vals;
  std::tie(all_rows, all_Ekm_vals) = get_local_contents(Ekm_all);
  assert(all_rows.at(0) == 0);
  assert(all_rows.at(all_rows.size() - 1) == kmb.end_ikm - 1);

  // Need to sort energies and get their permutation index to avoid considering
  // all columns of K when building a row.
  std::vector<std::pair<PetscScalar, PetscInt>> sorted_Ekm;
  for (std::size_t ikm = 0; ikm < static_cast<std::size_t>(kmb.end_ikm); ikm++) {
    sorted_Ekm.push_back(std::make_pair(all_Ekm_vals.at(ikm), ikm));
  }
  std::sort(sorted_Ekm.begin(), sorted_Ekm.end());
  // Now sorted_Ekm.at(i).first is the i'th energy in ascending order and
  // sorted_Ekm.at(i).second is the corresponding ikm value of that
  // energy.
  std::vector<PetscInt> ikm_to_sorted = invert_vals_indices(sorted_Ekm);
  assert(sorted_Ekm.size() == ikm_to_sorted.size());

  // We need to know what values to regard as 'effectively 0' in K.
  // Take this to be any values where the delta function factor is
  // below the threshold given by:
  //   delta(0) * DBL_EPSILON.
  // TODO is this an appropriate scale?
  // Is it sufficient to compare delta_fac value to determine if above
  // threshold, or should UU play a role in comparison and threshold?
  PetscReal threshold = delta_Gaussian(sigma, 0.0) * std::numeric_limits<PetscReal>::epsilon();

  // Assume Ekm has the same local distribution as K.
  // This should be true since K is N x N and Ekm is length N, and local distributions
  // are determined with PETSC_DECIDE.
  // TODO is there a better way to do this?
  PetscInt begin, end;
  ierr = VecGetOwnershipRange(Ekm, &begin, &end);CHKERRXX(ierr);

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
    ierr = VecScatterDestroy(&ctx);CHKERRXX(ierr);
    ierr = VecDestroy(&Ekm_all);CHKERRXX(ierr);
    ierr = VecDestroy(&Ekm);CHKERRXX(ierr);

    throw std::runtime_error("Did not get expected row distribution in K");
  }

  // Set the values of K.
  for (PetscInt local_row = begin; local_row < end; local_row++) {
    PetscInt row_count = row_counts_diag.at(local_row - begin) + row_counts_od.at(local_row - begin);

    std::vector<PetscInt> column_ikms;
    std::vector<PetscScalar> column_vals;
    std::tie(column_ikms, column_vals) = collision_row(kmb, H, sigma, disorder_term,
        sorted_Ekm, ikm_to_sorted, threshold, row_count, local_row);

    ierr = MatSetValues(K, 1, &local_row, column_ikms.size(), column_ikms.data(),
        column_vals.data(), INSERT_VALUES);CHKERRXX(ierr);
  }

  ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  ierr = VecScatterDestroy(&ctx);CHKERRXX(ierr);
  ierr = VecDestroy(&Ekm_all);CHKERRXX(ierr);
  ierr = VecDestroy(&Ekm);CHKERRXX(ierr);

  return K;
}

/** @brief Register an element of the row's nonzero diagonal or off-diagonal
 *         part count. Return true if the element is above threshold
 *         (and thus registered), or false otherwise.
 *  @todo Might want to change this to functional style: avoid updating row_diag
 *        and row_od and instead communicate what needs to update by the return
 *        value (a boost::optional<std::pair<PetscInt, PetscInt>>, for example).
 */
bool collision_count_nonzeros_elem(const double sigma,
    const std::vector<std::pair<PetscScalar, PetscInt>> &sorted_Ekm,
    const PetscReal threshold, const PetscInt begin, const PetscInt end,
    const PetscScalar E_row, const PetscInt sorted_col_index,
    PetscInt &row_diag, PetscInt &row_od);

/** @brief Construct the row structure of the local part of the collision matrix:
 *         return the number of nonzeros in the diagonal and off-diagonal parts
 *         of the collision matrix for the rows belonging to this process.
 */
template <std::size_t k_dim>
std::pair<std::vector<PetscInt>, std::vector<PetscInt>> collision_count_nonzeros(const kmBasis<k_dim> &kmb,
    const double sigma, const std::vector<std::pair<PetscScalar, PetscInt>> &sorted_Ekm,
    const std::vector<PetscInt> &ikm_to_sorted, const PetscReal threshold,
    const PetscInt begin, const PetscInt end) {
  std::vector<PetscInt> row_counts_diag;
  row_counts_diag.reserve(end - begin);
  std::vector<PetscInt> row_counts_od;
  row_counts_od.reserve(end - begin);

  // Iterate through local rows in K basis order.
  for (PetscInt row = begin; row < end; row++) {
    PetscInt row_diag = 1; // Always include the diagonal element.
    PetscInt row_od = 0;

    PetscInt sorted_row_index = ikm_to_sorted.at(row);
    PetscScalar E_row = sorted_Ekm.at(sorted_row_index).first;

    PetscInt end_up = static_cast<PetscInt>(std::floor(kmb.end_ikm/2.0) + 1);
    PetscInt end_down = static_cast<PetscInt>(std::ceil(kmb.end_ikm/2.0));

    // Iterate through columns in sorted order, moving up in energy from row.
    for (PetscInt di = 1; di < end_up; di++) {
      PetscInt sorted_col_index = wrap(sorted_row_index + di, kmb.end_ikm);

      // Process this element: add it to row_diag or row_od if applicable.
      bool more_elems = collision_count_nonzeros_elem(sigma, sorted_Ekm,
          threshold, begin, end, E_row, sorted_col_index, row_diag, row_od);

      if (not more_elems) {
        break;
      }
    }
    // Iterate through columns in sorted order, moving down in energy from row.
    for (PetscInt di = 1; di < end_down; di++) {
      PetscInt sorted_col_index = wrap(sorted_row_index - di, kmb.end_ikm);

      // Process this element: add it to row_diag or row_od if applicable.
      bool more_elems = collision_count_nonzeros_elem(sigma, sorted_Ekm,
          threshold, begin, end, E_row, sorted_col_index, row_diag, row_od);

      if (not more_elems) {
        break;
      }
    }

    row_counts_diag.push_back(row_diag);
    row_counts_od.push_back(row_od);
  }

  return std::make_pair(row_counts_diag, row_counts_od);
}

/** @brief Add an element of the row to the collection of nonzero values, if
 *         large enough. Return true if the element is above threshold
 *         (and thus added), or false otherwise.
 *  @todo Might want to change this to functional style: avoid updating column_ikms
 *        and column_vals and instead communicate what needs to update by the return
 *        value (a boost::optional<std::pair<PetscInt, PetscScalar>>, for example).
 */
template <typename UU>
bool collision_row_elem(const double sigma, const UU &disorder_term,
    const std::vector<std::pair<PetscScalar, PetscInt>> &sorted_Ekm,
    const PetscReal threshold, const PetscScalar E_row, const PetscInt row,
    const PetscInt sorted_col_index, std::vector<PetscInt> &column_ikms,
    std::vector<PetscScalar> &column_vals) {
  PetscScalar E_col = sorted_Ekm.at(sorted_col_index).first;
  PetscInt column = sorted_Ekm.at(sorted_col_index).second;

  double delta_fac = delta_Gaussian(sigma, E_row - E_col);

  // If this element is over threshold, we will store it.
  // TODO could assume delta_fac is always positive.
  // For Gaussian delta, this is true.
  if (std::abs(delta_fac) > threshold) {
    PetscScalar K_elem = -2*pi * delta_fac * disorder_term(row, column, column, row);
    column_ikms.push_back(column);
    column_vals.push_back(K_elem);
    return true;
  } else {
    // All elements farther away than this have a greater energy
    // difference. If this element is below threshold, the rest of them
    // will be too.
    return false;
  }
}

/** @brief Construct the row of the collision matrix with the given row index.
 *  @param threshold Include only elements with absolute value greater than this.
 *  @todo Common infrastructure between this and collision_count_nonzeros?
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
IndexValPairs collision_row(const kmBasis<k_dim> &kmb, const Hamiltonian &H,
    const double sigma, const UU &disorder_term,
    const std::vector<std::pair<PetscScalar, PetscInt>> &sorted_Ekm,
    const std::vector<PetscInt> &ikm_to_sorted, const PetscReal threshold,
    const PetscInt row_count, const PetscInt row) {
  std::vector<PetscInt> column_ikms;
  column_ikms.reserve(row_count);
  std::vector<PetscScalar> column_vals;
  column_vals.reserve(row_count);

  PetscInt sorted_row_index = ikm_to_sorted.at(row);
  PetscScalar E_row = sorted_Ekm.at(sorted_row_index).first;

  PetscInt end_up = static_cast<PetscInt>(std::floor(kmb.end_ikm/2.0) + 1);
  PetscInt end_down = static_cast<PetscInt>(std::ceil(kmb.end_ikm/2.0));

  // Iterate through columns in sorted order, moving up in energy from row.
  for (PetscInt di = 1; di < end_up; di++) {
    PetscInt sorted_col_index = wrap(sorted_row_index + di, kmb.end_ikm);

    // Process this element: add it to column_ikms and column_vals if applicable.
    bool more_elems = collision_row_elem(sigma, disorder_term, sorted_Ekm,
        threshold, E_row, row, sorted_col_index, column_ikms, column_vals);

    if (not more_elems) {
      break;
    }
  }
  // Iterate through columns in sorted order, moving down in energy from row.
  for (PetscInt di = 1; di < end_down; di++) {
    PetscInt sorted_col_index = wrap(sorted_row_index - di, kmb.end_ikm);

    // Process this element: add it to column_ikms and column_vals if applicable.
    bool more_elems = collision_row_elem(sigma, disorder_term, sorted_Ekm,
        threshold, E_row, row, sorted_col_index, column_ikms, column_vals);

    if (not more_elems) {
      break;
    }
  }
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

/** @brief Calculate the disorder-averaged on-site diagonal disorder term.
 *  @note There is an extra factor of U0^2/Nk appearing in <UU>. This is
 *        left out here to avoid passing the parameters.
 */
template <typename Hamiltonian>
double on_site_diagonal_disorder(const unsigned int Nbands, const Hamiltonian &H,
    const PetscInt ikm1, const PetscInt ikm2, const PetscInt ikm3,
    const PetscInt ikm4) {
  // Use Kahan summation for sum over band indices.
  std::complex<double> sum(0.0, 0.0);
  std::complex<double> c(0.0, 0.0);
  for (unsigned int i1 = 0; i1 < Nbands; i1++) {
    for (unsigned int i2 = 0; i2 < Nbands; i2++) {
      std::complex<double> contrib = std::conj(H.basis_component(ikm1, i1))
          * H.basis_component(ikm2, i1)
          * std::conj(H.basis_component(ikm3, i2))
          * H.basis_component(ikm4, i2);

      std::complex<double> y = contrib - c;
      std::complex<double> t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
  }

  // After sum, we should get a real number.
  // TODO - sure this is true?
  // TODO - what is appropriate tol value?
  // Nbands = sqrt(Nbands^2) via Kahan expected error.
  // 1 is an appropriate scale here: the basis component vectors are normalized
  // to 1.
  assert(std::abs(sum.imag()) < Nbands*std::numeric_limits<double>::epsilon());

  return sum.real();
}

} // namespace anomtrans

#endif // ANOMTRANS_COLLISION_H
