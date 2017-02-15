#ifndef ANOMTRANS_COLLISION_H
#define ANOMTRANS_COLLISION_H

#include <cassert>
#include <cstddef>
#include <limits>
#include <cmath>
#include <complex>
#include <vector>
#include <tuple>
#include <petscksp.h>
#include "constants.h"
#include "grid_basis.h"
#include "vec.h"
#include "mat.h"
#include "energy.h"

namespace anomtrans {

/** @brief Gaussian representation of the Dirac delta function.
 *  @param sigma Standard deviation of the Gaussian distribution.
 *  @param x Delta function argument.
 */
double delta_Gaussian(double sigma, double x);

/** @brief Construct the collision matrix: hbar K.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have the methods
 *           double energy(kmComps<dim>)
 *           and
 *           std::complex<double> basis_component(kmComps<dim>, i).
 *  @param spread Spread parameter for Gaussian delta function representation.
 *  @param disorder_term A function with signature
 *                       double f(ikm1, ikm2, ikm3, ikm4)
 *                       giving the disorder-averaged term
 *                       U_{ikm1, ikm2} U_{ikm3, ikm4}.
 *  @todo Sure that Gaussian delta function is appropriate? Lorentzian is natural
 *        given the origin of the term but a poor fit for generating sparsity.
 *        Would cold smearing be better than Gaussian?
 *  @todo Is the expected signature of disorder_term appropriate? Could be more
 *        restrictive in the accepted ikm values.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
Mat make_collision(kmBasis<k_dim> kmb, Hamiltonian H, double spread,
    UU disorder_term) {
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

  // We need to know what values to regard as 'effectively 0' in K.
  // Take this to be any values below the threshold given by
  // (maximum absolute value elem in K) * DBL_EPSILON.
  PetscReal global_max = collision_max(kmb, H, spread, disorder_term, all_Ekm_vals);
  PetscReal threshold = global_max * std::numeric_limits<PetscReal>::epsilon();

  // Assume Ekm has the same local distribution as K.
  // This should be true since K is N x N and Ekm is length N, and local distributions
  // are determined with PETSC_DECIDE.
  PetscInt begin, end;
  ierr = VecGetOwnershipRange(Ekm, &begin, &end);CHKERRXX(ierr);

  // Count how many nonzeros are in each local row on the diagonal portion
  // (i.e. those elements (ikm1, ikm2) with begin <= ikm2 < end) and the
  // off-diagonal portion (the rest).
  std::vector<PetscInt> row_counts_diag;
  std::vector<PetscInt> row_counts_od;
  std::tie(row_counts_diag, row_counts_od) = collision_count_nonzeros(kmb, H, spread,
      disorder_term, all_Ekm_vals, threshold, begin, end);
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

  // Set the values of K.
  for (PetscInt local_row = begin; local_row < end; local_row++) {
    PetscInt row_count = row_counts_diag.at(local_row - begin) + row_counts_od.at(local_row - begin);

    std::vector<PetscInt> column_ikms;
    std::vector<PetscScalar> column_vals;
    std::tie(column_ikms, column_vals) = collision_row(kmb, H, spread, disorder_term,
        all_Ekm_vals, threshold, row_count, local_row);

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

template <std::size_t k_dim, typename Hamiltonian, typename UU>
PetscReal collision_max(kmBasis<k_dim> kmb, Hamiltonian H, double spread,
    UU disorder_term, std::vector<PetscScalar> Ekm) {
  Vec max_vals;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm, &max_vals);CHKERRXX(ierr);

  PetscInt begin, end;
  ierr = VecGetOwnershipRange(max_vals, &begin, &end);CHKERRXX(ierr);

  std::vector<PetscInt> local_rows;
  local_rows.reserve(end - begin);
  std::vector<PetscScalar> local_vals;
  local_vals.reserve(end - begin);

  for (PetscInt row = begin; row < end; row++) {
    PetscReal row_max = std::numeric_limits<PetscReal>::lowest();
    for (PetscInt column = 0; column < kmb.end_ikm; column++) {
      // TODO this breaks if PetscScalar is complex.
      // Could use a template abs function that specialized to call floating
      // point abs or complex abs as needed.
      PetscScalar K_elem = collision_elem(kmb, H, spread, disorder_term, Ekm, row, column);
      if (std::abs(K_elem) > row_max) {
        row_max = std::abs(K_elem);
      }
    }

    local_rows.push_back(row);
    local_vals.push_back(row_max);
  }

  ierr = VecSetValues(max_vals, local_rows.size(), local_rows.data(), local_vals.data(), INSERT_VALUES);CHKERRXX(ierr);
  ierr = VecAssemblyBegin(max_vals);CHKERRXX(ierr);
  ierr = VecAssemblyEnd(max_vals);CHKERRXX(ierr);

  PetscReal global_max;
  ierr = VecMax(max_vals, nullptr, &global_max);CHKERRXX(ierr);

  ierr = VecDestroy(&max_vals);CHKERRXX(ierr);

  return global_max;
}

/** @brief Construct the row structure of the local part of the collision matrix:
 *         return the number of nonzeros in the diagonal and off-diagonal parts
 *         of the collision matrix for the rows belonging to this process.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
std::tuple<std::vector<PetscInt>, std::vector<PetscInt>> collision_count_nonzeros(kmBasis<k_dim> kmb,
    Hamiltonian H, double spread, UU disorder_term, std::vector<PetscScalar> Ekm,
    PetscReal threshold, PetscInt begin, PetscInt end) {
  std::vector<PetscInt> row_counts_diag;
  row_counts_diag.reserve(end - begin);
  std::vector<PetscInt> row_counts_od;
  row_counts_od.reserve(end - begin);

  for (PetscInt row = begin; row < end; row++) {
    PetscInt row_diag = 0;
    PetscInt row_od = 0;

    for (PetscInt column = 0; column < kmb.end_ikm; column++) {
      PetscScalar K_elem = collision_elem(kmb, H, spread, disorder_term, Ekm, row, column);
      if (std::abs(K_elem) > threshold) {
        if (begin <= column and column < end) {
          row_diag++;
        } else {
          row_od++;
        }
      }
    }

    row_counts_diag.push_back(row_diag);
    row_counts_od.push_back(row_od);
  }

  return std::make_tuple(row_counts_diag, row_counts_od);
}

/** @brief Construct the row of the collision matrix with the given row index.
 *  @param threshold Include only elements with absolute value greater than this.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
IndexValPairs collision_row(kmBasis<k_dim> kmb, Hamiltonian H, double spread,
    UU disorder_term, std::vector<PetscScalar> Ekm, PetscReal threshold,
    PetscInt row_count, PetscInt row) {
  std::vector<PetscInt> column_ikms;
  column_ikms.reserve(row_count);
  std::vector<PetscScalar> column_vals;
  column_vals.reserve(row_count);

  for (PetscInt column = 0; column < kmb.end_ikm; column++) {
    PetscScalar K_elem = collision_elem(kmb, H, spread, disorder_term, Ekm, row, column);
    if (std::abs(K_elem) > threshold) {
      column_ikms.push_back(column);
      column_vals.push_back(K_elem);
    }
  }

  assert(static_cast<PetscInt>(column_ikms.size()) == row_count);
  assert(static_cast<PetscInt>(column_vals.size()) == row_count);

  return IndexValPairs(column_ikms, column_vals);
}

/** @brief Construct the element of the collision matrix with the given row
 *         and column indices.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
PetscScalar collision_elem(kmBasis<k_dim> kmb, Hamiltonian H, double spread,
    UU disorder_term, std::vector<PetscScalar> Ekm, PetscInt row, PetscInt column) {
  if (column == row) {
    // Diagonal term sums over all vector indices.
    // Use Kahan summation.
    PetscScalar sum = 0.0;
    PetscScalar c = 0.0;
    for (PetscInt ikm_pp = 0; ikm_pp < kmb.end_ikm; ikm_pp++) {
      double delta_fac = delta_Gaussian(spread, Ekm.at(row) - Ekm.at(ikm_pp));
      PetscScalar contrib = 2*pi * disorder_term(row, ikm_pp, ikm_pp, row) * delta_fac;

      PetscScalar y = contrib - c;
      PetscScalar t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    return sum;
  } else {
    double delta_fac = delta_Gaussian(spread, Ekm.at(row) - Ekm.at(column));
    PetscScalar K_elem = -2*pi * disorder_term(row, column, column, row) * delta_fac;
    return K_elem;
  }
}

/** @brief Calculate the disorder-averaged on-site diagonal disorder term.
 */
template <std::size_t k_dim, typename Hamiltonian>
double on_site_diagonal_disorder(kmBasis<k_dim> kmb, Hamiltonian H,
    double U0, PetscInt ikm1, PetscInt ikm2, PetscInt ikm3, PetscInt ikm4) {
  auto km1 = kmb.decompose(ikm1);
  auto km4 = kmb.decompose(ikm4);
  if (std::get<0>(km1) != std::get<0>(km4)) {
    return 0.0;
  }
  auto km2 = kmb.decompose(ikm2);
  auto km3 = kmb.decompose(ikm3);
  if (std::get<0>(km2) != std::get<0>(km3)) {
    return 0.0;
  }

  // Use Kahan summation for sum over band indices.
  std::complex<double> sum(0.0, 0.0);
  std::complex<double> c(0.0, 0.0);
  for (unsigned int i1 = 0; i1 < kmb.Nbands; i1++) {
    for (unsigned int i2 = 0; i2 < kmb.Nbands; i2++) {
      std::complex<double> contrib = std::conj(H.basis_component(km1, i1))
          * H.basis_component(km2, i1)
          * std::conj(H.basis_component(km3, i2))
          * H.basis_component(km4, i2);

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
  double tol = kmb.Nbands*std::numeric_limits<double>::epsilon();
  assert(std::abs(sum.imag()) < tol);

  PetscInt Nk_tot = 1;
  for (std::size_t d = 0; d < k_dim; d++) {
    Nk_tot *= kmb.Nk.at(d);
  }

  return U0*U0*sum.real()/Nk_tot;
}

} // namespace anomtrans

#endif // ANOMTRANS_COLLISION_H
