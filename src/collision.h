#ifndef ANOMTRANS_COLLISION_H
#define ANOMTRANS_COLLISION_H

#include <cassert>
#include <cstddef>
#include <cfloat>
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
 *  @todo Better method for getting the sparsity structure: iterate once to obtain
 *        the number of nonzero columns in each row, and then again to fill in the
 *        values of these columns.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
Mat make_collision(kmBasis<k_dim> kmb, Hamiltonian H, double spread,
    UU disorder_term) {
  // The number of finite elements per row is on the order of the Fermi
  // surface size on that row.
  // The Fermi surface has dimension one less than k_dim, so estimate
  // its size as ~ (Nk)^((k_dim - 1)/k_dim).
  // TODO is this an appropriate value? Can check if too small by increasing and
  // seeing if performance of this function is improved.
  // Can check if too large by reducing and seeing if less memory is used
  // (is this correct? does the extra preallocated memory stay allocated after
  // maxtrix construction is complete?).
  // In particular, when k_dim is 1, get expected_elems_per_row = 1.
  // Clearly the appropriate value is at least this large.
  // TODO can replace this by iterating once over the elements and obtaining the
  // number of nonzeros in each row directly.
  PetscInt Nk_total = 1;
  for (std::size_t d = 0; d < k_dim; d++) {
    Nk_total *= kmb.Nk.at(d);
  }
  PetscInt expected_elems_per_row = std::ceil(std::pow(Nk_total, static_cast<double>(k_dim - 1)/k_dim));

  Mat K = make_Mat(kmb.end_ikm, kmb.end_ikm, expected_elems_per_row);
  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(K, &begin, &end);CHKERRXX(ierr);

  // TODO could make this an argument to avoid recomputing.
  Vec Ekm = get_energies(kmb, H);
  // We need the full contents of Ekm to construct each row of K.
  // Bring them to each process.
  // TODO could add parameter to get_energies to just construct Ekm as local vector.
  VecScatter ctx;
  Vec Ekm_all;
  ierr = VecScatterCreateToAll(Ekm, &ctx, &Ekm_all);CHKERRXX(ierr);
  ierr = VecScatterBegin(ctx, Ekm, Ekm_all, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);
  ierr = VecScatterEnd(ctx, Ekm, Ekm_all, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);

  std::vector<PetscInt> all_rows;
  std::vector<PetscScalar> all_Ekm_vals;
  std::tie(all_rows, all_Ekm_vals) = get_local_contents(Ekm_all);
  assert(all_rows.at(0) == 0);
  assert(all_rows.at(all_rows.size() - 1) == kmb.end_ikm - 1);

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    std::vector<PetscInt> column_ikms;
    std::vector<PetscScalar> column_vals;
    std::tie(column_ikms, column_vals) = collision_row(kmb, H, spread, disorder_term, all_Ekm_vals, local_row);

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

/** @brief Construct the row of the collision matrix with the given row index.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU>
IndexValPairs collision_row(kmBasis<k_dim> kmb, Hamiltonian H, double spread,
    UU disorder_term, std::vector<PetscScalar> Ekm, PetscInt row) {
  std::vector<PetscInt> column_ikms;
  std::vector<PetscScalar> column_vals;
  // TODO how should the threshold for including values be chosen?
  // Certainly we can use DBL_MIN.
  // Much tighter bound possible: can use something like
  // (maximum value of K element)*DBL_EPSILON.
  // Don't have all the K values here, but we could collect all values in the
  // row and then filter out those that don't meet threshold
  // (maximum value in row)*DBL_EPSILON.
  // Could use first iteration over K to get number of nonzeros in each row
  // to also get the maximum value.

  for (PetscInt column = 0; column < kmb.end_ikm; column++) {
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
      // TODO only add this value if magnitude above appropriate threshold
      column_ikms.push_back(column);
      column_vals.push_back(sum);
    } else {
      double delta_fac = delta_Gaussian(spread, Ekm.at(row) - Ekm.at(column));
      PetscScalar K_elem = -2*pi * disorder_term(row, column, column, row) * delta_fac;

      // TODO only add this value if magnitude above appropriate threshold
      column_ikms.push_back(column);
      column_vals.push_back(K_elem);
    }
  }

  return IndexValPairs(column_ikms, column_vals); 
}

/** @brief Calculate the disorder-averaged on-site diagonal disorder term.
 *  @note Does not include disorder strength: this should be passed to
 *        make_collision via a closure which multiplies the output of this
 *        function by U_0^2 where U_0 is the disorder strength. This is omitted
 *        to keep the signature of make_collision sufficiently general
 *        (other disorder terms may carry various other parameters).
 */
template <std::size_t k_dim, typename Hamiltonian>
double on_site_diagonal_disorder(kmBasis<k_dim> kmb, Hamiltonian H,
    PetscInt ikm1, PetscInt ikm2, PetscInt ikm3, PetscInt ikm4) {
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
  double tol = kmb.Nbands*DBL_EPSILON;
  assert(std::abs(sum.imag()) < tol);

  return sum.real();
}

} // namespace anomtrans

#endif // ANOMTRANS_COLLISION_H
