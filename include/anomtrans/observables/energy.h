#ifndef ANOMTRANS_ENERGY_H
#define ANOMTRANS_ENERGY_H

#include <cstddef>
#include <vector>
#include <algorithm>
#include <petscksp.h>
#include "util/vec.h"
#include "grid_basis.h"
#include "derivative.h"
#include "util/util.h"

namespace anomtrans {

/** @brief Construct a vector of energies using the given k-space discretization
 *         and Hamiltonian.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have
 *           the method energy(kmComps<dim>).
 *  @note Vec is a reference type (Petsc implementation: typedef struct _p_Vec* Vec).
 *        We can safely create an object of type Vec on the stack, initialize it with
 *        VecCreate, and return it. The caller will need to call VecDestroy(&v) on
 *        the returned Vec.
 *  @todo Make kmb a constant ref - avoid copy.
 */
template <std::size_t k_dim, typename Hamiltonian>
Vec get_energies(kmBasis<k_dim> kmb, Hamiltonian H) {
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
  Vec Ekm = get_energies(kmb, H);

  // First-order approximant for foward difference first derivative
  // = (E_{k+dk_i, m} - E_{k,m})/kmb.Nk.at(i)
  DerivStencil<1> stencil(DerivApproxType::forward, 1);

  std::array<Mat, k_dim> d_dk = make_d_dk_recip(kmb, stencil);

  Vec dE_dk;
  PetscErrorCode ierr = VecDuplicate(Ekm, &dE_dk);CHKERRXX(ierr);

  PetscReal dE_dk_max = 0.0;
  for (std::size_t d = 0; d < k_dim; d++) {
    ierr = MatMult(d_dk.at(d), Ekm, dE_dk);CHKERRXX(ierr);
    PetscReal dE_dk_d_max = get_Vec_MaxAbs(dE_dk) * kmb.Nk.at(d);

    if (dE_dk_d_max > dE_dk_max) {
      dE_dk_max = dE_dk_d_max;
    }
  }

  return dE_dk_max;
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
Mat apply_precession_term(const kmBasis<k_dim> &kmb, Hamiltonian H, Mat rho, double broadening) {
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

    std::vector<PetscScalar> result_row;

    for (PetscInt column_index = 0; column_index < ncols; column_index++) {
      PetscInt ikmp = cols[column_index];
      auto kmp = kmb.decompose(ikmp);
      // Make sure that <rho> is k-diagonal.
      // Prefer to check this with if/throw? Choosing assert() here since it's in inner loop.
      assert(std::get<0>(kmp) == std::get<0>(km));

      double ediff = H.energy(km) - H.energy(kmp);
      PetscScalar elem = std::complex<double>(0.0, -ediff/(std::pow(ediff, 2.0) + std::pow(broadening, 2.0)));

      result_row.push_back(elem);
    }

    assert(result_row.size() == static_cast<std::size_t>(ncols));
    ierr = MatSetValues(result, 1, &ikm, ncols, cols, result_row.data(), INSERT_VALUES);CHKERRXX(ierr);

    ierr = MatRestoreRow(rho, ikm, &ncols, &cols, &vals);CHKERRXX(ierr);
  }

  ierr = MatAssemblyBegin(result, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(result, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  return result;
}

} // namespace anomtrans

#endif // ANOMTRANS_ENERGY_H
