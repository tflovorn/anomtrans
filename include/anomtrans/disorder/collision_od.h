#ifndef ANOMTRANS_COLLISION_OD_H
#define ANOMTRANS_COLLISION_OD_H

#include <cstddef>
#include <complex>
#include <vector>
#include <tuple>
#include <petscksp.h>
#include "util/constants.h"
#include "grid_basis.h"
#include "util/vec.h"
#include "observables/energy.h"
#include "disorder/collision.h"

namespace anomtrans {

/** @brief Compute the result of applying the diagonal-to-off-diagonal collision operator
 *         $K^{od}$ to the given diagonal density matrix `n_all`: returns 
 *         $\hbar K^{od}(<n>)$.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have the methods
 *           double energy(kmComps<dim>)
 *           and
 *           std::complex<double> basis_component(ikm, i).
 *  @param sigma Standard deviation for Gaussian delta function representation.
 *  @param disorder_term A function with signature `complex<double> f(ikm1, ikm2, ikm3)`
 *                       giving the disorder-averaged term U_{ikm1, ikm2} U_{ikm2, ikm3}.
 *                       Must have k1 = k3 for valid result.
 *  @param n_all The diagonal density matrix to which the off-diagonal collision term
 *               is applied. Passed as a std::vector here since all nodes require all values.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU_OD>
OwnedMat apply_collision_od(const kmBasis<k_dim> &kmb, const Hamiltonian &H, const double sigma,
    const UU_OD &disorder_term, const std::vector<PetscScalar> &n_all) {
  // TODO could make this an argument to avoid recomputing.
  Vec Ekm = get_energies(kmb, H);
  // We need the full contents of Ekm to construct each row of K.
  // Bring them to each process.
  // TODO could add parameter to get_energies to just construct Ekm as local vector.
  Vec Ekm_all = scatter_to_all(Ekm);
  // TODO consolidate Ekm_all and Ekm_all_std - replace Ekm_all with std::vector?
  auto Ekm_all_std = split_scalars(std::get<1>(get_local_contents(Ekm_all))).first;

  // Need to sort energies and get their permutation index to avoid considering
  // all columns of K when building a row.
  SortResult sorted_Ekm;
  std::vector<PetscInt> ikm_to_sorted;
  std::tie(sorted_Ekm, ikm_to_sorted) = sort_energies(kmb, Ekm_all);

  PetscReal threshold = get_fermi_surface_threshold(sigma);

  // [S]_{km, k''m''} = \delta_{k, k''} [J(<n>)]_{k}^{m, m''}
  OwnedMat Jn = make_Mat(kmb.end_ikm, kmb.end_ikm, kmb.Nbands);

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(Jn.M, &begin, &end);CHKERRXX(ierr);

  for (PetscInt ikm = begin; ikm < end; ikm++) {
    kmComps<k_dim> km = kmb.decompose(ikm);
    kComps<k_dim> k = std::get<0>(km);
    unsigned int m = std::get<1>(km);

    PetscReal eps_km = Ekm_all_std.at(ikm);

    std::vector<PetscInt> column_ikms;
    std::vector<PetscScalar> column_vals;

    for (unsigned int mpp = 0; mpp < kmb.Nbands; mpp++) {
      if (mpp == m) {
        // Jn includes only off-diagonal terms.
        continue;
      }

      kmComps<k_dim> kmpp = std::make_tuple(k, mpp);
      PetscInt ikmpp = kmb.compose(kmpp);

      PetscReal eps_kmpp = Ekm_all_std.at(ikmpp);

      PetscScalar total = std::complex<double>(0.0, 0.0);

      auto add_km_term = [eps_km, ikm, ikmpp, &Ekm_all_std, &sigma, &disorder_term,
           &n_all, &total](PetscInt ikpmp) {
        PetscReal eps_kpmp = Ekm_all_std.at(ikpmp);

        std::complex<double> U_part = disorder_term(ikm, ikpmp, ikmpp);

        PetscReal ndiff = n_all.at(ikm).real() - n_all.at(ikpmp).real();
        double delta_fac = delta_Gaussian(sigma, eps_km - eps_kpmp);

        // TODO - use Kahan sum.
        total += pi * U_part * ndiff * delta_fac;
      };

      apply_on_fermi_surface(kmb, sigma, sorted_Ekm, ikm_to_sorted, threshold, ikm, add_km_term);

      auto add_kmpp_term = [eps_kmpp, ikm, ikmpp, &Ekm_all_std, &sigma, &disorder_term,
           &n_all, &total](PetscInt ikpmp) {
        PetscReal eps_kpmp = Ekm_all_std.at(ikpmp);

        std::complex<double> U_part = disorder_term(ikm, ikpmp, ikmpp);

        PetscReal ndiff = n_all.at(ikmpp).real() - n_all.at(ikpmp).real();
        double delta_fac = delta_Gaussian(sigma, eps_kmpp - eps_kpmp);

        // TODO - use Kahan sum.
        total += pi * U_part * ndiff * delta_fac;
      };

      apply_on_fermi_surface(kmb, sigma, sorted_Ekm, ikm_to_sorted, threshold, ikmpp, add_kmpp_term);

      column_ikms.push_back(ikmpp);
      column_vals.push_back(total);
    }

    ierr = MatSetValues(Jn.M, 1, &ikm, column_ikms.size(), column_ikms.data(), column_vals.data(),
        INSERT_VALUES);CHKERRXX(ierr);
  }

  ierr = MatAssemblyBegin(Jn.M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(Jn.M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  ierr = VecDestroy(&Ekm_all);CHKERRXX(ierr);
  ierr = VecDestroy(&Ekm);CHKERRXX(ierr);

  return Jn;
}

} // namespace anomtrans

#endif // ANOMTRANS_COLLISION_OD_H
