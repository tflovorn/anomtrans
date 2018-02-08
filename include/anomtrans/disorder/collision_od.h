#ifndef ANOMTRANS_COLLISION_OD_H
#define ANOMTRANS_COLLISION_OD_H

#include <cstddef>
#include <complex>
#include <vector>
#include <tuple>
#include <petscksp.h>
#include "util/constants.h"
#include "util/util.h"
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
 *  @param disorder_term A function with signature `complex<double> f(ikm1, ikm2, ikm3)`
 *                       giving the disorder-averaged term U_{ikm1, ikm2} U_{ikm2, ikm3}.
 *                       Must have k1 = k3 for valid result.
 *  @param n_all The diagonal density matrix to which the off-diagonal collision term
 *               is applied. Passed as a std::vector here since all nodes require all values.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU_OD, typename Delta>
OwnedMat apply_collision_od(const kmBasis<k_dim>& kmb, const Hamiltonian& H,
    const UU_OD& disorder_term, const Delta& delta, const std::vector<PetscScalar>& n_all) {
  // TODO could make this an argument to avoid recomputing.
  auto Ekm = get_energies(kmb, H);
  // We need the full contents of Ekm to construct each row of K.
  // Bring them to each process.
  // TODO could add parameter to get_energies to just construct Ekm as local vector.
  auto Ekm_all = scatter_to_all(Ekm.v);
  auto Ekm_all_std = split_scalars(std::get<1>(get_local_contents(Ekm_all.v))).first;

  // Need to sort energies and get their permutation index to avoid considering
  // all columns of K when building a row.
  SortResult sorted_Ekm;
  std::vector<PetscInt> ikm_to_sorted;
  std::tie(sorted_Ekm, ikm_to_sorted) = sort_energies(kmb, Ekm_all.v);

  PetscReal n_threshold = get_n_threshold(n_all);
  const auto nonzero_fs = find_nonzero_fs(delta, sorted_Ekm, n_all, n_threshold);

  // [S]_{km, k''m''} = \delta_{k, k''} [J(<n>)]_{k}^{m, m''}
  OwnedMat Jn = make_Mat(kmb.end_ikm, kmb.end_ikm, kmb.Nbands);

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(Jn.M, &begin, &end);CHKERRXX(ierr);

  for (PetscInt ikm = begin; ikm < end; ikm++) {
    kmComps<k_dim> km = kmb.decompose(ikm);
    kComps<k_dim> k = std::get<0>(km);
    unsigned int m = std::get<1>(km);

    std::vector<PetscInt> column_ikms;
    std::vector<PetscScalar> column_vals;

    for (unsigned int mpp = 0; mpp < kmb.Nbands; mpp++) {
      if (mpp == m) {
        // Jn includes only off-diagonal terms.
        continue;
      }

      kmComps<k_dim> kmpp = std::make_tuple(k, mpp);
      PetscInt ikmpp = kmb.compose(kmpp);

      PetscScalar total = std::complex<double>(0.0, 0.0);

      if (nonzero_fs.at(ikm)) {
        auto update_total_ikm = [ikm, ikmpp, &total, &disorder_term, &delta, &Ekm_all_std, &n_all](PetscInt ikpmp) {
          std::complex<double> U_part = disorder_term(ikm, ikpmp, ikmpp);
          PetscReal ndiff = n_all.at(ikm).real() - n_all.at(ikpmp).real(); // TODO remove real(), keep im part?
          PetscReal delta_factor = delta(Ekm_all_std.at(ikm), Ekm_all_std.at(ikpmp));

          // TODO - use Kahan sum.
          total += pi * U_part * ndiff * delta_factor;
        };

        apply_on_fermi_surface(delta, sorted_Ekm, ikm_to_sorted, ikm, update_total_ikm);
      }

      if (nonzero_fs.at(ikmpp)) {
        auto update_total_ikmpp = [ikm, ikmpp, &total, &disorder_term, &delta, &Ekm_all_std, &n_all](PetscInt ikpmp) {
          std::complex<double> U_part = disorder_term(ikm, ikpmp, ikmpp);
          PetscReal ndiff = n_all.at(ikmpp).real() - n_all.at(ikpmp).real(); // TODO remove real(), keep im part?
          PetscReal delta_factor = delta(Ekm_all_std.at(ikmpp), Ekm_all_std.at(ikpmp));

          // TODO - use Kahan sum.
          total += pi * U_part * ndiff * delta_factor;
        };

        apply_on_fermi_surface(delta, sorted_Ekm, ikm_to_sorted, ikmpp, update_total_ikmpp);
      }

      if (nonzero_fs.at(ikm) or nonzero_fs.at(ikmpp)) {
        column_ikms.push_back(ikmpp);
        column_vals.push_back(total);
      }
    }

    ierr = MatSetValues(Jn.M, 1, &ikm, column_ikms.size(), column_ikms.data(), column_vals.data(),
        INSERT_VALUES);CHKERRXX(ierr);
  }

  ierr = MatAssemblyBegin(Jn.M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(Jn.M, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  return Jn;
}

} // namespace anomtrans

#endif // ANOMTRANS_COLLISION_OD_H
