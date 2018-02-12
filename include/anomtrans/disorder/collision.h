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
#include "util/util.h"
#include "grid_basis.h"
#include "util/vec.h"
#include "util/mat.h"
#include "fermi_surface.h"
#include "observables/energy.h"

namespace anomtrans {

template <typename UU, typename Delta>
class CollisionContext {
public:
  const UU& disorder_term;
  const Delta& delta;
  const std::vector<PetscReal> Ekm;
  const SortResult sorted_Ekm;
  const std::vector<PetscInt> ikm_to_sorted;

  CollisionContext(const UU& _disorder_term, const Delta& _delta,
      const std::vector<PetscReal> _Ekm,
      const SortResult _sorted_Ekm, const std::vector<PetscInt> _ikm_to_sorted)
      : disorder_term(_disorder_term), delta(_delta), Ekm(_Ekm), sorted_Ekm(_sorted_Ekm),
        ikm_to_sorted(_ikm_to_sorted) {}
};

template <typename UU, typename Delta>
PetscErrorCode apply_collision_shell(Mat K, Vec n, Vec result) {
  auto n_all = std::get<1>(get_local_contents(scatter_to_all(n).v));

  void* ctx_void;
  PetscErrorCode ierr = MatShellGetContext(K, &ctx_void);CHKERRXX(ierr);
  CollisionContext<UU, Delta>* K_ctx = static_cast<CollisionContext<UU, Delta>*>(ctx_void);

  PetscReal n_threshold = get_n_threshold(n_all);
  auto nonzero_fs = find_nonzero_fs(K_ctx->delta, K_ctx->sorted_Ekm, n_all, n_threshold);

  PetscInt begin, end;
  ierr = VecGetOwnershipRange(n, &begin, &end);CHKERRXX(ierr);

  ierr = VecSet(result, 0.0);CHKERRXX(ierr);

  #pragma omp parallel
  {
    std::vector<PetscInt> local_rows;
    std::vector<PetscScalar> local_vals;

    #pragma omp for schedule(dynamic)
    for (PetscInt ikm = begin; ikm < end; ikm++) {
      if (nonzero_fs.at(ikm)) {
        local_rows.push_back(ikm);
        local_vals.push_back(collision_prod_elem(K_ctx->disorder_term, K_ctx->delta, K_ctx->Ekm,
              K_ctx->sorted_Ekm, K_ctx->ikm_to_sorted, n_all, ikm));
      }
    }

    #pragma omp critical
    {
      ierr = VecSetValues(result, local_rows.size(), local_rows.data(), local_vals.data(),
          INSERT_VALUES);CHKERRXX(ierr);
    }
  }

  ierr = VecAssemblyBegin(result);CHKERRXX(ierr);
  ierr = VecAssemblyEnd(result);CHKERRXX(ierr);

  return 0;
}

/** @brief Construct the collision matrix: hbar K.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param H Class instance giving the Hamiltonian of the system. Should have the methods
 *           double energy(kmComps<dim>)
 *           and
 *           std::complex<double> basis_component(ikm, i).
 *  @param disorder_term A function with signature
 *                       double f(ikm1, ikm2)
 *                       giving the disorder-averaged term
 *                       U_{ikm1, ikm2} U_{ikm2, ikm1}.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU, typename Delta>
std::pair<OwnedMat, CollisionContext<UU, Delta>> make_collision(const kmBasis<k_dim>& kmb, const Hamiltonian& H,
    const UU& disorder_term, const Delta& delta) {
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

  Mat K = nullptr;
  CollisionContext<UU, Delta> K_ctx(disorder_term, delta, Ekm_all_std, sorted_Ekm, ikm_to_sorted);
  auto K_shell = std::make_pair(OwnedMat(K), K_ctx);

  PetscErrorCode ierr = MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, kmb.end_ikm, kmb.end_ikm,
      &(K_shell.second), &(K_shell.first.M));CHKERRXX(ierr);
  ierr = MatShellSetOperation(K_shell.first.M, MATOP_MULT,
      reinterpret_cast<void (*)()>(&apply_collision_shell<UU, Delta>));CHKERRXX(ierr);

  return K_shell;
}

/** @brief Compute the value of [K^{dd} <n>]_k^{mm}, where (k, m) <-> ikm.
 */
template <typename UU, typename Delta>
PetscScalar collision_prod_elem(const UU& disorder_term, const Delta& delta,
    const std::vector<PetscReal>& Ekm, const SortResult& sorted_Ekm,
    const std::vector<PetscInt>& ikm_to_sorted, const std::vector<PetscScalar>& n_all,
    const PetscInt ikm) {
  PetscScalar total = 0.0;

  auto update_total = [ikm, &total, &disorder_term, &delta, &Ekm, &n_all](PetscInt ikpmp) {
    std::complex<double> U_part = disorder_term(ikm, ikpmp);
    PetscReal ndiff = n_all.at(ikm).real() - n_all.at(ikpmp).real(); // TODO remove real(), keep im part?
    PetscReal delta_factor = delta(Ekm.at(ikm), Ekm.at(ikpmp));

    // TODO - use Kahan sum.
    total += 2*pi * U_part * ndiff * delta_factor;
  };

  apply_on_fermi_surface(delta, sorted_Ekm, ikm_to_sorted, ikm, update_total);

  return total;
}

/** @brief Iterate over the indices ikpmp of the (k', m') points which are on the
 *         Fermi surface associated with the point ikm and call the function fs_fn(ikpmp)
 *         for each of these points.
 *  @param fs_fn A function with the signature `void fs_fn(PetscInt ikpmp)`.
 *  @note Points (k', m') on the Fermi surface of point (k, m) are given by those with
 *        `|\delta(E_{km}, E_{k'm'})| > delta.threshold`.
 */
template <typename Delta, typename F>
void apply_on_fermi_surface(const Delta& delta, const SortResult& sorted_Ekm,
    const std::vector<PetscInt>& ikm_to_sorted, PetscInt ikm, F fs_fn) {
  PetscInt end_ikm = sorted_Ekm.size();
  PetscInt sorted_ikm_index = ikm_to_sorted.at(ikm);
  PetscReal E_km = sorted_Ekm.at(sorted_ikm_index).first;

  // We will iterate through the list of sorted energies, moving up and down away from ikm.
  // If the highest or lowest energy is reached, the up and down iterations will
  // wrap around and continue.
  // `end_up` and `end_down` are 1 + the maximum number of elements which may be
  // encountered in this way.
  PetscInt end_up = static_cast<PetscInt>(std::floor(end_ikm/2.0) + 1);
  PetscInt end_down = static_cast<PetscInt>(std::ceil(end_ikm/2.0));

  // Iterate through columns in sorted order, moving up in energy from ikm.
  for (PetscInt di = 1; di < end_up; di++) {
    PetscInt sorted_ikpmp_index = wrap(sorted_ikm_index + di, end_ikm);
    PetscReal E_kpmp = sorted_Ekm.at(sorted_ikpmp_index).first;

    bool on_fermi_surface = std::abs(delta(E_km, E_kpmp)) > delta.threshold;

    // Are we still on the Fermi surface? If so, call fs_fn.
    if (on_fermi_surface) {
      PetscInt ikpmp = sorted_Ekm.at(sorted_ikpmp_index).second;
      fs_fn(ikpmp);
    } else {
      // No longer on Fermi surface: we can stop moving up in energy.
      break;
    }
  }

  // Iterate through columns in sorted order, moving down in energy from row.
  for (PetscInt di = 1; di < end_down; di++) {
    PetscInt sorted_ikpmp_index = wrap(sorted_ikm_index - di, end_ikm);
    PetscReal E_kpmp = sorted_Ekm.at(sorted_ikpmp_index).first;

    bool on_fermi_surface = std::abs(delta(E_km, E_kpmp)) > delta.threshold;

    // Are we still on the Fermi surface? If so, call fs_fn.
    if (on_fermi_surface) {
      PetscInt ikpmp = sorted_Ekm.at(sorted_ikpmp_index).second;
      fs_fn(ikpmp);
    } else {
      // No longer on Fermi surface: we can stop moving down in energy.
      break;
    }
  }
}

} // namespace anomtrans

#endif // ANOMTRANS_COLLISION_H
