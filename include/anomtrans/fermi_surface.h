#ifndef ANOMTRANS_FERMI_SURFACE_H
#define ANOMTRANS_FERMI_SURFACE_H

#include <cmath>
#include <limits>
#include <exception>
#include <vector>
#include <boost/container/vector.hpp>
#include <petscksp.h>
#include "util/constants.h"
#include "observables/energy.h"

namespace anomtrans {

/** @brief Compute a reasonable threshold for considering `n` nonzero.
 *         Since we do not have a general absolute scale for `n`, this threshold
 *         is based on the maximum `|n|` value in `n_all`.
 */
PetscReal get_n_threshold(const std::vector<PetscScalar> &n_all);

/** @brief Iterate over `ikm` values to find those Fermi surfaces which have at least one value
 *         at which `abs(n_all.at(ikm)) > n_threshold`. Return a vector where the entries are
 *         `true` for `ikm` on such Fermi surfaces, and `false` otherwise.
 *         The "Fermi surface" associated with point `ikm` is defined here as the set of points
 *         `ikpmp` on which `\delta(ikm, ikmpm) > delta.threshold`.
 *  @param delta Delta function representation `delta(E1, E2)`. Must satisfy the following
 *               conditions:
 *               (1) `\delta(E1, E2) = \delta(E2, E1)`;
 *               (2) `\delta(E, E) != 0`;
 *               (3) `\delta(E1, E2) >= 0`;
 *               (4) For fixed `E1 <= E2`, `\delta(E1, E2)` is non-increasing with
 *               increasing `E2`.
 *               (5) `\delta(E1, E2)` is normalized such that
 *               `\int_{-\infty}^{\infty} dE1 \delta(E1, E2) = 1`
 *               and there exists a limit such that
 *               `\int_{-\infty}^{\infty} dE1 \delta(E1, E2) f(E1) = f(E2)`.
 *               `delta` should also have member `delta.threshold` specifying the value below
 *               which it is considered to be vanishing.
 *  @param Ekm Sorted vector of E values over all `ikm`.
 *  @param n_all Vector of <n> values over all `ikm`.
 *  @param n_threshold Value below which `|n|` is considered to be vanishing.
 *  @note This function requires only O(`n_all.size()`) operations and memory.
 *  @note TODO - parallelization strategy. Each thread starts `center` in a different place,
 *        evenly distributed over the set of sorted indices.
 *        `center` of the i'th thread occupies an interval [c^{begin}_i, c^{end}_i),
 *        with c^{end}_i = c^{begin}_{i+1}, c^{begin}_{0} = 0,
 *        and c^{end}_{N_threads - 1} = n_all.size().
 *        `center` of the i'th thread initially takes the value c^{begin}_i, and iteration
 *        on that thread stops when that `center` takes the value c^{end}_i.
 */
template <typename Delta>
boost::container::vector<bool> find_nonzero_fs(const Delta& delta, const SortResult& Ekm,
    const std::vector<PetscScalar> &n_all, PetscReal n_threshold) {
  if (n_all.size() != Ekm.size()) {
    throw std::invalid_argument("must have equal size of n_all and Ekm");
  }
  if (n_threshold <= 0.0 or delta.threshold <= 0.0) {
    throw std::invalid_argument("must have n_threshold > 0 and delta.threshold > 0");
  }

  boost::container::vector<bool> nonzero_fs(n_all.size(), false);

  if (n_all.size() == 0) {
    return nonzero_fs;
  }
  PetscInt end = n_all.size();

  // Indices in the sorted sequence of energies, defining Fermi surface intervals.
  // The algorithm is defined such that each of these indices never takes the same value
  // twice, ensuring that `find_nonzero_fs` executes in time linearly proportional
  // to `n_all.size()`.
  PetscInt low = 0; // `L`
  PetscInt high = 0; // `H`
  PetscInt center = 0; // `C`

  // Initialization: find the first subinterval, in which `|n(C)| > n_threshold`,
  // `\delta(E(L), E(C)) > delta.threshold` and either `L = 0` or
  // `\delta(E(L-1), E(C)) < delta.threshold`, and `\delta(E(C), E(H)) > delta.threshold`
  // and either `H = n_all.size() - 1` or `\delta(E(C), E(H+1)) < delta.threshold`.
  while (std::abs(n_all.at(Ekm.at(center).second)) < n_threshold and center < end) {
    center++;
  }
  if (center == end) {
    return nonzero_fs;
  }

  nonzero_fs.at(Ekm.at(center).second) = true;
  low = center;
  high = center;
  while (low > 0
      and std::abs(delta(Ekm.at(low - 1).first, Ekm.at(center).first)) >= delta.threshold) {
    low--;
    nonzero_fs.at(Ekm.at(low).second) = true;
  }
  while (high < end - 1
      and std::abs(delta(Ekm.at(center).first, Ekm.at(high + 1).first)) >= delta.threshold) {
    high++;
    nonzero_fs.at(Ekm.at(high).second) = true;
  }
  //PetscPrintf(PETSC_COMM_WORLD, "init complete, low = %d, center = %d, high = %d\n", low, center, high);

  // Now we have a subinterval `[low, high]` with `low <= center <= high` and all points in the interval
  // `[low, high]` are marked as being on an occupied Fermi surface.
  // Now advance `center` util we find the next subinterval, defined in the same way as in
  // the initialization step.
  while (high < end - 1) {
    PetscInt high_prev = high; // `H_0`

    center++;
    while (center < end and std::abs(n_all.at(Ekm.at(center).second)) < n_threshold) {
      center++;
    }
    if (center == end) {
      return nonzero_fs;
    }

    // TODO: can jump to `low = center` and move backwards instead if `center > high_prev`
    // (since `\delta(E(C), E(H_0 - 1)) < delta.threshold` in this case).
    while (std::abs(delta(Ekm.at(low).first, Ekm.at(center).first)) < delta.threshold) {
      low++;
    }

    // Jump `high` ahead if center has passed it, so that moving it forward always moves it away from `center`.
    if (high < center) {
      high = center;
    }
    while (high < end - 1
        and std::abs(delta(Ekm.at(center).first, Ekm.at(high + 1).first)) >= delta.threshold) {
      high++;
    }

    // Mark the points in `[low, high]` which do not overlap with the previous subinterval.
    // If `low > high_prev`, this is the full interval `[low, high]`.
    // Otherwise this is the interval `(high_prev, high]` (which is empty if `high = high_prev`).
    if (low > high_prev) {
      for (PetscInt i = low; i <= high; i++) {
        nonzero_fs.at(Ekm.at(i).second) = true;
      }
    } else {
      for (PetscInt i = high_prev + 1; i <= high; i++) {
        nonzero_fs.at(Ekm.at(i).second) = true;
      }
    }

    //PetscPrintf(PETSC_COMM_WORLD, "advance complete, low = %d, center = %d, high = %d\n", low, center, high);
  }

  return nonzero_fs;
}

class DeltaBin {
  static PetscReal get_width(const SortResult &sorted_Ekm, unsigned int num_fs);

  /** @brief The index of the Fermi surface bin for energy E.
   */
  PetscInt get_bin(PetscReal E) const;

public:
  /** @brief The number of Fermi surface bins.
   */
  const std::size_t num_fs;

  /** @brief The width in energy of each Fermi surface bin.
   */
  const PetscReal width;

  /** @brief The highest and lowest energies on which the bins are defined.
   */
  const PetscReal E_min, E_max;

  /** @brief Value below which `operator()` delta function outputs can be considered 0.
   *         In this case, since our output is either 0 or 1/(bin width), any value
   *         below 1/(bin width) is suitable.
   */
  const PetscReal threshold;

  /** @brief Delta function representation in which energies are distributed over `_num_fs`
   *         equally-sized bins between the minimum and maximum energies in `sorted_Ekm`.
   *         Pairs of energies have `\delta(E1, E2) = 1/(bin width)` when `E1` and `E2` are
   *         in the same bin and 0 otherwise.
   */
  DeltaBin(const SortResult &sorted_Ekm, unsigned int _num_fs)
      : num_fs(_num_fs), width(get_width(sorted_Ekm, _num_fs)),
        E_min(sorted_Ekm.at(0).first), E_max(sorted_Ekm.at(sorted_Ekm.size() - 1).first),
        threshold(0.1 / width) {}

  PetscReal operator()(PetscReal E1, PetscReal E2) const;
};

class DeltaGaussian {
  static PetscReal get_threshold(PetscReal sigma);

public:
  static PetscReal get_sigma_min(PetscReal max_energy_difference);

  /** @brief Standard deviation of the Gaussian.
   */
  const PetscReal sigma;

  /** @brief Value below which `operator()` delta function outputs can be considered 0.
   */
  const PetscReal threshold;

  /** @brief Gaussian delta function representation.
   */
  DeltaGaussian(PetscReal _sigma) : sigma(_sigma), threshold(get_threshold(_sigma)) {}

  PetscReal operator()(PetscReal E1, PetscReal E2) const;
};

} // namespace anomtrans

#endif // ANOMTRANS_FERMI_SURFACE_H
