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
 *         at which `abs(n_all.at(ikm)) > threshold`. Return a vector where the entries are
 *         `true` for `ikm` on such Fermi surfaces, and `false` otherwise.
 *  @param delta Delta function representation `delta(E1, E2)`. Must satisfy the following
 *               conditions:
 *               (1) `\delta(E1, E2) = \delta(E2, E1)`;
 *               (2) `\delta(E, E) != 0`;
 *               (3) For fixed `E1 <= E2`, `|\delta(E1, E2)|` is non-increasing with
 *               increasing `E2`.
 *               (4) `\delta(E1, E2)` is normalized such that
 *               `\int_{-\infty}^{\infty} dE1 \delta(E1, E2) = 1`
 *               and there exists a limit such that
 *               `\int_{-\infty}^{\infty} dE1 \delta(E1, E2) f(E1) = f(E2)`.
 *               `delta` should also have member `delta.threshold` specifying the value below
 *               which it is considered to be vanishing.
 *  @param Ekm Sorted vector of E values over all `ikm`.
 *  @param n_all Vector of <n> values over all `ikm`.
 *  @param n_threshold Value below which `|n|` is considered to be vanishing.
 *  @note This function requires only O(`n_all.size()`) operations and memory.
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

  PetscInt low = 0; // `L`
  PetscInt high = 0; // `H`
  PetscInt n_highest_seen = -1; // `N`

  // Special case: first point. Check n(0).
  if (std::abs(n_all.at(Ekm.at(high).second)) > n_threshold) {
    n_highest_seen = 0;
    nonzero_fs.at(Ekm.at(0).second) = true;
  }

  // General case: L <= H; all n values <= H have been examined.
  while (high < static_cast<PetscInt>(n_all.size()) - 1) {
    // Advance H.
    high++;

    // Is delta(L, H) finite? If not, advance L until delta(L, H) is finite.
    // N is less than H. As long as L <= N, L is on an active Fermi surface.
    while (std::abs(delta(Ekm.at(low).first, Ekm.at(high).first)) < delta.threshold) {
      if (low <= n_highest_seen) {
        nonzero_fs.at(Ekm.at(low).second) = true;
      }
      low++;
    }

    // Is n(H) finite? If so, advance N to H.
    if (std::abs(n_all.at(Ekm.at(high).second)) > n_threshold) {
      n_highest_seen = high;
    }

    // Is L <= N? If so, L belongs to an active Fermi surface.
    if (low <= n_highest_seen) {
      nonzero_fs.at(Ekm.at(low).second) = true;
    }
  }

  // Special case: final point. If L <= N, all remaining points are on an active Fermi surface.
  if (low <= n_highest_seen) {
    while (low < static_cast<PetscInt>(n_all.size())) {
      nonzero_fs.at(Ekm.at(low).second) = true;
      low++;
    }
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
