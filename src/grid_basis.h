#ifndef ANOMTRANS_GRID_BASIS_H
#define ANOMTRANS_GRID_BASIS_H

#include <cstddef>
#include <array>
#include <tuple>
#include <petscksp.h>

namespace anomtrans {

namespace {

template <std::size_t ncomp>
std::array<PetscInt, ncomp> get_coeffs(std::array<unsigned int, ncomp> sizes) {
  std::array<PetscInt, ncomp> coeffs;
  for (std::size_t d = 0; d < ncomp; d++) {
    PetscInt coeff = 1;
    for (std::size_t dc = 0; dc < d; dc++) {
      coeff *= sizes.at(dc);
    }
    coeffs.at(d) = coeff;
  }
  return coeffs;
}

template <std::size_t ncomp>
PetscInt get_end_iall(std::array<unsigned int, ncomp> sizes) {
  PetscInt end_iall = 1;
  for (std::size_t d = 0; d < ncomp; d++) {
    end_iall *= sizes.at(d);
  }
  return end_iall;
}

} // namespace

/** @brief Provides translation of a composite index (represented as an array)
 *         into an element of a linear sequence, as well as the reverse process.
 */
template <std::size_t ncomp>
class GridBasis {
  // This class doesn't make sense for ncomp = 0.
  static_assert(ncomp > 0, "GridBasis must have at least one component");

  /** @brief Size of the basis in each dimension.
   */
  std::array<unsigned int, ncomp> sizes;
  /** @note Precompute compose() coefficients so we don't have to compute on
   *        every call. Coefficients are always the same for given sizes.
   */ 
  std::array<PetscInt, ncomp> coeffs;

public:
  const PetscInt end_iall;

  GridBasis(std::array<unsigned int, ncomp> _sizes)
      : sizes(_sizes), coeffs(get_coeffs(_sizes)), end_iall(get_end_iall(_sizes)) {}

  /** @brief Convert a linear sequence index `iall` into the corresponding
   *         composite index.
   */
  std::array<unsigned int, ncomp> decompose(PetscInt iall) {
    std::array<unsigned int, ncomp> comps;
    // Safe to access elem 0 here due to static_assert.
    comps.at(0) = iall % sizes.at(0);

    unsigned int prev = iall;
    for (std::size_t d = 1; d < ncomp; d++) {
      unsigned int new_residual = ((prev - comps.at(d-1)) / sizes.at(d-1));
      comps.at(d) = new_residual % sizes.at(d);
      prev = new_residual;
    }

    return comps;
  }

  /** @brief Convert a composite index `components` into the corresponding
   *         linear sequence index.
   */
  PetscInt compose(std::array<unsigned int, ncomp> components) {
    PetscInt total = 0;
    for (std::size_t d = 0; d < ncomp; d++) {
      total += coeffs.at(d) * components.at(d);
    }
    return total;
  }

  /** @brief Given a linear sequence index `iall` and a composite sequence index
   *         difference `Delta`, return the linear sequence index corresponding to
   *           decompose(iall) + Delta
   *         where components are allowed to wrap around their boundaries.
   */
  PetscInt add(PetscInt iall, std::array<int, ncomp> Delta) {
    auto comps = decompose(iall);
    std::array<unsigned int, ncomp> new_comps;
    for (std::size_t d = 0; d < ncomp; d++) {
      new_comps.at(d) = (comps.at(d) + Delta.at(d)) % sizes.at(d);
    }
    return compose(new_comps);
  }
};

template <std::size_t dim>
using kComps = std::array<unsigned int, dim>;

template <std::size_t dim>
using dkComps = std::array<int, dim>;

template <std::size_t dim>
using kmComps = std::tuple<kComps<dim>, unsigned int>;

template <std::size_t dim>
using kVals = std::array<double, dim>;

template <std::size_t dim>
using kmVals = std::tuple<kVals<dim>, unsigned int>;

template <std::size_t dim>
using DimMatrix = std::array<std::array<double, dim>, dim>;

namespace {

template <std::size_t dim>
GridBasis<dim+1> corresponding_GridBasis(kComps<dim> Nk, unsigned int Nbands) {
  std::array<unsigned int, dim+1> sizes;
  for (std::size_t d = 0; d < dim; d++) {
    sizes.at(d) = Nk.at(d);
  }
  sizes.at(dim) = Nbands;
  return GridBasis<dim+1>(sizes);
}

} // namespace

/** @brief Provides translation of the composite (ik, m) index into an element
 *         of a linear sequence, as well as the reverse process.
 */
template <std::size_t dim>
class kmBasis {
  // This class doesn't make sense for dim = 0.
  static_assert(dim > 0, "kmBasis must have spatial dimension > 0");

  // Note that since members are initialized in declaration order, this
  // declaration must come before the declaration of end_ikm.
  GridBasis<dim+1> gb;

public:
  const kComps<dim> Nk;
  const unsigned int Nbands;
  const PetscInt end_ikm;

  kmBasis(kComps<dim> _Nk, unsigned int _Nbands)
      : gb(corresponding_GridBasis(_Nk, _Nbands)), Nk(_Nk), Nbands(_Nbands),
        end_ikm(gb.end_iall) {}

  /** @brief Convert a linear sequence index `ikm` into the corresponding
   *         composite index (ik, m).
   */
  kmComps<dim> decompose(PetscInt ikm) {
    auto all_comps = gb.decompose(ikm);
    kComps<dim> iks;
    for (std::size_t d = 0; d < dim; d++) {
      iks.at(d) = all_comps.at(d);
    }
    unsigned int im = all_comps.at(dim);
    return kmComps<dim>(iks, im);
  }

  /** @brief Convert a composite index `ikm_comps` = (ik, m) into the
   *         corresponding linear sequence index.
   */
  PetscInt compose(kmComps<dim> ikm_comps) {
    std::array<unsigned int, dim+1> all_comps;
    for (std::size_t d = 0; d < dim; d++) {
      all_comps.at(d) = std::get<0>(ikm_comps).at(d);
    }
    all_comps.at(dim) = std::get<1>(ikm_comps);
    return gb.compose(all_comps);
  }

  /** @brief Given a linear sequence index (`ikm`) and the k part of a composite
   *         sequence index difference (`Delta_k`), return the linear sequence
   *         index corresponding to
   *           decompose(ikm) + Delta_k
   *         where k components are allowed to wrap around their boundaries
   *         (i.e. k-space periodicity is respected).
   */
  PetscInt add(PetscInt ikm, dkComps<dim> Delta_k) {
    std::array<int, dim+1> Delta_km;
    for (std::size_t d = 0; d < dim; d++) {
      Delta_km.at(d) = Delta_k.at(d);
    }
    Delta_km.at(dim) = 0;
    return gb.add(ikm, Delta_km);
  }
};

/** @brief Given a composite (ik, m) index `ikm_comps` and the number of k-points
 *         in each direction `Nk`, return the corresponding (k, m) value (where
 *         k is a point in reciprocal lattice coordinates).
 */
template <std::size_t dim>
kmVals<dim> km_at(kComps<dim> Nk, kmComps<dim> ikm_comps) {
  kVals<dim> ks;
  for (std::size_t d = 0; d < dim; d++) {
    ks.at(d) = std::get<0>(ikm_comps).at(d) / static_cast<double>(Nk.at(d));
  }
  kmVals<dim> km(ks, std::get<1>(ikm_comps));
  return km;
}

} // namespace anomtrans

#endif // ANOMTRANS_GRID_BASIS_H
