#ifndef ANOMTRANS_GRID_BASIS_H
#define ANOMTRANS_GRID_BASIS_H

#include <cstddef>
#include <array>
#include <tuple>
#include "dist_vec.h"

namespace anomtrans {

namespace {

template <std::size_t ncomp>
std::array<GO, ncomp> get_coeffs(std::array<unsigned int, ncomp> sizes) {
  std::array<GO, ncomp> coeffs;
  for (std::size_t d = 0; d < ncomp; d++) {
    GO coeff = 1;
    for (std::size_t dc = 0; dc < d; dc++) {
      coeff *= sizes.at(dc);
    }
    coeffs.at(d) = coeff;
  }
  return coeffs;
}

template <std::size_t ncomp>
GO get_end_iall(std::array<unsigned int, ncomp> sizes) {
  GO end_iall = 1;
  for (std::size_t d = 0; d < ncomp; d++) {
    end_iall *= sizes.at(d);
  }
  return end_iall;
}

} // namespace

template <std::size_t ncomp>
class GridBasis {
  // This class doesn't make sense for ncomp = 0.
  static_assert(ncomp > 0, "GridBasis must have at least one component");

  /* @brief Size of the basis in each dimension.
   */
  std::array<unsigned int, ncomp> sizes;
  /* @note Precompute compose() coefficients so we don't have to compute on
   *       every call. Coefficients are always the same for given sizes.
   */ 
  std::array<GO, ncomp> coeffs;

  /* @brief Teuchos Map corresponding to this basis.
   * @note This is included inside GridBasis to ensure consistency between
   *       the map's numGlobalElements and indexBase and the values that are
   *       assumed here: see the invariant for this entry.
   * @invariant This must be zero-indexed and have number of global entries
   *            equal to get_end_iall(sizes).
   */
  RCP<const Map> map;

public:
  const GO end_iall;

  GridBasis(std::array<unsigned int, ncomp> _sizes, MPIComm _comm)
      : sizes(_sizes), end_iall(get_end_iall(_sizes)), coeffs(get_coeffs(_sizes)),
        map(RCP<Map>(new Map(end_iall, 0, _comm))) {}

  std::array<unsigned int, ncomp> decompose(GO iall) {
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

  GO compose(std::array<unsigned int, ncomp> components) {
    GO total = 0;
    for (std::size_t d = 0; d < ncomp; d++) {
      total += coeffs.at(d) * components.at(d);
    }
    return total;
  }

  GO add(GO iall, std::array<int, ncomp> Delta) {
    auto comps = decompose(iall);
    std::array<unsigned int, ncomp> new_comps;
    for (std::size_t d = 0; d < ncomp; d++) {
      new_comps.at(d) = (comps.at(d) + Delta.at(d)) % sizes.at(d);
    }
    return compose(new_comps);
  }

  RCP<const Map> get_map() {
    return map;
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

namespace {

template <std::size_t dim>
GridBasis<dim+1> corresponding_GridBasis(kComps<dim> Nk, unsigned int Nbands, MPIComm comm) {
  std::array<unsigned int, dim+1> sizes;
  for (std::size_t d = 0; d < dim; d++) {
    sizes.at(d) = Nk.at(d);
  }
  sizes.at(dim) = Nbands;
  return GridBasis<dim+1>(sizes, comm);
}

} // namespace

template <std::size_t dim>
class kmBasis {
  // This class doesn't make sense for dim = 0.
  static_assert(dim > 0, "kmBasis must have spatial dimension > 0");

  kComps<dim> Nk;
  unsigned int Nbands;
  GridBasis<dim+1> gb;

public:
  const GO end_ikm;

  kmBasis(kComps<dim> _Nk, unsigned int _Nbands, MPIComm _comm)
      : Nk(_Nk), Nbands(_Nbands), gb(corresponding_GridBasis(_Nk, _Nbands, _comm)),
        end_ikm(gb.end_iall) {}

  kmComps<dim> decompose(GO ikm) {
    auto all_comps = gb.decompose(ikm);
    kComps<dim> iks;
    for (std::size_t d = 0; d < dim; d++) {
      iks.at(d) = all_comps.at(d);
    }
    unsigned int im = all_comps.at(dim);
    return kmComps<dim>(iks, im);
  }

  GO compose(kmComps<dim> ikm_comps) {
    std::array<unsigned int, dim+1> all_comps;
    for (std::size_t d = 0; d < dim; d++) {
      all_comps.at(d) = std::get<0>(ikm_comps).at(d);
    }
    all_comps.at(dim) = std::get<1>(ikm_comps);
    return gb.compose(all_comps);
  }

  GO add(GO ikm, dkComps<dim> Delta_k) {
    std::array<int, dim+1> Delta_km;
    for (std::size_t d = 0; d < dim; d++) {
      Delta_km.at(d) = Delta_k.at(d);
    }
    Delta_km.at(dim) = 0;
    return gb.add(ikm, Delta_km);
  }

  RCP<const Map> get_map() {
    return gb.get_map();
  }
};

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
