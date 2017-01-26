#ifndef ANOMTRANS_GRID_BASIS_H
#define ANOMTRANS_GRID_BASIS_H

#include <cstddef>
#include <array>
#include <tuple>
#include "dist_vec.h"

namespace anomtrans {

template <std::size_t ncomp>
std::array<GO, ncomp> get_coeffs(std::array<unsigned int, ncomp> sizes) {
  std::array<GO, ncomp> coeffs;
  for (std::size_t d = 0; d < ncomp; d++) {
    GO coeff = 1;
    for (std::size_t dc = 0; dc < d; dc++) {
      coeff *= sizes[dc];
    }
    coeffs[d] = coeff;
  }
  return coeffs;
}

template <std::size_t ncomp>
class GridBasis {
  // This class doesn't make sense for ncomp = 0.
  static_assert(ncomp > 0, "GridBasis must have at least one component");

  std::array<unsigned int, ncomp> sizes;
  // Coeffs are always the same for given sizes.
  // Precompute these so we don't have to compute on every call to compose().
  std::array<GO, ncomp> coeffs;

public:
  GridBasis(std::array<unsigned int, ncomp> _sizes)
      : sizes(_sizes), coeffs(get_coeffs(_sizes)) {}

  std::array<unsigned int, ncomp> decompose(GO iall) {
    std::array<unsigned int, ncomp> comps;
    // Safe to access elem 0 here due to static_assert.
    comps[0] = iall % sizes[0];

    unsigned int prev = iall;
    for (std::size_t d = 1; d < ncomp; d++) {
      unsigned int new_residual = ((prev - comps[d-1]) / sizes[d-1]);
      comps[d] = new_residual % sizes[d];
      prev = new_residual;
    }

    return comps;
  }

  GO compose(std::array<unsigned int, ncomp> components) {
    GO total = 0;
    for (std::size_t d = 0; d < ncomp; d++) {
      total += coeffs[d] * components[d];
    }
    return total;
  }

  GO add(GO iall, std::array<int, ncomp> Delta) {
    auto comps = decompose(iall);
    std::array<unsigned int, ncomp> new_comps;
    for (std::size_t d = 0; d < ncomp; d++) {
      new_comps[d] = (comps[d] + Delta[d]) % sizes[d];
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
GridBasis<dim+1> corresponding_GridBasis(kComps<dim> Nk, unsigned int Nbands) {
  std::array<unsigned int, dim+1> sizes;
  for (std::size_t d = 0; d < dim; d++) {
    sizes[d] = Nk[d];
  }
  sizes[dim] = Nbands;
  return GridBasis<dim+1>(sizes);
}

template <std::size_t dim>
class kmBasis {
  // This class doesn't make sense for dim = 0.
  static_assert(dim > 0, "kmBasis must have spatial dimension > 0");

  kComps<dim> Nk;
  unsigned int Nbands;
  GridBasis<dim+1> gb;

public:
  kmBasis(kComps<dim> _Nk, unsigned int _Nbands)
      : Nk(_Nk), Nbands(_Nbands), gb(corresponding_GridBasis(_Nk, _Nbands)) {}

  kmComps<dim> decompose(GO ikm) {
    auto all_comps = gb.decompose(ikm);
    kComps<dim> iks;
    for (std::size_t d = 0; d < dim; d++) {
      iks[d] = all_comps[d];
    }
    unsigned int im = all_comps[dim];
    return kmComps<dim>(iks, im);
  }

  GO compose(kmComps<dim> ikm_comps) {
    std::array<unsigned int, dim+1> all_comps;
    for (std::size_t d = 0; d < dim; d++) {
      all_comps[d] = std::get<0>(ikm_comps)[d];
    }
    all_comps[dim] = std::get<1>(ikm_comps);
    return gb.compose(all_comps);
  }

  kmVals<dim> km_at(GO ikm) {
    auto iks_m = decompose(ikm);
    kVals<dim> ks;
    for (std::size_t d = 0; d < dim; d++) {
      ks[d] = std::get<0>(iks_m)[d] / static_cast<double>(Nk[d]);
    }
    kmVals<dim> km(ks, std::get<1>(iks_m));
    return km;
  }

  GO add(GO ikm, dkComps<dim> Delta_k) {
    std::array<int, dim+1> Delta_km;
    for (std::size_t d = 0; d < dim; d++) {
      Delta_km[d] = Delta_k[d];
    }
    Delta_km[dim] = 0;
    return gb.add(ikm, Delta_km);
  }
};

} // namespace anomtrans

#endif // ANOMTRANS_GRID_BASIS_H
