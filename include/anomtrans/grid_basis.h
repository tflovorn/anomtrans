#ifndef ANOMTRANS_GRID_BASIS_H
#define ANOMTRANS_GRID_BASIS_H

#include <cassert>
#include <cstddef>
#include <array>
#include <tuple>
#include <petscksp.h>
#include "util/vec.h"
#include "util/mat.h"

namespace anomtrans {

/** @brief Provides translation of a composite index (represented as an array)
 *         into an element of a linear sequence, as well as the reverse process.
 *  @todo Could use constexpr if to implement member functions.
 */
template <std::size_t ncomp>
class GridBasis {
  // This class doesn't make sense for ncomp = 0.
  static_assert(ncomp > 0, "GridBasis must have at least one component");

  /** @brief Get coefficients to use in GridBasis::compose() for moving from a
   *         composite grid coordinate to an integer grid index.
   */
  static std::array<PetscInt, ncomp> get_coeffs(std::array<unsigned int, ncomp> sizes) {
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

  /** @brief Get index which is one past the final grid index; i.e. the index end_iall
   *         suitable for use in:
   *         for (PetscInt i = 0; i < end_iall; i++).
   */
  static PetscInt get_end_iall(std::array<unsigned int, ncomp> sizes) {
    PetscInt end_iall = 1;
    for (std::size_t d = 0; d < ncomp; d++) {
      end_iall *= sizes.at(d);
    }
    return end_iall;
  }

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
  std::array<unsigned int, ncomp> decompose(PetscInt iall) const {
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
  PetscInt compose(std::array<unsigned int, ncomp> components) const {
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
  PetscInt add(PetscInt iall, std::array<int, ncomp> Delta) const {
    auto comps = decompose(iall);
    std::array<unsigned int, ncomp> new_comps;
    for (std::size_t d = 0; d < ncomp; d++) {
      new_comps.at(d) = (comps.at(d) + Delta.at(d)) % sizes.at(d);
    }
    return compose(new_comps);
  }
};

/** @brief Components of a k-point, specified by integer indices between
 *         0 and (Nk - 1) in each reciprocal lattice coordinate direction.
 */
template <std::size_t dim>
using kComps = std::array<unsigned int, dim>;

/** @brief Difference between the kComps of two k-points.
 */
template <std::size_t dim>
using dkComps = std::array<int, dim>;

/** @brief A (k-point, band) pair which identifies one point in the kmBasis.
 */
template <std::size_t dim>
using kmComps = std::tuple<kComps<dim>, unsigned int>;

/** @brief Components of a k-point in real coordinates (reciprocal lattice
 *         or Cartesian).
 */
template <std::size_t dim>
using kVals = std::array<double, dim>;

/** @brief A (k-point, band) pair with the k-point specified in real coordinates
 *         (reciprocal lattice or Cartesian).
 */
template <std::size_t dim>
using kmVals = std::tuple<kVals<dim>, unsigned int>;

/** @brief Provides translation of the composite (ik, m) index into an element
 *         of a linear sequence, as well as the reverse process.
 *  @todo Could use constexpr if to implement member functions.
 */
template <std::size_t dim>
class kmBasis {
  // This class doesn't make sense for dim = 0.
  static_assert(dim > 0, "kmBasis must have spatial dimension > 0");

  /** @brief Constuct the GridBasis with sizes = (Nk(0), Nk(1), ..., Nbands).
   *  @todo Could use constexpr if to handle dim = 1, 2, 3.
   */
  static GridBasis<dim+1> corresponding_GridBasis(kComps<dim> Nk, unsigned int Nbands) {
    std::array<unsigned int, dim+1> sizes;
    for (std::size_t d = 0; d < dim; d++) {
      sizes.at(d) = Nk.at(d);
    }
    sizes.at(dim) = Nbands;
    return GridBasis<dim+1>(sizes);
  }

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
  kmComps<dim> decompose(PetscInt ikm) const {
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
  PetscInt compose(kmComps<dim> ikm_comps) const {
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
  PetscInt add(PetscInt ikm, dkComps<dim> Delta_k) const {
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

/** @brief Construct an array of k-diagonal matrices <km|A_i|km'>, where the elements are given
 *         by the function `f` with the signature
 *         std::array<PetscScalar, out_dim> f(PetscInt ikm, unsigned int mp)
 *         where the arguments specify km and m' and the elements of the output correspond to i.
 *  @note Placed here to avoid mat.h dependency on grid_basis.h. Prefer to place in mat instead?
 *  @todo Should use MATMPIBAIJ (block matrix) for this to avoid a lot of overhead.
 *        May need to change kmBasis ordering to support this - go from (k moves the fastest)
 *        to (m moves the fastest), i.e. keep elements with the same k-point together in blocks.
 *        This could also help to support caching strategies for eigenvectors etc.
 */
template <std::size_t out_dim, std::size_t k_dim, typename ElemFunc>
std::array<Mat, out_dim> construct_k_diagonal_Mat_array(kmBasis<k_dim> kmb, ElemFunc f) {
  static_assert(out_dim > 0, "must have at least one output element");

  std::array<Mat, out_dim> As;
  for (std::size_t d = 0; d < out_dim; d++) {
    As.at(d) = make_Mat(kmb.end_ikm, kmb.end_ikm, kmb.Nbands);
  }

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(As.at(0), &begin, &end);CHKERRXX(ierr);

  for (PetscInt ikm = begin; ikm < end; ikm++) {
    std::vector<PetscInt> row_cols;
    std::array<std::vector<PetscScalar>, out_dim> row_vals;
    row_cols.reserve(kmb.Nbands);

    for (std::size_t d = 0; d < out_dim; d++) {
      row_vals.at(d).reserve(kmb.Nbands);
    }

    auto k = std::get<0>(kmb.decompose(ikm));

    for (unsigned int mp = 0; mp < kmb.Nbands; mp++) {
      PetscInt ikmp = kmb.compose(std::make_tuple(k, mp));
      row_cols.push_back(ikmp);

      auto elems = f(ikm, mp);

      for (std::size_t d = 0; d < out_dim; d++) {
        row_vals.at(d).push_back(elems.at(d));
      }
    }

    for (std::size_t d = 0; d < out_dim; d++) {
      assert(row_cols.size() == row_vals.at(d).size());
      ierr = MatSetValues(As.at(d), 1, &ikm, row_cols.size(), row_cols.data(), row_vals.at(d).data(),
          INSERT_VALUES);CHKERRXX(ierr);
    }
  }

  for (std::size_t d = 0; d < out_dim; d++) {
    ierr = MatAssemblyBegin(As.at(d), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    ierr = MatAssemblyEnd(As.at(d), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  }

  return As;
}

/** @brief Construct a k-diagonal matrix <km|A|km'>, where the elements are given
 *         by the function `f` with the signature
 *         PetscScalar f(PetscInt ikm, unsigned int mp)
 *         where the arguments specify km and m'.
 *  @note Placed here to avoid mat.h dependency on grid_basis.h. Prefer to place in mat instead?
 *  @todo Generalize to `f` which returns an array of PetscScalars, in the same manner
 *        as vector_index_apply_multiple().
 *  @todo Should use MATMPIBAIJ (block matrix) for this to avoid a lot of overhead.
 *        May need to change kmBasis ordering to support this - go from (k moves the fastest)
 *        to (m moves the fastest), i.e. keep elements with the same k-point together in blocks.
 *        This could also help to support caching strategies for eigenvectors etc.
 */
template <std::size_t k_dim, typename ElemFunc>
Mat construct_k_diagonal_Mat(kmBasis<k_dim> kmb, ElemFunc f) {
  auto f_array = [&f](PetscInt ikm, unsigned int mp)->std::array<PetscScalar, 1> {
    return {f(ikm, mp)};
  };

  return construct_k_diagonal_Mat_array(kmb, f).at(0);
}

/** @brief Collect the elements <km|S|km'> for all k onto a vector on rank 0.
 *         Analogous to vec.h -> collect_contents(), but for k-diagonal matrices
 *         and with fixed band indices.
 *  @note Placed here to avoid mat.h dependency on grid_basis.h. Prefer to place in mat instead?
 */
template <std::size_t k_dim>
std::vector<PetscScalar> collect_band_elem(kmBasis<k_dim> kmb, Mat S,
    unsigned int m, unsigned int mp) {
  // Check that kmb has expected properties:
  // number of total points is divisible by number of bands;
  // points are ordered with all ks for one band first, then all ks for the next band, ...
  assert(kmb.end_ikm % kmb.Nbands == 0);
  kComps<k_dim> Nk_m1;
  PetscInt Nk_tot = 1;
  for (std::size_t d = 0; d < k_dim; d++) {
    Nk_m1.at(d) = kmb.Nk.at(d) - 1;
    Nk_tot *= kmb.Nk.at(d);
  }
  assert(kmb.compose(std::make_tuple(Nk_m1, 0u)) == Nk_tot - 1);

  // Construct a vector to hold the result: one value for each k.
  Vec S_mmp;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE,
      kmb.end_ikm / kmb.Nbands, &S_mmp);CHKERRXX(ierr);

  PetscInt begin, end;
  ierr = MatGetOwnershipRange(S, &begin, &end);CHKERRXX(ierr);

  std::vector<PetscInt> local_rows;
  std::vector<PetscScalar> local_values;

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    // Are we on a row with band index m?
    kComps<k_dim> this_k;
    unsigned int this_m;
    std::tie(this_k, this_m) = kmb.decompose(local_row);

    if (this_m != m) {
      // Not on row with band index m - skip this row.
      continue;
    }

    // On row with band index m - process this row.
    // S_mmp index includes only one band value, (m, mp).
    local_rows.push_back(kmb.compose(std::make_tuple(this_k, 0u)));

    kmComps<k_dim> kmp = std::make_tuple(this_k, mp);
    PetscInt ikmp = kmb.compose(kmp);

    PetscScalar value;
    ierr = MatGetValues(S, 1, &local_row, 1, &ikmp, &value);CHKERRXX(ierr);

    local_values.push_back(value);
  }

  assert(local_rows.size() == local_values.size());
  ierr = VecSetValues(S_mmp, local_rows.size(), local_rows.data(), local_values.data(),
      INSERT_VALUES);CHKERRXX(ierr);

  ierr = VecAssemblyBegin(S_mmp);CHKERRXX(ierr);
  ierr = VecAssemblyEnd(S_mmp);CHKERRXX(ierr);

  return collect_contents(S_mmp);
}

} // namespace anomtrans

#endif // ANOMTRANS_GRID_BASIS_H
