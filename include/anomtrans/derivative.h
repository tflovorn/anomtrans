#ifndef ANOMTRANS_DERIVATIVE_H
#define ANOMTRANS_DERIVATIVE_H

#include <cstddef>
#include <cmath>
#include <array>
#include <vector>
#include <tuple>
#include <utility>
#include <map>
#include <stdexcept>
#include <iostream>
#include <petscksp.h>
#include "util/constants.h"
#include "grid_basis.h"
#include "util/vec.h"
#include "util/mat.h"
#include "util/lattice.h"

namespace anomtrans {

using DeltaValPairs = std::pair<std::vector<PetscInt>, std::vector<PetscScalar>>;

/** @brief Types of finite difference approximation: backwards-looking,
 *         central (outwards-looking), or forwards-looking.
 */
enum struct DerivApproxType { backward, central, forward };

/** @brief Information required to compute a finite difference approximation
 *         to the derivative in one dimension.
 *  @tparam deriv_order Order of the derivative to calculate (1st derivative,
 *                      2nd, etc.).
 *  @note deriv_order is given as a template parameter to allow distinction
 *        between derivatives of different orders within the type system:
 *        they are essentially different, i.e. frequently cannot be substituted
 *        for one another, and so should have explicitly distinct identities.
 */
template <unsigned int deriv_order>
class DerivStencil {
  static_assert(deriv_order > 0, "derivative order must be > 0");

  /** @brief Information necessary to specify the type of finite difference:
   *         derivative order, approximation type, approximation order.
   */
  using DerivSpecifier = std::tuple<unsigned int, DerivApproxType, unsigned int>;

  static DeltaValPairs get_Delta_vals(DerivApproxType approx_type,
      unsigned int approx_order) {
    const std::map<DerivSpecifier, std::vector<PetscInt>> all_Deltas_1d {
      // {1, 0} --> uses f(x + h) and f(x)
      {DerivSpecifier(1, DerivApproxType::forward, 1), {1, 0}},
      // {0, -1} --> uses f(x) and f(x - h)
      {DerivSpecifier(1, DerivApproxType::backward, 1), {0, -1}},
      // {1, -1} --> uses f(x + h) and f(x - h)
      {DerivSpecifier(1, DerivApproxType::central, 2), {1, -1}}
    };

    const std::map<DerivSpecifier, std::vector<PetscScalar>> all_vals_1d {
      // {1, -1} --> f(x + h) - f(x)
      {DerivSpecifier(1, DerivApproxType::forward, 1), {1.0, -1.0}},
      // {1, -1} --> f(x) - f(x - h)
      {DerivSpecifier(1, DerivApproxType::backward, 1), {1.0, -1.0}},
      // {1/2, -1/2} --> (1/2) f(x + h) - (1/2) f(x - h)
      {DerivSpecifier(1, DerivApproxType::central, 2), {0.5, -0.5}}
    };

    DerivSpecifier spec = std::make_tuple(deriv_order, approx_type, approx_order);

    if (all_Deltas_1d.count(spec) == 0 or all_vals_1d.count(spec) == 0) {
      throw std::invalid_argument("The given finite-difference approximation is not implemented");
    }

    return std::make_pair(all_Deltas_1d.at(spec), all_vals_1d.at(spec));
  }

public:
  const DerivApproxType approx_type;
  const unsigned int approx_order;
  const DeltaValPairs Delta_vals;

  /** @brief Create a DerivStencil with the given derivative approximation
   *         type (backward, central, forward) and approximation order.
   */
  DerivStencil(DerivApproxType _approx_type, unsigned int _approx_order)
      : approx_type(_approx_type), approx_order(_approx_order),
        Delta_vals(get_Delta_vals(_approx_type, _approx_order)) {}
};

/** @brief If kmb is not periodic, need to use an appropriate stencil at the boundary
 *         (to avoid going outside the range of kmb). Detect this and choose an alternate
 *         stencil if necessary. Return a copy of the original stencil if no change
 *         is necessary.
 *  @todo Should be able to generalize beyond `deriv_order == 1` just by adding `deriv_order`
 *        template parameter (for `stencil` and return value).
 */
template <std::size_t k_dim>
DerivStencil<1> boundary_replacement_stencil(const kmBasis<k_dim> &kmb,
    const DerivStencil<1> &stencil, PetscInt row_ikm, std::size_t deriv_dir) {
  // Logically 'uninitialized' values for these: when set, min_bad_Delta_abs will be > 0
  // and bad_Delta_sign will be +/- 1.
  PetscInt min_bad_Delta_abs = -1;
  PetscInt bad_Delta_sign = 0;

  std::vector<PetscInt> Deltas_1d = stencil.Delta_vals.first;

  // Check each Delta in the stencil to see if it is OK.
  for (std::size_t Delta_index = 0; Delta_index < Deltas_1d.size(); Delta_index++) {
    dkComps<k_dim> Delta;
    for (std::size_t d_Delta = 0; d_Delta < k_dim; d_Delta++) {
      if (d_Delta == deriv_dir) {
        Delta.at(d_Delta) = Deltas_1d.at(Delta_index);
      } else {
        Delta.at(d_Delta) = 0;
      }
    }

    auto result = kmb.add(row_ikm, Delta);

    if (not result) {
      PetscInt bad_Delta = Delta.at(deriv_dir);
      PetscInt bad_Delta_abs = std::abs(bad_Delta);

      if (min_bad_Delta_abs < 0 or bad_Delta_abs < min_bad_Delta_abs) {
        min_bad_Delta_abs = bad_Delta_abs;
        if (bad_Delta > 0) {
          bad_Delta_sign = 1;
        } else {
          bad_Delta_sign = -1;
        }
      }
    }
  }

  // If min_bad_Delta_abs is still negative, we did not encounter any
  // points where `stencil` is inappropriate.
  if (min_bad_Delta_abs < 0) {
    return stencil;
  }
  // If we get here, there are some points where `stencil` is inappropriate.
  // Choose a stencil that will work.
  // For forward, replace with backward, and vice versa.
  // For central, choose forward or backward as appropriate based on `bad_Delta_sign`.
  // TODO - could also choose a smaller central based on `min_bad_Delta_abs`.
  if (stencil.approx_type == DerivApproxType::forward) {
    return DerivStencil<1>(DerivApproxType::backward, stencil.approx_order);
  } else if (stencil.approx_type == DerivApproxType::backward) {
    return DerivStencil<1>(DerivApproxType::forward, stencil.approx_order);
  } else {
    if (bad_Delta_sign > 0) {
      return DerivStencil<1>(DerivApproxType::backward, stencil.approx_order / 2);
    } else {
      return DerivStencil<1>(DerivApproxType::forward, stencil.approx_order / 2);
    }
  }
}

/** @brief Construct a row of the finite difference derivative operator with
 *         the given stencil along the given direction in reciprocal lattice
 *         coordinates.
 *  @note At k-space boundaries, if `kmb` is not periodic, a different stencil is chosen
 *        to avoid sampling points outside the boundary of `kmb`.
 */
template <std::size_t k_dim>
IndexValPairs finite_difference(const kmBasis<k_dim> &kmb,
    const DerivStencil<1> &stencil, PetscInt row_ikm, std::size_t deriv_dir) {
  std::vector<PetscInt> column_ikms;
  std::vector<PetscScalar> column_vals;

  // Use an alternate stencil at the k-space boundary, if necessary.
  auto alt_stencil = boundary_replacement_stencil(kmb, stencil, row_ikm, deriv_dir);

  std::vector<PetscInt> Deltas_1d = alt_stencil.Delta_vals.first;
  std::vector<PetscScalar> vals_1d = alt_stencil.Delta_vals.second;

  for (std::size_t Delta_index = 0; Delta_index < Deltas_1d.size(); Delta_index++) {
    dkComps<k_dim> Delta;
    for (std::size_t d_Delta = 0; d_Delta < k_dim; d_Delta++) {
      if (d_Delta == deriv_dir) {
        Delta.at(d_Delta) = Deltas_1d.at(Delta_index);
      } else {
        Delta.at(d_Delta) = 0;
      }
    }

    column_ikms.push_back(*(kmb.add(row_ikm, Delta)));
    column_vals.push_back(vals_1d.at(Delta_index) / kmb.k_step(deriv_dir));
  }

  return IndexValPairs(column_ikms, column_vals);
}

/** @brief Construct a k_dim-dimensional vector of matrices representing d/dk
 *         along each of the k_dim directions in reciprocal lattice coordinates.
 *         d/dk is calculated using the central derivative of the given order.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param order Order of approximation to use for finite difference calculation.
 *  @todo Could/should the derivative calculation be improved using Richardson
 *        extrapolation? (i.e. consider two different k-point densities, Nk and
 *        2*Nk, and calculate the derivative using the difference in derivative
 *        estimates from these; see documentation of Python package numdifftools
 *        for discussion of this technique).
 *        NOTE - use of Richardson extrapolation for this would spoil use of
 *        this to calculate E(k+dk) - E(k) by matrix-vector product: would need
 *        another interface for this. (Could add parameter here to select
 *        accelerated or not - then this problem goes away by selecting
 *        non-accelerated.)
 *  @todo Generalize to derivatives beyond order 1?
 */
template <std::size_t k_dim>
std::array<Mat, k_dim> make_d_dk_recip(const kmBasis<k_dim> &kmb,
    const DerivStencil<1> &stencil) {
  PetscInt expected_elems_per_row = stencil.approx_order*k_dim;

  std::array<Mat, k_dim> d_dk_recip;
  // TODO could factor out loop body, same for each d
  // (d just used in finite_difference call and putting into array)
  for (std::size_t d = 0; d < k_dim; d++) {
    Mat d_dk_d = make_Mat(kmb.end_ikm, kmb.end_ikm, expected_elems_per_row);
    
    PetscInt begin, end;
    PetscErrorCode ierr = MatGetOwnershipRange(d_dk_d, &begin, &end);CHKERRXX(ierr);

    // TODO would it be better to group row data and only call MatSetValues once?
    for (PetscInt local_row = begin; local_row < end; local_row++) {
      std::vector<PetscInt> column_ikms;
      std::vector<PetscScalar> column_vals;
      std::tie(column_ikms, column_vals) = finite_difference(kmb, stencil, local_row, d);
      
      ierr = MatSetValues(d_dk_d, 1, &local_row, column_ikms.size(),
          column_ikms.data(), column_vals.data(), INSERT_VALUES);CHKERRXX(ierr);
    }

    ierr = MatAssemblyBegin(d_dk_d, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    ierr = MatAssemblyEnd(d_dk_d, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    
    d_dk_recip.at(d) = d_dk_d;
  }

  return d_dk_recip;
}

/** @brief Construct a k_dim-dimensional vector of matrices representing d/dk
 *         along each of the k_dim directions in Cartesian coordinates.
 *         d/dk is calculated using the central derivative of the given order.
 *  @param D Matrix with elements [D]_{ci} giving the c'th Cartesian component
 *           (in order x, y, z) of the i'th lattice vector.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 *  @param order Order of approximation to use for finite difference calculation.
 *  @todo Generalize to derivatives beyond order 1?
 */
template <std::size_t k_dim>
std::array<Mat, k_dim> make_d_dk_Cartesian(DimMatrix<k_dim> D, const kmBasis<k_dim> &kmb,
    const DerivStencil<1> &stencil) {
  auto d_dk_recip = make_d_dk_recip(kmb, stencil);

  // Each d/dk_c could contain elements from each d/dk_i.
  PetscInt expected_elems_per_row = stencil.approx_order*k_dim*k_dim;

  std::array<Mat, k_dim> d_dk_Cart;
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    // d_dk_Cart[dc] = \sum_i D[c, i] * d_dk[i]
    std::array<PetscScalar, k_dim> coeffs;
    for (std::size_t di = 0; di < k_dim; di++) {
      double Dci = D.at(dc).at(di);
      double coeff = Dci / (2*pi);
      coeffs.at(di) = coeff;
    }

    Mat d_dk_c = Mat_from_sum_const(coeffs, d_dk_recip, expected_elems_per_row);

    d_dk_Cart.at(dc) = d_dk_c;
  }

  return d_dk_Cart;
}

/** @brief Apply the derivative operator `deriv` to the k-diagonal matrix `x`.
 *  @note [apply_deriv(d, x)]_{km, km'} = \sum_{k'} [d]_{km, k'm} [x]_{k'm, k'm'}
 *           = \sum_{k'} [d x]_{km, k'm'}.
 */
template <std::size_t k_dim>
Mat apply_deriv(const kmBasis<k_dim> &kmb, Mat deriv, Mat rho) {
  PetscInt size_m, size_n;
  PetscErrorCode ierr = MatGetSize(deriv, &size_m, &size_n);CHKERRXX(ierr);
  assert(size_m == size_n);

  Mat prod;
  ierr = MatMatMult(deriv, rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &prod);CHKERRXX(ierr);

  PetscInt expected_elems_per_row = kmb.Nbands;
  Mat result = make_Mat(size_m, size_m, expected_elems_per_row);

  PetscInt begin, end;
  ierr = MatGetOwnershipRange(deriv, &begin, &end);CHKERRXX(ierr);

  for (PetscInt km = begin; km < end; km++) {
    kComps<k_dim> k = std::get<0>(kmb.decompose(km));

    PetscInt prod_ncols;
    const PetscInt *prod_cols;
    const PetscScalar *prod_vals;
    ierr = MatGetRow(prod, km, &prod_ncols, &prod_cols, &prod_vals);CHKERRXX(ierr);

    std::map<PetscInt, PetscScalar> result_row;

    for (PetscInt prod_cols_index = 0; prod_cols_index < prod_ncols; prod_cols_index++) {
      PetscInt kpmp = prod_cols[prod_cols_index];
      PetscScalar val = prod_vals[prod_cols_index];

      unsigned int mp = std::get<1>(kmb.decompose(kpmp));
      PetscInt kmp = kmb.compose(std::make_tuple(k, mp));

      if (result_row.count(kmp) != 0) {
        // [d]_{km, k'm} will have a small, fixed number of nonzeros in each row,
        // set by the dimension and the order of the finite difference approximation.
        // Because of this, not concerned about rounding error here.
        result_row[kmp] += val;
      } else {
        result_row[kmp] = val;
      }
    }

    ierr = MatRestoreRow(prod, km, &prod_ncols, &prod_cols, &prod_vals);CHKERRXX(ierr);

    std::vector<PetscInt> result_row_cols;
    std::vector<PetscScalar> result_row_vals;
    result_row_cols.reserve(result_row.size());
    result_row_vals.reserve(result_row.size());

    for (auto it = result_row.begin(); it != result_row.end(); ++it) {
      result_row_cols.push_back(it->first);
      result_row_vals.push_back(it->second);
    }

    assert(result_row_cols.size() == result_row_vals.size());
    ierr = MatSetValues(result, 1, &km, result_row_cols.size(),
        result_row_cols.data(), result_row_vals.data(), INSERT_VALUES);CHKERRXX(ierr);
  }

  ierr = MatAssemblyBegin(result, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  ierr = MatAssemblyEnd(result, MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);

  ierr = MatDestroy(&prod);CHKERRXX(ierr);

  return result;
}

} // namespace anomtrans

#endif // ANOMTRANS_DERIVATIVE_H
