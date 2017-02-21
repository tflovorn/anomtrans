#ifndef ANOMTRANS_DERIVATIVE_H
#define ANOMTRANS_DERIVATIVE_H

#include <cstddef>
#include <array>
#include <vector>
#include <tuple>
#include <utility>
#include <map>
#include <stdexcept>
#include <iostream>
#include <petscksp.h>
#include "constants.h"
#include "grid_basis.h"
#include "vec.h"
#include "mat.h"

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
      {DerivSpecifier(1, DerivApproxType::forward, 1), {1, 0}},
      {DerivSpecifier(1, DerivApproxType::central, 2), {1, -1}}
    };
    const std::map<DerivSpecifier, std::vector<PetscScalar>> all_vals_1d {
      {DerivSpecifier(1, DerivApproxType::forward, 1), {1.0, -1.0}},
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

template <std::size_t k_dim>
IndexValPairs finite_difference(const kmBasis<k_dim> &kmb,
    const DerivStencil<1> &stencil, PetscInt row_ikm, std::size_t deriv_dir) {
  std::vector<PetscInt> Deltas_1d = stencil.Delta_vals.first;
  std::vector<PetscScalar> vals_1d = stencil.Delta_vals.second;

  std::vector<PetscInt> column_ikms;
  std::vector<PetscScalar> column_vals;

  double k_d_spacing = 1.0/kmb.Nk.at(deriv_dir);

  // better to use vector::size_type here?
  for (std::size_t Delta_index = 0; Delta_index < Deltas_1d.size(); Delta_index++) {
    dkComps<k_dim> Delta;
    for (std::size_t d_Delta = 0; d_Delta < k_dim; d_Delta++) {
      if (d_Delta == deriv_dir) {
        Delta.at(d_Delta) = Deltas_1d.at(Delta_index);
      } else {
        Delta.at(d_Delta) = 0;
      }
    }

    column_ikms.push_back(kmb.add(row_ikm, Delta));
    column_vals.push_back(k_d_spacing * vals_1d.at(Delta_index));
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
std::array<Mat, k_dim> make_d_dk_recip(kmBasis<k_dim> kmb,
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
std::array<Mat, k_dim> make_d_dk_Cartesian(DimMatrix<k_dim> D, kmBasis<k_dim> kmb,
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

} // namespace anomtrans

#endif // ANOMTRANS_DERIVATIVE_H
