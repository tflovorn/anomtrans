#ifndef ANOMTRANS_DERIVATIVE_H
#define ANOMTRANS_DERIVATIVE_H

#include <cstddef>
#include <array>
#include <vector>
#include <tuple>
#include <map>
#include <stdexcept>
#include <iostream>
#include <petscksp.h>
#include "constants.h"
#include "grid_basis.h"
#include "vec.h"
#include "mat.h"

namespace anomtrans {

template <std::size_t k_dim>
IndexValPairs finite_difference(kmBasis<k_dim> kmb,
    unsigned int order, PetscInt row_ikm, std::size_t deriv_dir) {
  if (order % 2 == 1) {
    throw std::invalid_argument("Only even-order central finite differences are defined");
  }

  // TODO could declare these globally to avoid constructing here on each call.
  const std::map<unsigned int, std::vector<PetscInt>> all_Deltas_1d {
    {2, {1, -1}}
  };
  const std::map<unsigned int, std::vector<PetscScalar>> all_vals_1d {
    {2, {0.5, -0.5}}
  };

  if (all_Deltas_1d.count(order) == 0 or all_vals_1d.count(order) == 0) {
    throw std::invalid_argument("The given finite-difference order is not implemented");
  }

  std::vector<int> Deltas_1d = all_Deltas_1d.at(order);
  std::vector<double> vals_1d = all_vals_1d.at(order);

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
 */
template <std::size_t k_dim>
std::array<Mat, k_dim> make_d_dk_recip(kmBasis<k_dim> kmb,
    unsigned int order) {
  PetscInt expected_elems_per_row = order*k_dim;

  std::array<Mat, k_dim> d_dk_recip;
  // TODO could factor out loop body, same for each d
  // (d just used in finite_difference call and putting into array)
  for (std::size_t d = 0; d < k_dim; d++) {
    Mat d_dk_d = make_Mat(kmb, expected_elems_per_row);
    
    PetscInt begin, end;
    PetscErrorCode ierr = MatGetOwnershipRange(d_dk_d, &begin, &end);CHKERRXX(ierr);

    // TODO would it be better to group row data and only call MatSetValues once?
    for (PetscInt local_row = begin; local_row < end; local_row++) {
      std::vector<PetscInt> column_ikms;
      std::vector<PetscScalar> column_vals;
      std::tie(column_ikms, column_vals) = finite_difference(kmb, order, local_row, d);
      
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
 */
/*
template <std::size_t k_dim>
std::array<RCP<CrsMatrix<double>>, k_dim> make_d_dk_Cartesian(DimMatrix<k_dim> D,
    kmBasis<k_dim> kmb, unsigned int order) {
  auto kmb_map = kmb.get_map();
  std::array<RCP<CrsMatrix<double>>, k_dim> d_dk_recip = make_d_dk_recip(kmb, order);

  auto kmb_map = kmb.get_map();
  std::array<RCP<CrsMatrix<double>>, k_dim> d_dk_Cart;
  for (std::size_t d = 0; d < k_dim; d++) {
    d_dk_Cart.at(d) = RCP(new CrsMatrix(kmb_map, 0));
  }

  for (std::size_t dc = 0; dc < k_dim; dc++) {
    for (std::size_t di = 0; di < k_dim; di++) {
      double Dci = D.at(c).at(i);
      double coeff = Dci / (2*pi);
      // Distributed application of: d_dk_Cart[dc] += coeff*d_dk_recip[di]
      d_dk_Cart.at(dc) = d_dk_Cart.at(dc)->add(coeff, d_dk_recip.at(di), 1.0, kmb_map, kmb_map, nullptr);
    }
  }

  for (std::size_t d = 0; d < k_dim; d++) {
    d_dk_Cart.at(d)->fillComplete();
  }

  return d_dk_Cart;
}
*/
} // namespace anomtrans

#endif // ANOMTRANS_DERIVATIVE_H
