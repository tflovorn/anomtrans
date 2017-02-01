#ifndef ANOMTRANS_DERIVATIVE_H
#define ANOMTRANS_DERIVATIVE_H

#include <array>
#include <vector>
#include <tuple>
#include <map>
#include <stdexcept>
#include "constants.h"
#include "dist_vec.h"
#include "grid_basis.h"

namespace anomtrans {

template <std::size_t k_dim>
std::tuple<std::vector<GO>, std::vector<double>> finite_difference(kmBasis<k_dim> kmb,
    unsigned int order, GO row_ikm, std::size_t deriv_dir) {
  if (order % 2 == 1) {
    throw std::invalid_argument("Only even-order central finite differences are defined");
  }

  // TODO could declare these globally to avoid constructing here on each call.
  const std::map<unsigned int, std::vector<int>> all_Deltas_1d {
    {2, {1, -1}}
  };
  const std::map<unsigned int, std::vector<double>> all_vals_1d {
    {2, {0.5, -0.5}}
  };

  if (all_Deltas_1d.count(order) == 0 or all_vals_1d.count(order) == 0) {
    throw std::invalid_argument("The given finite-difference order is not implemented");
  }

  std::vector<int> Deltas_1d = all_Deltas_1d.at(order);
  std::vector<double> vals_1d = all_vals_1d.at(order);

  std::vector<GO> column_ikms;
  std::vector<double> column_vals;

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
    column_vals.push_back(vals_1d.at(Delta_index) / k_d_spacing);
  }

  return std::tuple<std::vector<GO>, std::vector<double>>(column_ikms, column_vals);
}

/** @brief Construct a k_dim-dimensional vector of matrices representing d/dk
 *         along each of the k_dim directions in reciprocal lattice coordinates.
 *         d/dk is calculated using the central derivative of the given order.
 *  @param kmb Object representing the discretization of k-space and the number
 *             of bands.
 */
template <std::size_t k_dim>
std::array<RCP<CrsMatrix<double>>, k_dim> make_d_dk_recip(kmBasis<k_dim> kmb,
    unsigned int order) {
  auto kmb_map = kmb.get_map();
  std::array<RCP<CrsMatrix<double>>, k_dim> d_dk_recip;
  for (std::size_t d = 0; d < k_dim; d++) {
    // Use the simplest behavior of CrsMatrix: we do not supply a maximum
    // number of entries per row, by setting the second argument (maxNumEntriesPerRow)
    // to 0 and keeping the default third argument (which determines how
    // maxNumEntriesPerRow is used).
    d_dk_recip.at(d) = RCP<CrsMatrix<double>>(new CrsMatrix<double>(kmb_map, 0));
  }

  const LO num_local_elements = static_cast<LO>(kmb_map->getNodeNumElements());

  for (LO local_row = 0; local_row < num_local_elements; local_row++) {
    const GO global_row = kmb_map->getGlobalElement(local_row);
    for (std::size_t d = 0; d < k_dim; d++) {
      std::vector<GO> column_ikms;
      std::vector<double> column_vals;
      std::tie(column_ikms, column_vals) = finite_difference(kmb, order, global_row, d);

      d_dk_recip.at(d)->insertGlobalValues(global_row, column_ikms, column_vals);
    }
  }
  
  for (std::size_t d = 0; d < k_dim; d++) {
    d_dk_recip.at(d)->fillComplete();
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

} // namespace anomtrans

#endif // ANOMTRANS_DERIVATIVE_H
