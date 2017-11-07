#ifndef ANOMTRANS_UTIL_LATTICE_H
#define ANOMTRANS_UTIL_LATTICE_H

namespace anomtrans {

/** @brief A (dim x dim) matrix represented as nested arrays.
 */
template <std::size_t dim>
using DimMatrix = std::array<std::array<double, dim>, dim>;

/** @brief A vector in lattice coordinates.
 */
template <std::size_t dim>
using LatVec = std::array<int, dim>;

/** @brief A vector in Cartesian coordinates.
 */
template <std::size_t dim>
using CartVec = std::array<double, dim>;

/** @brief Convert a vector in lattice coordinates to Cartesian coordinates.
 *  @param L_lat A vector in lattice coordinates.
 *  @param D A matrix giving the lattice vectors: D[c][i] is the c'th Cartesian
 *           component of the i'th lattice vector.
 *  @todo Break this out to its own file for Cartesian vector operations?
 */
template <std::size_t dim>
CartVec<dim> lat_vec_to_Cart(DimMatrix<dim> D, LatVec<dim> L_lat) {
  CartVec<dim> L_Cart;
  // TODO could factor out L_lat -> L_Cart
  for (std::size_t dc = 0; dc < dim; dc++) {
    L_Cart.at(dc) = 0.0;
    for (std::size_t di = 0; di < dim; di++) {
      L_Cart.at(dc) += L_lat.at(di) * D.at(dc).at(di);
    }
  }

  return L_Cart;
}

/** @brief Calculate |L_Cart|^2.
 *  @todo Break this out to its own file for Cartesian vector operations?
 */
template <std::size_t dim>
double norm2_CartVec(CartVec<dim> L_Cart) {
  double norm2_L_Cart = 0.0;
  for (std::size_t dc = 0; dc < dim; dc++) {
    norm2_L_Cart += std::pow(L_Cart.at(dc), 2.0);
  }
  return norm2_L_Cart;
}

} // namespace anomtrans

#endif // ANOMTRANS_UTIL_LATTICE_H
