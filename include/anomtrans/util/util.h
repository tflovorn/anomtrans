#ifndef ANOMTRANS_UTIL_H
#define ANOMTRANS_UTIL_H

#include <cmath>
#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <stdexcept>
#include <boost/optional.hpp>
#include <json.hpp>
#include <petscksp.h>

namespace anomtrans {

/** @brief Expand an array `u_in` of size `u_dim` to size `expand_dim` by filling
 *         empty spaces with zeros.
 *  @note Using deduction of trailing template arguments, this can be called as
 *        expand<3>(u) to expand u to size 3.
 */
template <std::size_t expand_dim, std::size_t u_dim>
std::array<PetscScalar, expand_dim> expand(std::array<PetscScalar, u_dim> u_in) {
  std::array<PetscScalar, expand_dim> u;
  for (std::size_t d = 0; d < expand_dim; d++) {
    if (d < u_dim) {
      u.at(d) = u_in.at(d);
    } else {
      u.at(d) = 0.0;
    }
  }
  return u;
}

/** @brief Compute the cross product of the vectors `u_in` and `v_in`.
 *         `u_in` or `v_in` may have dimension less than 3; if so, promote them to
 *         dimension 3 by filling the missing values with zeros.
 *  @todo Break this out to its own file for Cartesian vector operations?
 */
template <std::size_t u_dim, std::size_t v_dim>
std::array<PetscScalar, 3> cross(std::array<PetscScalar, u_dim> u_in, std::array<PetscScalar, v_dim> v_in) {
  std::array<PetscScalar, 3> u = expand<3>(u_in);
  std::array<PetscScalar, 3> v = expand<3>(v_in);

  std::array<PetscScalar, 3> cross;
  cross.at(0) = u.at(1) * v.at(2) - u.at(2) * v.at(1);
  cross.at(1) = -u.at(0) * v.at(2) + u.at(2) * v.at(0);
  cross.at(2) = u.at(0) * v.at(1) - u.at(1) * v.at(0);

  return cross;
}

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

/** @brief Given a vector `xs` where `xs.at(i).second` is a vector index,
 *         return the corresponding vector which inverts indices and values.
 *  @return A vector `ys` where ys.at(xs.at(i).second) == i.
 */
template <typename T>
std::vector<PetscInt> invert_vals_indices(std::vector<std::pair<T, PetscInt>> xs) {
  std::vector<PetscInt> ys(xs.size());
  for (std::size_t i = 0; i < xs.size(); i++) {
    ys.at(xs.at(i).second) = i;
  }
  return ys;
}

/** @brief Wrap x to a range of values [0, 1, ..., N-1].
 *         Negative values of x are wrapped starting from the right-hand side
 *         of the range.
 *  @note If x >= 0, then wrap(x, N) == x % N.
 *        wrap(-1, N) = N-1; wrap(-N, N) = 0; wrap(-(N+1), N) = N-1.
 */
PetscInt wrap(PetscInt x, PetscInt N);

/** @brief Return evenly space numbers over the interval from start to stop,
 *         including both endpoints. If num == 1, only start is included.
 *  @param start The first value of the sequence.
 *  @param stop The last value of the sequence.
 *  @param num The length of the sequence.
 */
std::vector<double> linspace(double start, double stop, unsigned int num);

/** @brief Split the vector of (complex) scalars v into a pair of vectors
 *         giving the real and imaginary parts of the elements of v.
 */
std::pair<std::vector<PetscReal>, std::vector<PetscReal>> split_scalars(std::vector<PetscScalar> v);

/** @brief Convert an array `x` of real elements to an array of complex elements
 *         with real parts given by the elements of `x` and 0 imaginary parts.
 */
template <std::size_t len>
std::array<PetscScalar, len> make_complex_array(std::array<PetscReal, len> x) {
  std::array<PetscScalar, len> result;
  for (std::size_t i = 0; i < len; i++) {
    result.at(i) = std::complex<double>(x.at(i), 0.0);
  }

  return result;
}

/** @brief Wrapper around std::getenv.
 *  @param var Name of the environment variable to get the value of.
 *  @return An optional which contains the environment variable's
 *          value only if that environment variable exists; it contains
 *          no value otherwise.
 */
boost::optional<std::string> getenv_optional(const std::string& var);

/** @brief Return true if the JSON data stored at known_path is the same as that
 *         contained in j_test, and false otherwise.
 */
bool check_json_equal(std::string test_path, std::string known_path);

/** @brief Check if each member of the lists xs and ys are equal, considering
 *         floating-point numbers to be distinct if the absolute value of their
 *         difference is greater than tol.
 *  @note This generic version does not do the floating-point comparison: that
 *        is handled in the float specializations.
 */
template <typename T>
bool check_equal_within(std::vector<T> xs, std::vector<T> ys, PetscReal eps_abs, PetscReal eps_rel) {
  if (xs.size() != ys.size()) {
    return false;
  }
  return xs == ys;
}

/** @brief Specialize check_equal_within to handle lists-of-lists.
 */
template <typename T>
bool check_equal_within(std::vector<std::vector<T>> xs, std::vector<std::vector<T>> ys, PetscReal eps_abs, PetscReal eps_rel) {
  if (xs.size() != ys.size()) {
    return false;
  }
  // TODO better to use vector::size_type?
  // Get error in parsing trying to use vector<vector<T>>::size_type.
  for (std::size_t i = 0; i < xs.size(); i++) {
    if (not check_equal_within(xs.at(i), ys.at(i), eps_abs, eps_rel)) {
      return false;
    }
  }
  return true;
}

/** @brief Specialize check_equal_within to handle PetscReal.
 */
template <>
bool check_equal_within<PetscReal>(std::vector<PetscReal> xs, std::vector<PetscReal> ys, PetscReal eps_abs, PetscReal eps_rel);

/** @brief Specialize check_equal_within to handle PetscScalar = complex<PetscReal>.
 */
template <>
bool check_equal_within<PetscScalar>(std::vector<PetscScalar> xs, std::vector<PetscScalar> ys, PetscReal eps_abs, PetscReal eps_rel);

/** @brief Check if x and y are approximately equal, up to given absolute
 *         and relative tolerances.
 *  @note Returns true if |x - y| < eps_abs or
 *                        |x - y| < eps_rel * max(|x|, |y|).
 *        Here |...| is std::abs(T) and max(...) is std::max(|T|, |T|).
 */
template <typename T>
bool scalars_approx_equal(T x, T y, PetscReal eps_abs, PetscReal eps_rel) {
  auto diff = std::abs(x - y);
  if (diff < eps_abs) {
    return true;
  }
  auto max_norm = std::max(std::abs(x), std::abs(y));
  return diff < eps_rel * max_norm;
}

} // namespace anomtrans

#endif // ANOMTRANS_UTIL_H
