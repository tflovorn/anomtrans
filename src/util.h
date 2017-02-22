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

/** @brief Given a vector `xs` where `xs.at(i).second` is a vector index,
 *         return the corresponding vector which inverts indices and values.
 *  @return A vector `ys` where ys.at(xs.at(i).second) == i.
 */
std::vector<PetscInt> invert_vals_indices(std::vector<std::pair<PetscScalar, PetscInt>> xs);

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
bool check_equal_within(std::vector<T> xs, std::vector<T> ys, PetscReal tol) {
  if (xs.size() != ys.size()) {
    return false;
  }
  return xs == ys;
}

/** @brief Specialize check_equal_within to handle lists-of-lists.
 */
template <typename T>
bool check_equal_within(std::vector<std::vector<T>> xs, std::vector<std::vector<T>> ys, PetscReal tol) {
  if (xs.size() != ys.size()) {
    return false;
  }
  // TODO better to use vector::size_type?
  // Get error in parsing trying to use vector<vector<T>>::size_type.
  for (std::size_t i = 0; i < xs.size(); i++) {
    if (not check_equal_within(xs.at(i), ys.at(i), tol)) {
      return false;
    }
  }
  return true;
}

static_assert(std::is_same<PetscScalar, PetscReal>::value,
    "The implementation of check_equal_within assumes that PetscScalar is a real-valued type.");

/** @brief Specialize check_equal_within to handle floats.
 *  @note We assume that PetscReal and PetscScalar are the same type.
 *        An additional specialization for PetscScalar will not compile
 *        when there are the same type.
 *  @todo Handle PetscScalar != PetscReal.
 */
template <>
bool check_equal_within<PetscReal>(std::vector<PetscReal> xs, std::vector<PetscReal> ys, PetscReal tol);

} // namespace anomtrans

#endif // ANOMTRANS_UTIL_H
