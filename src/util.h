#ifndef ANOMTRANS_UTIL_H
#define ANOMTRANS_UTIL_H

#include <vector>
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
  cross.at(0) = v.at(1) * u.at(2) - v.at(2) * u.at(1);
  cross.at(1) = -v.at(0) * u.at(2) + v.at(2) * u.at(0);
  cross.at(2) = v.at(0) * u.at(1) - v.at(1) * u.at(0);

  return cross;
}

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

} // namespace anomtrans

#endif // ANOMTRANS_UTIL_H
