#ifndef ANOMTRANS_UTIL_H
#define ANOMTRANS_UTIL_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <boost/optional.hpp>
#include <json.hpp>

namespace anomtrans {

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
