#ifndef ANOMTRANS_UTIL_H
#define ANOMTRANS_UTIL_H

#include <vector>

namespace anomtrans {

/** @brief Return evenly space numbers over the interval from start to stop,
 *         including both endpoints. If num == 1, only start is included.
 *  @param start The first value of the sequence.
 *  @param stop The last value of the sequence.
 *  @param num The length of the sequence.
 */
std::vector<double> linspace(double start, double stop, unsigned int num);

} // namespace anomtrans

#endif // ANOMTRANS_UTIL_H
