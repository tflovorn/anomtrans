#ifndef ANOMTRANS_CONSTANTS_H
#define ANOMTRANS_CONSTANTS_H

#include <cmath>
#include <limits>

namespace anomtrans {
  const double pi = acos(-1.0);

  /** @brief Natural logarithm of the smallest normalized positive double.
   */
  const double LN_DBL_MIN = std::log(std::numeric_limits<double>::min());

  /** @brief Natural logarithm of machine epsilon for doubles.
   */
  const double LN_DBL_EPS = std::log(std::numeric_limits<double>::epsilon());
}

#endif // ANOMTRANS_CONSTANTS_H
