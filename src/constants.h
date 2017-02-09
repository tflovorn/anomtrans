#ifndef ANOMTRANS_CONSTANTS_H
#define ANOMTRANS_CONSTANTS_H

#include <cmath>
#include <cfloat>

namespace anomtrans {
  const double pi = acos(-1.0);

  /** @brief Natural logarith of the smallest normalized positive double.
   *  @todo Could use std::numeric_limits<double>::min() instead of DBL_MIN.
   */
  const double LN_DBL_MIN = std::log(DBL_MIN);
}

#endif // ANOMTRANS_CONSTANTS_H
