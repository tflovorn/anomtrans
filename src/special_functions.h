#ifndef ANOMTRANS_SPECIAL_FUNCTIONS_H
#define ANOMTRANS_SPECIAL_FUNCTIONS_H

#include <cmath>
#include "constants.h"

namespace anomtrans {

/** @brief Calculate the Fermi-Dirac distribution function
 *         f(E, beta) = (e^{beta*E} + 1)^{-1}.
 *  @note Uses the implementation strategy followed by the GNU Scientific Library
 *        function gsl_sf_fermi_dirac_m1, where the Fermi function is rearranged to
 *        always compute the exponential function with nonpositive argument.
 *        If -beta*E < ln(DBL_MIN), e^{-beta*E} will underflow, making
 *        e^{-beta*E}/(1 + e^{-beta*E}) produce a denormal number. In this case,
 *        f(E, beta) is very close to 0 and we replace it with this value.
 *  @todo Is it useful to consider the production of a denormal number by e^{-x}
 *        when x > 0 in the expression 1/(1 + e^{-x})? In this case the returned
 *        value is normalized, but we may have an intermediate denormal.
 */
double fermi_dirac(double beta, double E);

} // namespace anomtrans

#endif // ANOMTRANS_SPECIAL_FUNCTIONS_H
