#include <cfloat>
#include "Teuchos_UnitTestHarness.hpp"
#include "special_functions.h"

namespace {

TEUCHOS_UNIT_TEST( Special_Functions, Fermi_Dirac ) {
  double tol = 10*DBL_EPSILON;

  double beta = 10.0;
  double E_below_min = 2*anomtrans::LN_DBL_MIN/beta;
  double E_above_max = -E_below_min;

  TEST_FLOATING_EQUALITY( anomtrans::fermi_dirac(E_below_min, beta), 1.0, tol );
  TEST_FLOATING_EQUALITY( anomtrans::fermi_dirac(E_above_max, beta), 0.0, tol );

  TEST_FLOATING_EQUALITY( anomtrans::fermi_dirac(0.0, beta), 0.5, tol );
}

} // namespace
