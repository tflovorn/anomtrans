#include "dyn_dm_graph.h"

namespace anomtrans {

boost::optional<int> get_impurity_order(boost::optional<std::shared_ptr<DynDMGraphNode>> A,
    boost::optional<std::shared_ptr<DynDMGraphNode>> B) {
  if (A and B) {
    if ((*A)->impurity_order == (*B)->impurity_order) {
      return (*A)->impurity_order;
    } else {
      return boost::none;
    }
  } else if (A) {
    return (*A)->impurity_order;
  } else if (B) {
    return (*B)->impurity_order;
  } else {
    throw std::invalid_argument("at least one of A and B must be specified");
  }
}

namespace internal {

std::complex<double> get_driving_sum_scale(DynVariation Evar) {
  if (Evar == DynVariation::cos) {
    return std::complex<double>(1.0/2.0, 0.0);
  } else {
    return std::complex<double>(0.0, 1.0/2.0);
  }
}

std::complex<double> get_driving_sum_upper_factor(DynVariation Evar) {
  if (Evar == DynVariation::cos) {
    return 1.0;
  } else {
    return -1.0;
  }
}

} // namespace internal

} // namespace anomtrans
