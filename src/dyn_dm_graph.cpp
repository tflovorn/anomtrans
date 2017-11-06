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

} // namespace anomtrans
