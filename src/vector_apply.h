#ifndef ANOMTRANS_VECTOR_APPLY_H
#define ANOMTRANS_VECTOR_APPLY_H

#include <cstddef>
#include <Tpetra_DefaultPlatform.hpp>
#include "dist_vec.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief Apply a function `f` to each element of the vector `v_in` and return
 *         the corresponding vector of outputs.
 *  @note The template argument out_scalar must be specified when invoking
 *        this function. The remaining template arguments can be deduced from
 *        the arguments.
 */ 
template <typename out_scalar, std::size_t k_dim, typename in_scalar, typename F>
DistVec<out_scalar> vector_elem_apply(kmBasis<k_dim> kmb, DistVec<in_scalar> v_in, F f) {
  auto kmb_map = kmb.get_map();
  DistVec<out_scalar> v_out(kmb_map);

  // The use of `template` in this expression is discussed here:
  // http://stackoverflow.com/a/613132
  v_in.template sync<Kokkos::HostSpace>();
  v_out.template sync<Kokkos::HostSpace>();
  auto v_in_2d = v_in.template getLocalView<Kokkos::HostSpace>();
  auto v_in_1d = Kokkos::subview(v_in_2d, Kokkos::ALL(), 0);
  auto v_out_2d = v_out.template getLocalView<Kokkos::HostSpace>();
  auto v_out_1d = Kokkos::subview(v_out_2d, Kokkos::ALL(), 0);
  v_out.template modify<Kokkos::HostSpace>();

  const LO num_local_elements = static_cast<LO>(kmb_map->getNodeNumElements());

  for (LO local_row = 0; local_row < num_local_elements; local_row++) {
    v_out_1d(local_row) = f(v_in_1d(local_row));
  }

  v_out.template sync<DistVecMemorySpace<out_scalar>>();

  return v_out;
}

} // namespace anomtrans

#endif // ANOMTRANS_VECTOR_APPLY_H
