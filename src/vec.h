#ifndef ANOMTRANS_VEC_H
#define ANOMTRANS_VEC_H

#include <cassert>
#include <exception>
#include <vector>
#include <tuple>
#include <petscksp.h>
#include "grid_basis.h"

namespace anomtrans {

using IndexValPairs = std::tuple<std::vector<PetscInt>, std::vector<PetscScalar>>;

IndexValPairs get_local_contents(Vec v);

using stdvec_size = std::vector<PetscInt>::size_type;

/** @brief Scatter the contents of `v` onto rank 0 and return a std::vector
 *         with the local contents (which on rank 0 will be `v`'s values
 *         and an empty std::vector on other ranks).
 */
std::vector<PetscScalar> collect_contents(Vec v);

/** @brief Apply a function `f` to each element of the vector `v_in` and return
 *         the corresponding vector of outputs.
 *  @todo Should v_in be const here? This is certainly the intended behavior.
 */ 
template <std::size_t k_dim, typename F>
Vec vector_elem_apply(kmBasis<k_dim> kmb, Vec v_in, F f) {
  Vec v_out;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm, &v_out);CHKERRXX(ierr);

  // This node is assigned elements in the range begin <= i < end.
  PetscInt begin_in, end_in;
  ierr = VecGetOwnershipRange(v_in, &begin_in, &end_in);CHKERRXX(ierr);

  PetscInt begin_out, end_out;
  ierr = VecGetOwnershipRange(v_out, &begin_out, &end_out);CHKERRXX(ierr);

  if (begin_in != begin_out or end_in != end_out) {
    throw std::runtime_error("got different local element ranges for input and output vectors in vector_apply");
  }

  std::vector<PetscInt> local_in_rows;
  std::vector<PetscScalar> local_in_vals;
  std::tie(local_in_rows, local_in_vals) = get_local_contents(v_in);

  std::vector<PetscScalar> local_out_vals;
  local_out_vals.reserve(end_out - begin_out);

  for (stdvec_size i = 0; i < local_in_rows.size(); i++) {
    PetscScalar out_val = f(local_in_vals.at(i));

    local_out_vals.push_back(out_val);
  }
  assert(local_out_vals.size() == local_in_rows.size());

  // TODO would we be better off adding these elements one at a time (contrary to
  // the PETSc manual's advice), since we don't have them precomputed?
  // Doing it this way uses extra memory inside this scope.
  ierr = VecSetValues(v_out, local_in_rows.size(), local_in_rows.data(), local_out_vals.data(), INSERT_VALUES);CHKERRXX(ierr);

  ierr = VecAssemblyBegin(v_out);CHKERRXX(ierr);
  ierr = VecAssemblyEnd(v_out);CHKERRXX(ierr);

  return v_out;
}

} // namespace anomtrans

#endif // ANOMTRANS_VEC_H
