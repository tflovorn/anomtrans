#include "vec.h"

namespace anomtrans {

IndexValPairs get_local_contents(Vec v) {
  PetscInt begin, end;
  PetscErrorCode ierr = VecGetOwnershipRange(v, &begin, &end);CHKERRXX(ierr);
  PetscInt num_local_rows = end - begin;
  assert( num_local_rows >= 0 );

  std::vector<PetscInt> local_rows;
  local_rows.reserve(num_local_rows);

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    local_rows.push_back(local_row);
  }

  // TODO do we really need to 0-initialize this, or would reserve() be OK?
  std::vector<PetscScalar> local_vals(num_local_rows);

  ierr = VecGetValues(v, num_local_rows, local_rows.data(), local_vals.data());CHKERRXX(ierr);

  return IndexValPairs(local_rows, local_vals);
}

} // namespace anomtrans
