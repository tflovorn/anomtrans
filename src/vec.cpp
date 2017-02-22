#include "vec.h"

namespace anomtrans {

PetscReal get_Vec_MaxAbs(Vec v) {
  Vec v_abs;
  PetscErrorCode ierr = VecDuplicate(v, &v_abs);CHKERRXX(ierr);
  ierr = VecCopy(v, v_abs);CHKERRXX(ierr);
  ierr = VecAbs(v_abs);CHKERRXX(ierr);

  PetscReal v_abs_max;
  ierr = VecMax(v_abs, nullptr, &v_abs_max);CHKERRXX(ierr);

  ierr = VecDestroy(&v_abs);CHKERRXX(ierr);

  return v_abs_max;
}

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

std::vector<PetscScalar> collect_contents(Vec v) {
  VecScatter ctx;
  Vec collected;
  PetscErrorCode ierr = VecScatterCreateToZero(v, &ctx, &collected);CHKERRXX(ierr);
  ierr = VecScatterBegin(ctx, v, collected, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);
  ierr = VecScatterEnd(ctx, v, collected, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);

  auto local_collected = get_local_contents(collected);

  ierr = VecScatterDestroy(&ctx);CHKERRXX(ierr);
  ierr = VecDestroy(&collected);CHKERRXX(ierr);

  // Have all the values in v on rank 0, so we don't care about keeping the rows
  // information.
  return std::get<1>(local_collected);
}

} // namespace anomtrans
