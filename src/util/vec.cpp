#include "util/vec.h"

namespace anomtrans {

OwnedVec::OwnedVec() {}

OwnedVec::OwnedVec(Vec _v) : v(_v) {}

OwnedVec::~OwnedVec() {
  // If we don't own a `Vec`, do nothing.
  if (v == nullptr) {
    return;
  }

  // Release the `Vec` that we own. If the release operation fails,
  // suppress the error. (Could be nice to report the error, but maybe if
  // deallocation fails, PetscPrintf would also fail...).
  try {
    PetscErrorCode ierr = VecDestroy(&v);CHKERRXX(ierr);
  } catch (...) {}
}

OwnedVec::OwnedVec(OwnedVec&& other) : v(other.v) {
  other.v = nullptr;
}

OwnedVec& OwnedVec::operator=(OwnedVec&& other) {
  if (this != &other) {
    v = other.v;
    other.v = nullptr;
  }
  return *this;
}

OwnedVec make_Vec(PetscInt m) {
  Vec v;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, m, &v);CHKERRXX(ierr);

  return OwnedVec(v);
}

OwnedVec make_Vec_with_structure(Vec other) {
  Vec v;
  PetscErrorCode ierr = VecDuplicate(other, &v);CHKERRXX(ierr);

  return OwnedVec(v);
}

OwnedVec make_Vec_copy(Vec other) {
  Vec v;
  PetscErrorCode ierr = VecDuplicate(other, &v);CHKERRXX(ierr);
  ierr = VecCopy(other, v);CHKERRXX(ierr);

  return OwnedVec(v);
}

PetscReal get_Vec_MaxAbs(Vec v) {
  auto v_abs = make_Vec_with_structure(v);
  PetscErrorCode ierr = VecCopy(v, v_abs.v);CHKERRXX(ierr);
  ierr = VecAbs(v_abs.v);CHKERRXX(ierr);

  PetscReal v_abs_max;
  ierr = VecMax(v_abs.v, nullptr, &v_abs_max);CHKERRXX(ierr);

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

OwnedVec scatter_to_all(Vec v) {
  VecScatter ctx;
  Vec v_all;
  PetscErrorCode ierr = VecScatterCreateToAll(v, &ctx, &v_all);CHKERRXX(ierr);
  ierr = VecScatterBegin(ctx, v, v_all, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);
  ierr = VecScatterEnd(ctx, v, v_all, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);

  ierr = VecScatterDestroy(&ctx);CHKERRXX(ierr);

  return OwnedVec(v_all);
}

} // namespace anomtrans
