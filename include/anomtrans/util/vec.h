#ifndef ANOMTRANS_VEC_H
#define ANOMTRANS_VEC_H

#include <cassert>
#include <exception>
#include <vector>
#include <tuple>
#include <petscksp.h>

namespace anomtrans {

/** @brief Pair of indices and corresponding vector values.
 *  @todo Would this be better as std::vector<std::pair<PetscInt, PetscScalar>>?
 *        This alternate type is required by collision.
 */
using IndexValPairs = std::tuple<std::vector<PetscInt>, std::vector<PetscScalar>>;

IndexValPairs get_local_contents(Vec v);

using stdvec_size = std::vector<PetscInt>::size_type;

/** @brief Get the largest absolute value of any element in v.
 *  @todo Could pass in a vector, which is used internally to hold the
 *        absolute value of each element. Passing in the vector prevents it from
 *        being allocated on each call if this function is called repeatedly.
 */
PetscReal get_Vec_MaxAbs(Vec v);

/** @brief Scatter the contents of `v` onto rank 0 and return a std::vector
 *         with the local contents (which on rank 0 will be `v`'s values
 *         and an empty std::vector on other ranks).
 */
std::vector<PetscScalar> collect_contents(Vec v);

/** @brief Create and return a Vec which has the contents of `v` scattered to
 *         all ranks.
 */
Vec scatter_to_all(Vec v);

/** @brief Construct a vector u = \sum_d coeffs(d) * vs(d).
 *  @pre The length of coeffs and vs should be at least 1.
 *  @pre All vectors in vs should have the same length.
 */
template <std::size_t len>
Vec Vec_from_sum_const(std::array<PetscScalar, len> coeffs, std::array<Vec, len> vs) {
  static_assert(len > 0, "must have at least 1 Vec for Vec_from_sum_const");

  Vec u;
  PetscErrorCode ierr = VecDuplicate(vs.at(0), &u);CHKERRXX(ierr);
  ierr = VecSet(u, 0.0);CHKERRXX(ierr);

  ierr = VecMAXPY(u, len, coeffs.data(), vs.data());CHKERRXX(ierr);

  return u;
}

/** @brief Apply a function `f` to each element of the vector `v_in` and return
 *         the corresponding vector of outputs.
 *  @param v_in The vector of function inputs.
 *  @param f A function with the signature
 *             PetscScalar f(PetscScalar).
 *  @todo Should this be replaced with use of PETSc PF functions? PFApplyVec
 *        does what we want to do here. However, creating the PF to pass to
 *        PFApplyVec via PFCreate, PFSet seems to require a function like this
 *        to exist.
 *  @todo Should v_in be const here? This is certainly the intended behavior.
 */ 
template <typename F>
Vec vector_elem_apply(Vec v_in, F f) {
  PetscInt v_in_size;
  PetscErrorCode ierr = VecGetSize(v_in, &v_in_size);CHKERRXX(ierr);

  Vec v_out;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, v_in_size, &v_out);CHKERRXX(ierr);

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

/** @brief Construct an arrays of Vec's by applying a function `f` to each index of the
 *         vector of length N. Each returned Vec corresponds to one return value position
 *         of `f`.
 *  @param N The global length of the desired vector.
 *  @param f A function with the signature
 *           std::array<PetscScalar, out_dim> f(PetscInt).
 *  @pre out_dim must be at least 1.
 *  @invariant f(i) will only be called for rows i belonging to the current rank.
 */
template <std::size_t out_dim, typename F>
std::array<Vec, out_dim> vector_index_apply_multiple(PetscInt N, const F &f) {
  static_assert(out_dim > 0, "Must have at least one output of f");

  std::array<Vec, out_dim> vs;

  for (std::size_t oi = 0; oi < out_dim; oi++) {
    PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE,
        N, &(vs.at(oi)));CHKERRXX(ierr);
  }

  PetscInt begin, end;
  PetscErrorCode ierr = VecGetOwnershipRange(vs.at(0), &begin, &end);CHKERRXX(ierr);

  std::vector<PetscInt> rows;
  std::array<std::vector<PetscScalar>, out_dim> vals;

  rows.reserve(end - begin);

  for (std::size_t oi = 0; oi < out_dim; oi++) {
    vals.at(oi).reserve(end - begin);
  }

  for (PetscInt i = begin; i < end; i++) {
    rows.push_back(i);

    std::array<PetscScalar, out_dim> i_vals = f(i);

    for (std::size_t oi = 0; oi < out_dim; oi++) {
      vals.at(oi).push_back(i_vals.at(oi));
    }
  }

  for (std::size_t oi = 0; oi < out_dim; oi++) {
    assert(rows.size() == vals.at(oi).size());

    ierr = VecSetValues(vs.at(oi), rows.size(), rows.data(), vals.at(oi).data(),
        INSERT_VALUES);CHKERRXX(ierr);

    ierr = VecAssemblyBegin(vs.at(oi));CHKERRXX(ierr);
    ierr = VecAssemblyEnd(vs.at(oi));CHKERRXX(ierr);
  }

  return vs;
}

/** @brief Construct a Vec by applying a function `f` to each index of the
 *         vector of length N.
 *  @param N The global length of the desired vector.
 *  @param f A function with the signature
 *           PetscScalar f(PetscInt).
 *  @invariant f(i) will only be called for rows i belonging to the current rank.
 */
template <typename F>
Vec vector_index_apply(PetscInt N, const F &f) {
  // This function is a special case of vector_index_apply_multiple where out_dim = 1.
  auto f_multiple = [&f](PetscInt i)->std::array<PetscScalar, 1> {
    return { f(i) };
  };

  return vector_index_apply_multiple<1>(N, f_multiple).at(0);
}

} // namespace anomtrans

#endif // ANOMTRANS_VEC_H
