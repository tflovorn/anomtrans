#ifndef ANOMTRANS_VEC_H
#define ANOMTRANS_VEC_H

#include <cassert>
#include <exception>
#include <vector>
#include <tuple>
#include <petscksp.h>

namespace anomtrans {

/** @brief Container which owns a PETSc `Vec`. When an `OwnedVec` is destroyed,
 *         the corresponding destruction operation for the contained `Vec` is called.
 *         An `OwnedVec` has the same semantics as a `unique_ptr`: it cannot be copied,
 *         but it may be moved from.
 *  @invariant The owned `Vec` is exposed as a public non-const field to allow this field
 *             to be used in PETSc interfaces without wrapping all PETSc functions.
 *             This `Vec` must not be destroyed or reassigned.
 *  @note The note about `Vec` versus `OwnedVec` arguments to functions in the documentation
 *        of `OwnedMat` applies here also.
 */
class OwnedVec {
public:
  Vec v;

  OwnedVec();

  OwnedVec(Vec _v);

  ~OwnedVec();

  OwnedVec(const OwnedVec& other) = delete;

  OwnedVec& operator=(const OwnedVec& other) = delete;

  OwnedVec(OwnedVec&& other);

  OwnedVec& operator=(OwnedVec&& other);
};

/** @brief Create an `OwnedVec` of length `m` distributed over all ranks.
 */
OwnedVec make_Vec(PetscInt m);

/** @brief Create an `OwnedVec` with the same structure (length, distribution over ranks)
 *         as `other`. Values are not copied.
 */
OwnedVec make_Vec_with_structure(Vec other);

/** @brief Create an `OwnedVec` with the same structure and values as `other`.
 */
OwnedVec make_Vec_copy(Vec other);

/** @brief Convert an array `vs` of `OwnedVec`s to the corresponding array of `Vec`s.
 *         `vs` still maintains ownership.
 */
template <std::size_t len>
std::array<Vec, len> as_Vec_array(std::array<OwnedVec, len>& vs) {
  std::array<Vec, len> raw_vs;

  for (std::size_t i = 0; i < len; i++) {
    raw_vs.at(i) = vs.at(i).v;
  }

  return raw_vs;
}

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
OwnedVec scatter_to_all(Vec v);

/** @brief Construct a vector u = \sum_d coeffs(d) * vs(d).
 *  @pre The length of coeffs and vs should be at least 1.
 *  @pre All vectors in vs should have the same length.
 */
template <std::size_t len>
OwnedVec Vec_from_sum_const(std::array<PetscScalar, len> coeffs, std::array<OwnedVec, len>& vs) {
  static_assert(len > 0, "must have at least 1 Vec for Vec_from_sum_const");

  auto u = make_Vec_with_structure(vs.at(0).v);
  PetscErrorCode ierr = VecSet(u.v, 0.0);CHKERRXX(ierr);

  ierr = VecMAXPY(u.v, len, coeffs.data(), as_Vec_array(vs).data());CHKERRXX(ierr);

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
OwnedVec vector_elem_apply(Vec v_in, F f) {
  PetscInt v_in_size;
  PetscErrorCode ierr = VecGetSize(v_in, &v_in_size);CHKERRXX(ierr);

  auto v_out = make_Vec(v_in_size);

  // This node is assigned elements in the range begin <= i < end.
  PetscInt begin_in, end_in;
  ierr = VecGetOwnershipRange(v_in, &begin_in, &end_in);CHKERRXX(ierr);

  PetscInt begin_out, end_out;
  ierr = VecGetOwnershipRange(v_out.v, &begin_out, &end_out);CHKERRXX(ierr);

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
  ierr = VecSetValues(v_out.v, local_in_rows.size(), local_in_rows.data(), local_out_vals.data(), INSERT_VALUES);CHKERRXX(ierr);

  ierr = VecAssemblyBegin(v_out.v);CHKERRXX(ierr);
  ierr = VecAssemblyEnd(v_out.v);CHKERRXX(ierr);

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
std::array<OwnedVec, out_dim> vector_index_apply_multiple(PetscInt N, const F &f) {
  static_assert(out_dim > 0, "Must have at least one output of f");

  std::array<OwnedVec, out_dim> vs;
  for (std::size_t oi = 0; oi < out_dim; oi++) {
    vs.at(oi) = make_Vec(N);
  }

  PetscInt begin, end;
  PetscErrorCode ierr = VecGetOwnershipRange(vs.at(0).v, &begin, &end);CHKERRXX(ierr);

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

    ierr = VecSetValues(vs.at(oi).v, rows.size(), rows.data(), vals.at(oi).data(),
        INSERT_VALUES);CHKERRXX(ierr);

    ierr = VecAssemblyBegin(vs.at(oi).v);CHKERRXX(ierr);
    ierr = VecAssemblyEnd(vs.at(oi).v);CHKERRXX(ierr);
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
OwnedVec vector_index_apply(PetscInt N, const F &f) {
  // This function is a special case of vector_index_apply_multiple where out_dim = 1.
  auto f_multiple = [&f](PetscInt i)->std::array<PetscScalar, 1> {
    return { f(i) };
  };

  OwnedVec v = std::move(vector_index_apply_multiple<1>(N, f_multiple).at(0));

  return v;
}

} // namespace anomtrans

#endif // ANOMTRANS_VEC_H
