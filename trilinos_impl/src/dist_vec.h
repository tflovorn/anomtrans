#ifndef ANOMTRANS_DIST_VEC_H
#define ANOMTRANS_DIST_VEC_H

#include <cstdint>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Teuchos_RCP.hpp>

namespace anomtrans {

/** @brief Global ordinal type for use as index into Tpetra vectors and matrices.
 */
using GO = int32_t;

/** @brief Local ordinal type for use as index into the local part of Tpetra
 *         vectors and matrices (i.e. the part mapped to a particular process).
 */
using LO = int32_t;

/** @brief Tpetra distributed vector with our preferred size.
 */
template <typename Scalar>
using DistVec = Tpetra::Vector<Scalar, LO, GO>;

/** @brief Global memory space for DistVec.
 */
template <typename Scalar>
using DistVecMemorySpace = typename DistVec<Scalar>::node_type::memory_space;

/** @brief Type that holds the magnitude of a Tpetra vector with the given
 *         scalar type.
 */
template <typename Scalar>
using DistVecMag = typename DistVec<Scalar>::mag_type;

/** @brief Tpetra mapping from vector indices to processes, using our preferred
 *         vector size.
 */
using Map = Tpetra::Map<LO, GO>;

/** @brief Teuchos reference-counted pointer.
 */
template <typename T>
using RCP = Teuchos::RCP<T>;

/** @brief Tpetra sparse matrix, compressed row storage.
 */
template <typename Scalar>
using CrsMatrix = Tpetra::CrsMatrix<Scalar, LO, GO>;

} // namespace anomtrans

#endif  // ANOMTRANS_DIST_VEC_H
