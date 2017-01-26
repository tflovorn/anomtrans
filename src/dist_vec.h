#ifndef ANOMTRANS_DIST_VEC_H
#define ANOMTRANS_DIST_VEC_H

#include <cstdint>
//#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Tpetra_Vector.hpp>

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

/** @brief Type that holds the magnitude of a Tpetra vector with the given
 *         scalar type.
 */
template <typename Scalar>
using DistVecMag = typename DistVec<Scalar>::mag_type;
//using DistVecMag = Teuchos::ScalarTraits<Scalar>::magnitudeType;

/** @brief Tpetra mapping from vector indices to processes, using our preferred
 *         vector size.
 */
using Map = Tpetra::Map<LO, GO>;

} // namespace anomtrans

#endif  // ANOMTRANS_DIST_VEC_H
