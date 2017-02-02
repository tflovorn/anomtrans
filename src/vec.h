#ifndef ANOMTRANS_VEC_H
#define ANOMTRANS_VEC_H

#include <cassert>
#include <vector>
#include <tuple>
#include <petscksp.h>

namespace anomtrans {

using IndexValPair = std::tuple<std::vector<PetscInt>, std::vector<PetscScalar>>;

IndexValPair get_local_contents(Vec v);

} // namespace anomtrans

#endif // ANOMTRANS_VEC_H
