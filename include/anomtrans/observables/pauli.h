#ifndef ANOMTRANS_PAULI_H
#define ANOMTRANS_PAULI_H

#include <Eigen/Core>

namespace anomtrans {

std::array<Eigen::Matrix2cd, 3> pauli_matrices();

} // namespace anomtrans

#endif // ANOMTRANS_PAULI_H
