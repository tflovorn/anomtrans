#ifndef ANOMTRANS_DISORDER_H
#define ANOMTRANS_DISORDER_H

#include <cmath>
#include <limits>
#include <complex>

namespace anomtrans {

/** @brief Calculate the disorder-averaged on-site diagonal disorder term.
 *  @note There is an extra factor of U0^2/Nk appearing in <UU>. This is
 *        left out here to avoid passing the parameters.
 */
template <typename Hamiltonian>
double on_site_diagonal_disorder(const unsigned int Nbands, const Hamiltonian &H,
    const PetscInt ikm1, const PetscInt ikm2, const PetscInt ikm3,
    const PetscInt ikm4) {
  // Use Kahan summation for sum over band indices.
  std::complex<double> sum(0.0, 0.0);
  std::complex<double> c(0.0, 0.0);
  for (unsigned int i1 = 0; i1 < Nbands; i1++) {
    for (unsigned int i2 = 0; i2 < Nbands; i2++) {
      std::complex<double> contrib = std::conj(H.basis_component(ikm1, i1))
          * H.basis_component(ikm2, i1)
          * std::conj(H.basis_component(ikm3, i2))
          * H.basis_component(ikm4, i2);

      std::complex<double> y = contrib - c;
      std::complex<double> t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
  }

  // After sum, we should get a real number.
  // TODO - sure this is true?
  // TODO - what is appropriate tol value?
  // Nbands = sqrt(Nbands^2) via Kahan expected error.
  // 1 is an appropriate scale here: the basis component vectors are normalized
  // to 1.
  assert(std::abs(sum.imag()) < Nbands*std::numeric_limits<double>::epsilon());

  return sum.real();
}

} // namespace anomtrans

#endif // ANOMTRANS_DISORDER_H
